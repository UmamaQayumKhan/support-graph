"""Policy knowledge base with BM25 retrieval.

Lexical-only retrieval is a deliberate choice here, not a shortcut. The KB is a
handful of policy documents and the queries that matter are near-verbatim
("overage", "SAML_AUDIENCE_MISMATCH", "net-30"), which is where BM25 is
strongest. It also means the agent needs no embedding API, so the whole graph
runs offline and the retrieval step costs nothing.

If this KB grew past a few hundred documents, the honest answer is to add a
dense arm and fuse -- which is exactly what the companion AtlasRAG service does,
and this module's interface is deliberately the same shape so it could be
swapped for a call to that service.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = frozenset(
    "a an the of to in for on at by is are was were be been do does did and or but "
    "if then than that this these those it its as with from i we you my our your".split()
)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if "_" in tok:
            out.extend(p for p in tok.split("_") if p)
    return [t for t in out if t not in _STOP and len(t) > 1]


@dataclass
class Passage:
    doc_id: str
    section: str
    text: str
    ref: str

    def label(self) -> str:
        title = self.doc_id.replace("-", " ").title()
        return f"{title} | {self.section}" if self.section else title


@dataclass
class KnowledgeBase:
    passages: list[Passage] = field(default_factory=list)
    k1: float = 1.5
    b: float = 0.75
    _counts: list[Counter] = field(default_factory=list)
    _lens: list[int] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _avg: float = 0.0

    def add(self, passage: Passage) -> None:
        toks = tokenize(f"{passage.section} {passage.text}")
        counts = Counter(toks)
        self.passages.append(passage)
        self._counts.append(counts)
        self._lens.append(len(toks))
        for term in counts:
            self._df[term] += 1
        self._avg = sum(self._lens) / max(1, len(self._lens))

    def _idf(self, term: str) -> float:
        n = len(self.passages)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5)) if df else 0.0

    def search(self, query: str, top_k: int = 4) -> list[tuple[Passage, float]]:
        terms = tokenize(query)
        if not terms or not self.passages:
            return []
        scored: list[tuple[Passage, float]] = []
        for i, counts in enumerate(self._counts):
            dl = self._lens[i] or 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avg or 1))
                    score += self._idf(term) * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append((self.passages[i], score))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.passages)


_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_kb(kb_dir: str | Path) -> KnowledgeBase:
    """One passage per markdown section, so a citation points at a section
    rather than a whole document."""
    kb = KnowledgeBase()
    root = Path(kb_dir)
    if not root.exists():
        return kb

    for path in sorted(root.glob("*.md")):
        doc_id = path.stem
        stack: list[tuple[int, str]] = []
        body: list[str] = []

        def emit() -> None:
            content = "\n".join(body).strip()
            if not content:
                return
            section = " > ".join(name for _, name in stack[1:]) or (
                stack[0][1] if stack else ""
            )
            kb.add(
                Passage(
                    doc_id=doc_id,
                    section=section,
                    text=content,
                    ref=f"{doc_id}#{_slug(section) or 'top'}",
                )
            )

        for line in path.read_text(encoding="utf-8").splitlines():
            m = _HEADING.match(line)
            if m:
                emit()
                body = []
                level = len(m.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, m.group(2)))
            else:
                body.append(line)
        emit()

    return kb


def passages_to_context(hits: list[tuple[Passage, float]]) -> tuple[str, list[dict[str, Any]]]:
    """Numbered context plus the citation list, so the drafting node and the
    verifier agree on what [1] refers to."""
    blocks: list[str] = []
    cites: list[dict[str, Any]] = []
    for i, (passage, score) in enumerate(hits, start=1):
        blocks.append(f"[{i}] {passage.label()}\n{passage.text}")
        cites.append(
            {
                "n": i,
                "doc_id": passage.doc_id,
                "section": passage.section,
                "ref": passage.ref,
                "label": passage.label(),
                "score": round(score, 4),
            }
        )
    return "\n\n---\n\n".join(blocks), cites
