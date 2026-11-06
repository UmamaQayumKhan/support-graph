FROM python:3.12-slim AS builder
WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 agent
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY eval/ ./eval/
COPY data/ ./data/

RUN chown -R agent:agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://localhost:8000/healthz',timeout=4)); \
sys.exit(0 if b.get('status')=='ok' and b.get('kb_passages',0)>0 else 1)"

CMD ["uvicorn", "supportgraph.api:app", "--host", "0.0.0.0", "--port", "8000"]
