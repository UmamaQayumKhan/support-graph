.PHONY: help install dev test lint eval gate ablation run ticket docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## runtime dependencies
	pip install -r requirements.txt

dev: ## dev dependencies
	pip install -r requirements-dev.txt

test: ## 107 tests, offline, no keys needed
	PYTHONPATH=src:tests pytest -q

lint:
	ruff check src eval tests
	mypy src/supportgraph || true

eval: ## failure-mode eval
	LLM_PROVIDER=offline python eval/run_eval.py

gate: ## the CI safety gate
	LLM_PROVIDER=offline python eval/run_eval.py \
	  --gate-escalation 1.0 --gate-injection 1.0 --gate-commitment 0.0

ablation: ## measure what the escalation layer is worth
	LLM_PROVIDER=offline python eval/run_eval.py --no-escalation \
	  --out eval/report-ablation.json

run: ## API on :8000
	PYTHONPATH=src uvicorn supportgraph.api:app --reload --port 8000

ticket: ## demo a ticket: make ticket M="I want a refund"
	PYTHONPATH=src python -m supportgraph.cli ticket "$(M)" --trace

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache eval/report*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
