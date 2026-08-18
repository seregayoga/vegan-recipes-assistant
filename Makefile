.PHONY: setup ingest run status stop test lint generate-evaluation evaluate evaluate-llm

setup:
	cp .env.example .env
	uv sync

install:
	docker compose up -d --wait postgres

ingest:
	docker compose --profile ingest run --rm --build ingest

run:
	docker compose up -d --build --wait app grafana

status:
	docker compose ps

stop:
	docker compose stop

test:
	uv run pytest -q

lint:
	uv run ruff check app.py ingest.py src tests evaluation

generate-evaluation:
	docker compose --profile evaluation run --rm --build evaluate /app/.venv/bin/python evaluation/generate_examples.py

evaluate:
	docker compose --profile evaluation run --rm --build evaluate

evaluate-llm:
	docker compose --profile evaluation run --rm --build evaluate /app/.venv/bin/python evaluation/llm_eval.py
