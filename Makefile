.PHONY: fmt lint typecheck test all

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

lint:
	uv run ruff check src/ tests/

typecheck:
	uv run mypy src/ tests/

test:
	uv run pytest tests/ -v

all: fmt lint typecheck test
