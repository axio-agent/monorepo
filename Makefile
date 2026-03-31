PACKAGES := axio axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-codex axio-transport-nebius axio-transport-openai \
            axio-tui axio-tui-guards axio-tui-rag

.PHONY: $(PACKAGES) all pytest linter typing test tests

all: linter typing pytest

linter:
	@uv run ruff check $(PACKAGES)
	@uv run ruff format --check $(PACKAGES)

typing pytest: $(PACKAGES)

$(PACKAGES):
	@uv run --directory $@ mypy .
	@uv run --directory $@ pytest -vv

test: pytest
tests: pytest
