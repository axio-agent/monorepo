PACKAGES := axio axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-anthropic axio-transport-codex axio-transport-openai \
            axio-tui axio-tui-guards axio-tui-rag

.PHONY: $(PACKAGES) all pytest linter typing test tests

all: linter typing pytest

linter:
	@for pkg in $(PACKAGES); do uv run --directory $$pkg ruff check . && uv run --directory $$pkg ruff format --check . || exit 1; done

typing pytest: $(PACKAGES)

$(PACKAGES):
	@uv run --directory $@ mypy .
	@uv run --directory $@ pytest -vv

test: pytest
tests: pytest
