PACKAGES := axio axio-context-sqlite axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-anthropic axio-transport-codex axio-transport-openai \
            axio-tui axio-tui-guards examples/gas_town examples/agent_swarm

.PHONY: $(PACKAGES) all pytest linter typing test tests test-docs

all: linter typing pytest test-docs

linter:
	@for pkg in $(PACKAGES); do uv run --directory $$pkg ruff check . && uv run --directory $$pkg ruff format --check . || exit 1; done

typing pytest: $(PACKAGES)

$(PACKAGES):
	@uv run --directory $@ mypy .
	@uv run --directory $@ pytest -q

test-docs:
	@uv run --directory docs pytest -q .

test: pytest
tests: pytest
