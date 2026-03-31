PACKAGES := axio axio-tools-docker axio-tools-local axio-tools-mcp \
            axio-transport-codex axio-transport-nebius axio-transport-openai \
            axio-tui axio-tui-guards axio-tui-rag

.PHONY: lint ruff mypy pytest $(PACKAGES)

lint: ruff mypy

ruff:
	@for pkg in $(PACKAGES); do \
		echo "==> ruff $$pkg"; \
		uv run --directory $$pkg ruff check && \
		uv run --directory $$pkg ruff format --check || exit 1; \
	done

mypy:
	@for pkg in $(PACKAGES); do \
		echo "==> mypy $$pkg"; \
		uv run --directory $$pkg mypy . || exit 1; \
	done

pytest:
	@for pkg in $(PACKAGES); do \
		echo "==> pytest $$pkg"; \
		uv run --directory $$pkg pytest -vv || exit 1; \
	done
