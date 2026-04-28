# axio-agent

[![GitHub](https://img.shields.io/badge/github-mosquito%2Faxio--agent-181717?logo=github&logoColor=white)](https://github.com/mosquito/axio-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://python.org)

Monorepo for **axio** (*Asynchronous eXtensible Intelligent Orchestration*) - a minimal,
streaming-first, protocol-driven foundation for LLM-powered agents.

---

## Packages

| Package | PyPI | Description |
|---|---|---|
| [`axio`](axio/) | [![PyPI](https://img.shields.io/pypi/v/axio)](https://pypi.org/project/axio/) | Core library: Agent, Tool, Transport protocol, ContextStore, PermissionGuard |
| [`axio-tui`](axio-tui/) | [![PyPI](https://img.shields.io/pypi/v/axio-tui)](https://pypi.org/project/axio-tui/) | Textual TUI application + SQLite context store + plugin discovery |
| [`axio-transport-anthropic`](axio-transport-anthropic/) | [![PyPI](https://img.shields.io/pypi/v/axio-transport-anthropic)](https://pypi.org/project/axio-transport-anthropic/) | Anthropic Claude transport with prompt caching |
| [`axio-transport-openai`](axio-transport-openai/) | [![PyPI](https://img.shields.io/pypi/v/axio-transport-openai)](https://pypi.org/project/axio-transport-openai/) | OpenAI-compatible transport (OpenAI, Nebius, OpenRouter, custom) |
| [`axio-transport-codex`](axio-transport-codex/) | [![PyPI](https://img.shields.io/pypi/v/axio-transport-codex)](https://pypi.org/project/axio-transport-codex/) | ChatGPT OAuth transport via Responses API |
| [`axio-tools-local`](axio-tools-local/) | [![PyPI](https://img.shields.io/pypi/v/axio-tools-local)](https://pypi.org/project/axio-tools-local/) | File, shell, and Python execution tools |
| [`axio-tools-mcp`](axio-tools-mcp/) | [![PyPI](https://img.shields.io/pypi/v/axio-tools-mcp)](https://pypi.org/project/axio-tools-mcp/) | MCP server bridge |
| [`axio-tools-docker`](axio-tools-docker/) | [![PyPI](https://img.shields.io/pypi/v/axio-tools-docker)](https://pypi.org/project/axio-tools-docker/) | Docker sandbox tool provider |
| [`axio-context-sqlite`](axio-context-sqlite/) | [![PyPI](https://img.shields.io/pypi/v/axio-context-sqlite)](https://pypi.org/project/axio-context-sqlite/) | SQLite-backed persistent context store |
| [`axio-tui-guards`](axio-tui-guards/) | [![PyPI](https://img.shields.io/pypi/v/axio-tui-guards)](https://pypi.org/project/axio-tui-guards/) | PathGuard + LLMGuard permission plugins |

---

## Using axio

### Install the library

```bash
pip install axio
```

Add transports and tools as needed:

```bash
pip install axio-transport-anthropic axio-transport-openai
pip install axio-tools-local axio-tools-mcp
```

### Install the TUI

The recommended way is an isolated installation with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "axio-tui[anthropic,openai,codex,local,mcp,guards]"
```

Or with pip:

```bash
pip install "axio-tui[all]"
```

The `axio` CLI entry point is provided by `axio-tui`.

---

## Developing

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) - used for the workspace, virtual environments, and running tools

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and sync

```bash
git clone https://github.com/mosquito/axio-agent.git
cd axio-agent
uv sync --all-packages
```

`uv sync --all-packages` installs every workspace member and their dev dependencies into a single shared virtual environment at `.venv/`. All local packages resolve to their workspace sources automatically - no `pip install -e` needed.

### How the uv workspace works

The root `pyproject.toml` declares a [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/):

```toml
[tool.uv.workspace]
members = ["axio", "axio-tui", "axio-transport-anthropic", ...]

[tool.uv.sources]
axio                     = { workspace = true }
axio-transport-anthropic = { workspace = true }
# ...
```

Each `members` directory is a self-contained Python package with its own `pyproject.toml`, `src/` layout, and `tests/`. They share one `uv.lock` and one `.venv`. When you edit code in `axio/src/`, it is immediately visible to every other package in the workspace without reinstalling.

To run a command inside a specific package's context (e.g. to pick up that package's test configuration):

```bash
uv run --directory axio-transport-anthropic pytest
```

### Make targets

All day-to-day tasks go through `make`. Never call `uv run pytest` or ruff directly at the repo root.

```bash
make              # lint + type-check + tests for all packages + doc tests
make linter       # ruff check + ruff format --check on all packages
make typing       # mypy --strict on all packages
make pytest       # pytest on all packages
make test-docs    # markdown-pytest on docs/
```

Run a single package:

```bash
make PACKAGES=axio-transport-anthropic
```

Run doc tests for a single file:

```bash
uv run --directory docs pytest -v guides/best-practices.md
```

### Code style

- **Formatter / linter**: [ruff](https://docs.astral.sh/ruff/), line length 119, `py312` target
- **Type checker**: [mypy](https://mypy.readthedocs.io/) strict mode
- All public APIs must pass `mypy --strict`

### Doc tests

Documentation lives in `docs/` and is tested with [markdown-pytest](https://github.com/mosquito/markdown-pytest). Code blocks in `.md` files are annotated with HTML comments:

```markdown
<!-- name: test_my_example -->
```python
import asyncio
# ...
```
```

Hidden setup blocks (stubs that shouldn't appear in the rendered docs):

```markdown
<!--
name: test_my_example
```python
# hidden setup code - not visible in rendered docs
```
-->
```

### Adding a new package

1. Create `axio-<name>/pyproject.toml` with `[build-system]` (hatchling), `src/` layout, and a `[dependency-groups] dev` section matching the other packages.
2. Add the package name to `[tool.uv.workspace] members` and `[tool.uv.sources]` in the root `pyproject.toml`.
3. Add the package to `PACKAGES` in `Makefile`.
4. Run `uv sync --all-packages` to update `uv.lock`.

---

## Repository layout

```
axio-agent/
├── pyproject.toml          # workspace root - members + shared uv.sources
├── uv.lock                 # single lockfile for the whole workspace
├── Makefile                # lint / type / test targets
├── examples/               # runnable example scripts
├── docs/                   # Sphinx + markdown-pytest documentation
│   ├── conf.py
│   ├── pyproject.toml
│   └── guides/
├── axio/                   # core library
├── axio-tui/               # TUI application
├── axio-transport-*/       # transport implementations
├── axio-tools-*/           # tool providers
├── axio-context-sqlite/    # SQLite context store
└── axio-tui-guards/        # permission guard plugins
```

---

## License

MIT
