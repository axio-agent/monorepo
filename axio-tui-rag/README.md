# axio-tui-rag

[![PyPI](https://img.shields.io/pypi/v/axio-tui-rag)](https://pypi.org/project/axio-tui-rag/)
[![Python](https://img.shields.io/pypi/pyversions/axio-tui-rag)](https://pypi.org/project/axio-tui-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAG (Retrieval-Augmented Generation) plugin for [axio-tui](https://github.com/axio-agent/axio-tui).

Index your local files into a vector store and let the agent search them semantically. Point the agent at a codebase, documentation, or notes — it retrieves relevant chunks without reading every file.

## Features

- **File indexing** — chunk and embed files into a [LanceDB](https://lancedb.github.io/lancedb/) vector store
- **Semantic search** — retrieve the most relevant chunks for a query using embedding similarity
- **Incremental updates** — re-indexing a file replaces its old chunks; unchanged files are skipped
- **Two axio tools** — `index_files` and `semantic_search` registered via entry points

## Installation

```bash
pip install axio-tui-rag
```

Or install as part of the TUI bundle:

```bash
pip install "axio-tui[rag]"
```

## Usage

### Via axio-tui (recommended)

Launch the TUI with RAG enabled — the agent gains two new tools automatically:

```bash
pip install "axio-tui[rag,openai]"
axio
```

Then ask the agent:
- *"Index all Python files in ./src"*
- *"Search for how authentication is implemented"*

### Standalone

```python
from axio import Agent
from axio.context import MemoryContextStore
from axio_transport_openai import OpenAITransport
from axio_tui_rag import IndexFiles, SemanticSearch
from axio.tool import Tool

tools = [
    Tool(name="index_files",    description=IndexFiles.__doc__ or "",    handler=IndexFiles),
    Tool(name="semantic_search", description=SemanticSearch.__doc__ or "", handler=SemanticSearch),
]

agent = Agent(
    system="You are a code assistant. Index files before searching.",
    tools=tools,
    transport=OpenAITransport(api_key="sk-...", model="gpt-4o"),
)
```

## Tools

### `index_files`

Index one or more files into the vector store for later semantic search. Files are chunked and embedded. Re-indexing a file replaces its old chunks.

Parameters:
- `paths: list[str]` — file paths to index (relative to cwd)

### `semantic_search`

Search the vector store for chunks semantically similar to a query. Returns ranked excerpts with file paths and line numbers.

Parameters:
- `query: str` — natural-language search query
- `limit: int` (default: 5) — number of results to return

## Plugin registration

```toml
[project.entry-points."axio.tools"]
index_files     = "axio_tui_rag:IndexFiles"
semantic_search = "axio_tui_rag:SemanticSearch"
```

## Part of the axio ecosystem

[axio](https://github.com/axio-agent/axio) · [axio-tui](https://github.com/axio-agent/axio-tui) · [axio-tui-guards](https://github.com/axio-agent/axio-tui-guards)

## License

MIT
