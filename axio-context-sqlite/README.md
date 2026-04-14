# axio-context-sqlite

[![PyPI](https://img.shields.io/pypi/v/axio-context-sqlite)](https://pypi.org/project/axio-context-sqlite/)
[![Python](https://img.shields.io/pypi/pyversions/axio-context-sqlite)](https://pypi.org/project/axio-context-sqlite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SQLite-backed persistent context store for [axio](https://github.com/axio-agent/monorepo).

## Installation

```bash
pip install axio-context-sqlite
```

## Usage

Open a connection with `connect()`, then create a `SQLiteContextStore` bound to a
session. The caller owns the connection and is responsible for closing it.

<!-- name: test_readme_usage -->
```python
import asyncio
import tempfile
import pathlib
from axio_context_sqlite import connect, SQLiteContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        store = SQLiteContextStore(conn, session_id="my-session")
        await store.append(Message(role="user", content=[TextBlock(text="Hello")]))
        history = await store.get_history()
        assert len(history) == 1
    finally:
        await conn.close()

asyncio.run(main())
```

`SQLiteContextStore` implements the `axio.context.ContextStore` ABC and persists
conversation history across process restarts. Multiple sessions can coexist in
the same database file, isolated by `session_id` and `project`.

### Agent integration

<!--
name: test_readme_agent
```python
from axio.testing import StubTransport, make_text_response
transport = StubTransport([make_text_response("Hi!")])
```
-->
<!-- name: test_readme_agent -->
```python
import asyncio
import tempfile
import pathlib
from axio.agent import Agent
from axio_context_sqlite import connect, SQLiteContextStore

async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        ctx = SQLiteContextStore(conn, session_id="main")
        agent = Agent(system="You are helpful.", tools=[], transport=transport)
        result = await agent.run("Hello!", ctx)
        assert result == "Hi!"
    finally:
        await conn.close()

asyncio.run(main())
```

### Listing sessions

<!-- name: test_readme_list_sessions -->
```python
import asyncio
import tempfile
import pathlib
from axio_context_sqlite import connect, SQLiteContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        store = SQLiteContextStore(conn, session_id="main", project="/myproject")
        await store.append(Message(role="user", content=[TextBlock(text="hi")]))
        sessions = await store.list_sessions()
        for s in sessions:
            print(s.session_id, s.preview, s.message_count)
        assert len(sessions) == 1
    finally:
        await conn.close()

asyncio.run(main())
```

### Forking

`fork()` copies the current session's messages into a new session — useful for
branching conversations without affecting the original:

<!-- name: test_readme_fork -->
```python
import asyncio
import tempfile
import pathlib
from axio_context_sqlite import connect, SQLiteContextStore
from axio.messages import Message
from axio.blocks import TextBlock

async def main() -> None:
    conn = await connect(pathlib.Path(tempfile.mkdtemp()) / "chat.db")
    try:
        store = SQLiteContextStore(conn, session_id="main")
        await store.append(Message(role="user", content=[TextBlock(text="original")]))
        branch = await store.fork()
        assert branch.session_id != store.session_id
        assert len(await branch.get_history()) == 1
    finally:
        await conn.close()

asyncio.run(main())
```

## License

MIT
