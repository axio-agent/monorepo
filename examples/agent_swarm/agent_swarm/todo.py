"""SQLite-backed todo list tool for the orchestrator.

The open ``aiosqlite.Connection`` is passed as tool context — same lifetime
pattern as the agent run that owns it.
"""

from __future__ import annotations

from typing import Annotated, Literal

import aiosqlite
from axio.tool import Tool, ToolHandler
from pydantic import Field

STATUS_ICON = {"todo": "○", "in_progress": "◑", "done": "●", "blocked": "✗"}

DDL = """
CREATE TABLE IF NOT EXISTS todos (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    item     TEXT NOT NULL,
    status   TEXT NOT NULL DEFAULT 'todo'
)
"""


class TodoTool(ToolHandler[aiosqlite.Connection]):
    """Manage a todo list to track task progress.
    Use at the start of every step to review what is pending,
    add new tasks as you plan work, and mark items done as work completes.
    Actions:
      list   — show all items with their status
      add    — append a new item (provide `item`)
      update — change item status (provide `id` and `status`)"""

    action: Annotated[
        Literal["list", "add", "update"],
        Field(description="list · add · update"),
    ]
    item: Annotated[str, Field(default="", description="Item text (required for add)")]
    id: Annotated[int, Field(default=0, description="Item ID (required for update)")]
    status: Annotated[
        Literal["todo", "in_progress", "done", "blocked"],
        Field(default="todo", description="New status (required for update)"),
    ]

    async def __call__(self, context: aiosqlite.Connection) -> str:
        db = context

        if self.action == "list":
            async with db.execute("SELECT id, item, status FROM todos ORDER BY id") as cur:
                rows = await cur.fetchall()
            if not rows:
                return "(empty)"
            return "\n".join(f"{STATUS_ICON.get(str(r[2]), '?')} [{r[0]}] {r[1]}  ({r[2]})" for r in rows)

        if self.action == "add":
            cur = await db.execute("INSERT INTO todos (item, status) VALUES (?, 'todo')", (self.item,))
            await db.commit()
            return f"Added [{cur.lastrowid}]: {self.item}"

        # update
        cur = await db.execute("UPDATE todos SET status = ? WHERE id = ?", (self.status, self.id))
        await db.commit()
        if cur.rowcount:
            return f"[{self.id}] → {self.status}"
        return f"Item {self.id} not found"


def make_todo_tool(db: aiosqlite.Connection, guards: tuple = ()) -> Tool:
    """Create a todo tool with *db* as its context."""
    return Tool(name="todo", description=TodoTool.__doc__ or "", handler=TodoTool, context=db, guards=guards)
