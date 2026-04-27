"""SQLite-backed Bead store for the Gas Town example.

Beads are the atomic unit of work in Gas Town.  An open ``aiosqlite.Connection``
is passed directly as tool context — the connection is opened and closed in
``run_gastown()``, same pattern as an ``aiohttp.ClientSession``.
"""

from __future__ import annotations

from typing import Annotated, Literal

import aiosqlite
from axio.tool import Tool, ToolHandler
from pydantic import ConfigDict, Field

BStatus = Literal["open", "in_progress", "closed", "blocked"]

STATUS_ICON: dict[str, str] = {"open": "○", "in_progress": "◑", "closed": "●", "blocked": "✗"}

DDL = """
    CREATE TABLE IF NOT EXISTS beads (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        title    TEXT    NOT NULL,
        status   TEXT    NOT NULL DEFAULT 'open',
        assignee TEXT    NOT NULL DEFAULT '',
        notes    TEXT    NOT NULL DEFAULT ''
    )
"""


# ---------------------------------------------------------------------------
# Low-level helpers (used by swarm.py)
# ---------------------------------------------------------------------------


async def bead_summary(db: aiosqlite.Connection) -> str:
    """Return a formatted one-line-per-bead status listing."""
    async with db.execute("SELECT id, title, status, assignee FROM beads ORDER BY id") as cur:
        rows = await cur.fetchall()
    if not rows:
        return "(no beads)"
    lines = []
    for bead_id, title, status, assignee in rows:
        icon = STATUS_ICON.get(str(status), "?")
        asgn = f" → {assignee}" if assignee else ""
        lines.append(f"{icon} [{bead_id}] {title}  ({status}){asgn}")
    return "\n".join(lines)


async def get_bead(db: aiosqlite.Connection, bead_id: int) -> tuple[int, str, str, str, str] | None:
    """Return (id, title, status, assignee, notes) or None if not found."""
    async with db.execute("SELECT id, title, status, assignee, notes FROM beads WHERE id=?", (bead_id,)) as cur:
        row = await cur.fetchone()
    return row  # type: ignore[return-value]


async def mark_in_progress(db: aiosqlite.Connection, bead_id: int, assignee: str = "polecat") -> None:
    """Mark a bead as in_progress with the given assignee."""
    await db.execute(
        "UPDATE beads SET status='in_progress', assignee=? WHERE id=?",
        (assignee, bead_id),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# BeadTool — given to all agents; context is the open aiosqlite.Connection
# ---------------------------------------------------------------------------


class BeadTool(ToolHandler[aiosqlite.Connection]):
    """Manage the shared bead store (convoy issue tracker). Data is persisted to SQLite.

    Actions:
      list   — show all beads with their status
      create — create a new bead (provide `title`)
      update — change status or assignee (provide `id` plus `status` or `assignee`)
      close  — mark a bead done (provide `id`)
      note   — append a note to a bead (provide `id` and `notes`)"""

    model_config = ConfigDict(validate_default=True)

    action: Annotated[
        Literal["list", "create", "update", "close", "note"],
        Field(description="list · create · update · close · note"),
    ]
    id: Annotated[int, Field(default=0, description="Bead ID (required for update / close / note)")]
    title: Annotated[str, Field(default="", description="Bead title (required for create)")]
    status: Annotated[BStatus | None, Field(default=None, description="New status (for update)")]
    assignee: Annotated[str, Field(default="", description="Assignee name (for update)")]
    notes: Annotated[str, Field(default="", description="Note text to append (for note)")]

    async def __call__(self, context: aiosqlite.Connection) -> str:
        db = context

        if self.action == "list":
            return await bead_summary(db)

        if self.action == "create":
            cur = await db.execute("INSERT INTO beads (title) VALUES (?)", (self.title,))
            await db.commit()
            return f"Created bead [{cur.lastrowid}]: {self.title}"

        row = await get_bead(db, self.id)
        if row is None:
            return f"Bead {self.id} not found"
        bead_id, _, cur_status, cur_assignee, cur_notes = row

        if self.action == "update":
            new_status = self.status if self.status is not None else cur_status
            new_assignee = self.assignee or cur_assignee
            await db.execute(
                "UPDATE beads SET status=?, assignee=? WHERE id=?",
                (new_status, new_assignee, bead_id),
            )
            await db.commit()
            return f"[{bead_id}] updated — status={new_status} assignee={new_assignee or '(none)'}"

        if self.action == "close":
            await db.execute("UPDATE beads SET status='closed' WHERE id=?", (bead_id,))
            await db.commit()
            return f"[{bead_id}] closed"

        # note
        merged = (cur_notes + "\n" + self.notes).strip()
        await db.execute("UPDATE beads SET notes=? WHERE id=?", (merged, bead_id))
        await db.commit()
        return f"[{bead_id}] note appended"


def make_bead_tool(db: aiosqlite.Connection, guards: tuple = ()) -> Tool:
    """Create a bead tool with *db* as its context."""
    return Tool(name="bead", description=BeadTool.__doc__ or "", handler=BeadTool, context=db, guards=guards)
