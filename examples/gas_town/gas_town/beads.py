"""SQLite-backed Bead store for the Gas Town example.

Beads are the atomic unit of work in Gas Town.  An open ``aiosqlite.Connection``
is passed directly as tool context - the connection is opened and closed in
``run_gastown()``, same pattern as an ``aiohttp.ClientSession``.
"""

from __future__ import annotations

from typing import Annotated, Literal

import aiosqlite
from axio import CONTEXT, Field, Tool

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
    """Return a formatted per-bead status listing including notes."""
    async with db.execute("SELECT id, title, status, assignee, notes FROM beads ORDER BY id") as cur:
        rows = await cur.fetchall()
    if not rows:
        return "(no beads)"
    lines = []
    for bead_id, title, status, assignee, notes in rows:
        icon = STATUS_ICON.get(str(status), "?")
        asgn = f" → {assignee}" if assignee else ""
        lines.append(f"{icon} [{bead_id}] {title}  ({status}){asgn}")
        if notes:
            for note_line in notes.strip().splitlines():
                lines.append(f"     {note_line}")
    return "\n".join(lines)


async def get_bead(db: aiosqlite.Connection, bead_id: int) -> tuple[int, str, str, str, str] | None:
    """Return (id, title, status, assignee, notes) or None if not found."""
    async with db.execute("SELECT id, title, status, assignee, notes FROM beads WHERE id=?", (bead_id,)) as cur:
        row = await cur.fetchone()
    return row  # type: ignore[return-value]


async def has_active_beads(db: aiosqlite.Connection) -> bool:
    """Return True if any beads are actively being worked by a polecat (in_progress)."""
    async with db.execute("SELECT 1 FROM beads WHERE status = 'in_progress' LIMIT 1") as cur:
        return await cur.fetchone() is not None


async def get_unreviewed_closed_beads(db: aiosqlite.Connection) -> list[tuple[int, str]]:
    """Return (id, title) for closed beads not yet reviewed by the Refinery."""
    async with db.execute(
        "SELECT id, title FROM beads WHERE status='closed' AND notes NOT LIKE '%refinery:reviewed%'"
    ) as cur:
        rows = await cur.fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


async def mark_in_progress(db: aiosqlite.Connection, bead_id: int, assignee: str = "polecat") -> None:
    """Mark a bead as in_progress with the given assignee."""
    await db.execute(
        "UPDATE beads SET status='in_progress', assignee=? WHERE id=?",
        (assignee, bead_id),
    )
    await db.commit()


# ---------------------------------------------------------------------------
# bead_tool - given to all agents; context is the open aiosqlite.Connection
# ---------------------------------------------------------------------------


async def bead(
    action: Annotated[
        Literal["list", "create", "update", "close", "note"],
        Field(description="list · create · update · close · note"),
    ],
    id: Annotated[int, Field(default=0, description="Bead ID (required for update / close / note)")] = 0,
    title: Annotated[str, Field(default="", description="Bead title (required for create)")] = "",
    status: Annotated[BStatus | None, Field(default=None, description="New status (for update)")] = None,
    assignee: Annotated[str, Field(default="", description="Assignee name (for update)")] = "",
    notes: Annotated[str, Field(default="", description="Note text to append (for note)")] = "",
) -> str:
    """Manage the shared bead store (convoy issue tracker). Data is persisted to SQLite.

    Actions:
      list   - show all beads with their status
      create - create a new bead (provide `title`)
      update - change status or assignee (provide `id` plus `status` or `assignee`)
      close  - mark a bead done (provide `id`)
      note   - append a note to a bead (provide `id` and `notes`)"""
    db: aiosqlite.Connection = CONTEXT.get()

    if action == "list":
        return await bead_summary(db)

    if action == "create":
        cur = await db.execute("INSERT INTO beads (title) VALUES (?)", (title,))
        await db.commit()
        return f"Created bead [{cur.lastrowid}]: {title}"

    row = await get_bead(db, id)
    if row is None:
        return f"Bead {id} not found"
    bead_id, _, cur_status, cur_assignee, cur_notes = row

    if action == "update":
        new_status = status if status is not None else cur_status
        new_assignee = assignee or cur_assignee
        await db.execute(
            "UPDATE beads SET status=?, assignee=? WHERE id=?",
            (new_status, new_assignee, bead_id),
        )
        await db.commit()
        return f"[{bead_id}] updated - status={new_status} assignee={new_assignee or '(none)'}"

    if action == "close":
        await db.execute("UPDATE beads SET status='closed' WHERE id=?", (bead_id,))
        await db.commit()
        return f"[{bead_id}] closed"

    # note
    merged = (cur_notes + "\n" + notes).strip()
    await db.execute("UPDATE beads SET notes=? WHERE id=?", (merged, bead_id))
    await db.commit()
    return f"[{bead_id}] note appended"


def make_bead_tool(db: aiosqlite.Connection, guards: tuple = ()) -> Tool:
    """Create a bead tool with *db* as its context."""
    return Tool(name="bead", handler=bead, context=db, guards=guards)
