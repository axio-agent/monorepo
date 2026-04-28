"""Persistent notes tool for the orchestrator.

Notes are plain-text files saved to ``.axio-swarm/notes/``.  The directory
``Path`` is passed as tool context - fixed for the lifetime of a run.

File format::

    description: one-line summary shown in list output
    ---
    body content here...
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from axio import CONTEXT, Field, Tool

SEPARATOR = "---"


def _check_file(path: Path) -> str | None:
    """Return an error string if *path* is a symlink (deletes it) or is not a regular file.

    Returns None when the path is safe to use (a real file or does not exist yet).
    """
    if path.is_symlink():
        path.unlink()
        return f"Rejected symlink at '{path.name}' (deleted)"
    if path.exists() and not path.is_file():
        return f"'{path.name}' is not a regular file"
    return None


def _parse(text: str) -> tuple[str, str]:
    """Return (description, body) from a note file."""
    first, sep, rest = text.partition(f"\n{SEPARATOR}\n")
    if sep and first.startswith("description:"):
        return first[len("description:") :].strip(), rest
    return "", text


def _format(description: str, body: str) -> str:
    return f"description: {description.strip()}\n{SEPARATOR}\n{body.strip()}"


async def notes(
    action: Annotated[
        Literal["list", "read", "write", "append", "drop"],
        Field(description="list · read · write · append · drop"),
    ],
    name: Annotated[
        str, Field(default="", description="Note name without extension (required for read/write/append)")
    ] = "",
    description: Annotated[
        str,
        Field(
            default="",
            description="One-line summary shown in list output (required for write/append when creating)",
        ),
    ] = "",
    content: Annotated[
        str, Field(default="", description="Body text to write or append (required for write/append)")
    ] = "",
) -> str:
    """Save and retrieve notes in the swarm's internal notes directory.

    Use notes to persist findings, decisions, summaries, and project state
    across iterations - anything you want to remember later in the session.

    Actions:
      list   - list all saved notes with their descriptions
      read   - read a note by name (returns description + body)
      write  - create or overwrite a note (name, description, and content required)
      append - append text to a note's body (creates if missing; description required on create)
      drop   - remove a note by name
    """
    notes_dir: Path = CONTEXT.get()
    notes_dir.mkdir(parents=True, exist_ok=True)

    path = notes_dir / f"{name}.md" if name else None

    match action:
        case "list":
            files = sorted(notes_dir.glob("*.md"))
            if not files:
                return "(no notes yet)"
            lines = []
            for f in files:
                if f.is_symlink():
                    f.unlink()
                    continue
                if not f.is_file():
                    continue
                desc, _ = _parse(f.read_text())
                lines.append(f"{f.stem} - {desc}" if desc else f.stem)
            return "\n".join(lines) if lines else "(no notes yet)"

        case "read":
            if path is None:
                return "name is required"
            if err := _check_file(path):
                return err
            if not path.exists():
                return f"Note '{name}' not found"
            desc, body = _parse(path.read_text())
            return f"[{desc}]\n\n{body}" if desc else body

        case "write":
            if path is None:
                return "name is required"
            if not description:
                return "description is required for write"
            if err := _check_file(path):
                return err
            path.write_text(_format(description, content))
            return f"Note '{name}' saved"

        case "append":
            if path is None:
                return "name is required"
            if err := _check_file(path):
                return err
            if path.exists():
                desc, body = _parse(path.read_text())
            else:
                if not description:
                    return "description is required when creating a new note"
                desc, body = description, ""
            merged = (body.rstrip() + "\n\n" + content.strip()).strip()
            path.write_text(_format(desc, merged))
            return f"Note '{name}' updated"

        case "drop":
            if path is None:
                return "name is required"
            path.unlink(missing_ok=True)
            return f"Note '{name}' dropped"

    return "unknown action"


def make_notes_tool(workspace: Path, guards: tuple = ()) -> Tool[Path]:
    """Create a notes tool that stores files in *workspace*/.axio-swarm/notes/."""
    return Tool(
        name="notes",
        handler=notes,
        context=workspace / ".axio-swarm" / "notes",
        guards=guards,
    )
