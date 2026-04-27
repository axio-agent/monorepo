"""Tests for the notes tool."""

from pathlib import Path

import pytest

from agent_swarm.notes import NotesTool, _format, _parse, make_notes_tool

# ---------------------------------------------------------------------------
# _parse / _format helpers
# ---------------------------------------------------------------------------


def test_parse_with_header():
    text = "description: my summary\n---\nbody content"
    desc, body = _parse(text)
    assert desc == "my summary"
    assert body == "body content"


def test_parse_without_header_backward_compat():
    """Old notes without a description header must still be readable."""
    text = "just plain content\nno header here"
    desc, body = _parse(text)
    assert desc == ""
    assert body == text


def test_parse_non_description_first_line():
    """A file whose first line doesn't start with 'description:' is treated as headerless."""
    text = "# Heading\n---\nsome body"
    desc, body = _parse(text)
    assert desc == ""
    assert body == text


def test_format_roundtrip():
    desc, body = _parse(_format("my note", "the body"))
    assert desc == "my note"
    assert body == "the body"


# ---------------------------------------------------------------------------
# NotesTool actions
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    return tmp_path / "notes"


async def _call(action: str, notes_dir: Path, **kwargs: object) -> str:
    tool = NotesTool(action=action, **kwargs)  # type: ignore[arg-type]
    result = await tool(notes_dir)
    assert isinstance(result, str)
    return result


class TestList:
    async def test_empty(self, notes_dir: Path) -> None:
        assert await _call("list", notes_dir) == "(no notes yet)"

    async def test_shows_name_and_description(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="domain", description="Domain findings", content="body")
        result = await _call("list", notes_dir)
        assert result == "domain - Domain findings"

    async def test_multiple_notes_sorted(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="zzz", description="Last", content="")
        await _call("write", notes_dir, name="aaa", description="First", content="")
        lines = (await _call("list", notes_dir)).splitlines()
        assert lines[0].startswith("aaa")
        assert lines[1].startswith("zzz")

    async def test_legacy_note_no_description(self, notes_dir: Path) -> None:
        """Old notes without a description header must appear in list without crashing."""
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "old.md").write_text("plain old content")
        result = await _call("list", notes_dir)
        assert "old" in result


class TestWrite:
    async def test_creates_note(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="n", description="desc", content="body")
        assert (notes_dir / "n.md").exists()

    async def test_requires_description(self, notes_dir: Path) -> None:
        result = await _call("write", notes_dir, name="n", content="body")
        assert "description" in result.lower()
        assert not (notes_dir / "n.md").exists()

    async def test_requires_name(self, notes_dir: Path) -> None:
        result = await _call("write", notes_dir, description="d", content="body")
        assert "name" in result.lower()

    async def test_overwrites_existing(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="n", description="v1", content="old")
        await _call("write", notes_dir, name="n", description="v2", content="new")
        desc, body = _parse((notes_dir / "n.md").read_text())
        assert desc == "v2"
        assert body == "new"


class TestRead:
    async def test_returns_description_and_body(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="n", description="summary", content="the body")
        result = await _call("read", notes_dir, name="n")
        assert "summary" in result
        assert "the body" in result

    async def test_missing_note(self, notes_dir: Path) -> None:
        result = await _call("read", notes_dir, name="missing")
        assert "not found" in result

    async def test_requires_name(self, notes_dir: Path) -> None:
        result = await _call("read", notes_dir)
        assert "name" in result.lower()

    async def test_legacy_note_readable(self, notes_dir: Path) -> None:
        """Old notes without a header must be returned as-is."""
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "old.md").write_text("legacy content")
        result = await _call("read", notes_dir, name="old")
        assert "legacy content" in result


class TestAppend:
    async def test_appends_to_existing(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="n", description="d", content="first")
        await _call("append", notes_dir, name="n", content="second")
        _, body = _parse((notes_dir / "n.md").read_text())
        assert "first" in body
        assert "second" in body

    async def test_preserves_description_on_append(self, notes_dir: Path) -> None:
        await _call("write", notes_dir, name="n", description="original desc", content="v1")
        await _call("append", notes_dir, name="n", content="v2")
        desc, _ = _parse((notes_dir / "n.md").read_text())
        assert desc == "original desc"

    async def test_create_via_append_requires_description(self, notes_dir: Path) -> None:
        result = await _call("append", notes_dir, name="new", content="body")
        assert "description" in result.lower()
        assert not (notes_dir / "new.md").exists()

    async def test_create_via_append_with_description(self, notes_dir: Path) -> None:
        await _call("append", notes_dir, name="new", description="created via append", content="body")
        desc, body = _parse((notes_dir / "new.md").read_text())
        assert desc == "created via append"
        assert "body" in body

    async def test_append_to_legacy_note(self, notes_dir: Path) -> None:
        """Appending to an old note without a header must not lose existing content."""
        notes_dir.mkdir(parents=True, exist_ok=True)
        (notes_dir / "old.md").write_text("legacy content")
        await _call("append", notes_dir, name="old", content="new section")
        text = (notes_dir / "old.md").read_text()
        assert "legacy content" in text
        assert "new section" in text


class TestMakeNotesTool:
    def test_context_path(self, tmp_path: Path) -> None:
        tool = make_notes_tool(tmp_path)
        assert tool.context == tmp_path / ".axio-swarm" / "notes"


# ---------------------------------------------------------------------------
# Symlink protection
# ---------------------------------------------------------------------------


class TestSymlinkProtection:
    """Notes tool must reject symlinks and never follow them to host files."""

    async def test_read_rejects_symlink(self, notes_dir: Path, tmp_path: Path) -> None:
        notes_dir.mkdir(parents=True, exist_ok=True)
        secret = tmp_path / "secret.txt"
        secret.write_text("secret content")
        (notes_dir / "n.md").symlink_to(secret)

        result = await _call("read", notes_dir, name="n")

        assert "secret content" not in result
        assert not (notes_dir / "n.md").is_symlink(), "symlink should be deleted"

    async def test_write_rejects_symlink(self, notes_dir: Path, tmp_path: Path) -> None:
        notes_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_path / "target.txt"
        target.write_text("original")
        (notes_dir / "n.md").symlink_to(target)

        result = await _call("write", notes_dir, name="n", description="d", content="injected")

        assert "Rejected" in result
        assert target.read_text() == "original", "target must not be overwritten"
        assert not (notes_dir / "n.md").is_symlink(), "symlink should be deleted"

    async def test_append_rejects_symlink(self, notes_dir: Path, tmp_path: Path) -> None:
        notes_dir.mkdir(parents=True, exist_ok=True)
        target = tmp_path / "target.txt"
        target.write_text("original")
        (notes_dir / "n.md").symlink_to(target)

        result = await _call("append", notes_dir, name="n", description="d", content="injected")

        assert "Rejected" in result
        assert target.read_text() == "original", "target must not be modified"
        assert not (notes_dir / "n.md").is_symlink(), "symlink should be deleted"

    async def test_list_skips_symlinks(self, notes_dir: Path, tmp_path: Path) -> None:
        notes_dir.mkdir(parents=True, exist_ok=True)
        await _call("write", notes_dir, name="real", description="real note", content="body")
        secret = tmp_path / "secret.md"
        secret.write_text("description: injected\n---\nbad")
        (notes_dir / "evil.md").symlink_to(secret)

        result = await _call("list", notes_dir)

        assert "real" in result
        assert "injected" not in result
        assert not (notes_dir / "evil.md").is_symlink(), "symlink should be deleted"
