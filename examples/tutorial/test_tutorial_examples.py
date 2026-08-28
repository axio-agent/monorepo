from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


TUTORIAL_ROOT = Path(__file__).parent
REPOSITORY_ROOT = TUTORIAL_ROOT.parents[1]
LESSON_ROOT = REPOSITORY_ROOT / "docs" / "agent-harness"
DOCUMENTATION_ROOT = REPOSITORY_ROOT / "docs"
EXAMPLES = tuple(
    path
    for path in sorted(TUTORIAL_ROOT.glob("*.py"))
    if path.name not in {Path(__file__).name}
)
LESSONS = tuple(sorted(LESSON_ROOT.glob("*.md")))
PUBLIC_DOCUMENTATION = tuple(
    path
    for path in sorted(DOCUMENTATION_ROOT.rglob("*"))
    if path.suffix in {".md", ".rst"} and "_build" not in path.parts
)
LEGACY_DOMAIN_PATTERN = re.compile(
    r"tic" + r"ket|(?<![A-Z0-9_])A" + r"X-(?:\d+|XX)(?![A-Z0-9_])",
    re.IGNORECASE,
)
MARKER_PATTERN = re.compile(
    r"^[ \t]*# \[docs:(start|end)-([A-Za-z0-9_-]+)\]$",
    re.MULTILINE,
)
LITERALINCLUDE_PATTERN = re.compile(
    r"^```\{literalinclude\}\s+(?P<target>\S+)\s*$\n"
    r"(?P<options>.*?)^```$",
    re.MULTILINE | re.DOTALL,
)
DOWNLOAD_PATTERN = re.compile(
    r"\{download\}`[^`\n]*<(?P<target>[^>\n]+)>`",
)
START_MARKER_PATTERN = re.compile(r"# \[docs:start-([A-Za-z0-9_-]+)\]")
END_MARKER_PATTERN = re.compile(r"# \[docs:end-([A-Za-z0-9_-]+)\]")


def _directive_option(options: str, name: str) -> str:
    match = re.search(
        rf"^:{re.escape(name)}:\s+(?P<value>.+)$",
        options,
        re.MULTILINE,
    )
    assert match is not None, f"literalinclude is missing :{name}:"
    return match.group("value").strip()


def _marker_option(options: str, name: str) -> str:
    value = _directive_option(options, name)
    assert value.startswith('"') and value.endswith('"'), (
        f":{name}: must quote its marker"
    )
    return value[1:-1]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_documentation_markers_are_balanced(example: Path) -> None:
    markers = MARKER_PATTERN.finditer(example.read_text())
    active_marker: str | None = None
    completed_markers: set[str] = set()

    for marker in markers:
        boundary, name = marker.groups()
        if boundary == "start":
            assert active_marker is None, (
                f"{name!r} overlaps open marker {active_marker!r}"
            )
            assert name not in completed_markers, f"duplicate marker {name!r}"
            active_marker = name
            continue

        assert active_marker is not None, f"end marker {name!r} has no start"
        assert name == active_marker, (
            f"end marker {name!r} closes open marker {active_marker!r}"
        )
        completed_markers.add(name)
        active_marker = None

    assert completed_markers, f"{example.name} has no documentation markers"
    assert active_marker is None, f"start marker {active_marker!r} has no end"


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda path: path.stem)
def test_lesson_sources_match_documentation_contract(lesson: Path) -> None:
    lesson_text = lesson.read_text()
    expected_source = TUTORIAL_ROOT / f"{lesson.stem.replace('-', '_')}.py"
    expected_target = f"../../examples/tutorial/{expected_source.name}"
    expected_caption = f"examples/tutorial/{expected_source.name}"

    assert expected_source.exists(), f"missing source for {lesson.name}"
    source_text = expected_source.read_text()

    downloads = list(DOWNLOAD_PATTERN.finditer(lesson_text))
    assert len(downloads) == 1, f"{lesson.name} must have one download"
    assert downloads[0].group("target") == expected_target

    includes = list(LITERALINCLUDE_PATTERN.finditer(lesson_text))
    assert includes, f"{lesson.name} has no literalinclude"
    referenced_markers: set[str] = set()

    for include in includes:
        assert include.group("target") == expected_target
        options = include.group("options")
        assert _directive_option(options, "language") == "python"
        assert _directive_option(options, "caption") == expected_caption

        start_marker = _marker_option(options, "start-after")
        end_marker = _marker_option(options, "end-before")
        start_match = START_MARKER_PATTERN.fullmatch(start_marker)
        end_match = END_MARKER_PATTERN.fullmatch(end_marker)
        assert start_match is not None, f"invalid start marker {start_marker!r}"
        assert end_match is not None, f"invalid end marker {end_marker!r}"
        assert start_match.group(1) == end_match.group(1), (
            f"marker pair does not match: {start_marker!r}, {end_marker!r}"
        )

        marker_name = start_match.group(1)
        assert marker_name not in referenced_markers, (
            f"{lesson.name} includes marker {marker_name!r} more than once"
        )
        referenced_markers.add(marker_name)
        assert source_text.count(start_marker) == 1
        assert source_text.count(end_marker) == 1
        assert source_text.index(start_marker) < source_text.index(end_marker)

    defined_markers = {
        name
        for boundary, name in MARKER_PATTERN.findall(source_text)
        if boundary == "start"
    }
    assert referenced_markers == defined_markers, (
        f"{lesson.name} references {sorted(referenced_markers)}, "
        f"but {expected_source.name} defines {sorted(defined_markers)}"
    )


def test_every_tutorial_source_has_one_lesson() -> None:
    expected_sources = {
        TUTORIAL_ROOT / f"{lesson.stem.replace('-', '_')}.py" for lesson in LESSONS
    }
    assert set(EXAMPLES) == expected_sources


@pytest.mark.parametrize(
    "source",
    (*PUBLIC_DOCUMENTATION, *EXAMPLES),
    ids=lambda path: str(path.relative_to(REPOSITORY_ROOT)),
)
def test_public_material_uses_generic_example_domain(source: Path) -> None:
    match = LEGACY_DOMAIN_PATTERN.search(source.read_text())
    assert match is None, f"legacy example domain remains in {source}:{match.start()}"


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.stem)
def test_complete_example_runs(example: Path, tmp_path: Path) -> None:
    working_directory = tmp_path / example.stem
    working_directory.mkdir()
    environment = os.environ.copy()
    environment["AXIO_TUTORIAL_DATA_DIR"] = str(working_directory)

    completed = subprocess.run(
        [sys.executable, str(example)],
        cwd=working_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, (
        f"{example.name} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
