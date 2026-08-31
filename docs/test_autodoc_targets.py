"""The reference pages point at names that exist."""

from __future__ import annotations

import importlib
import pathlib
import re

#: Directive to whether its target names a module or something inside one.
_DIRECTIVES = {"automodule": True, "autoclass": False, "autofunction": False, "autodata": False}


def _resolves(directive: str, target: str) -> bool:
    try:
        if _DIRECTIVES[directive]:
            importlib.import_module(target)
            return True
        module, _, name = target.rpartition(".")
        return hasattr(importlib.import_module(module), name) if module else False
    except ImportError:
        return False


def test_every_autodoc_target_can_be_imported() -> None:
    # A directive naming something that moved renders an empty page and says nothing about it.
    pattern = re.compile(r"^\.\. (automodule|autoclass|autofunction|autodata):: ([\w.]+)", re.M)
    missing = [
        f"{page.name}: {target}"
        for page in pathlib.Path(__file__).parent.rglob("*.md")
        for directive, target in pattern.findall(page.read_text())
        if not _resolves(directive, target)
    ]

    assert not missing, missing
