"""Smoke test: every symbol in __all__ must be importable from axio."""

import axio


def test_all_symbols_importable() -> None:
    for name in axio.__all__:
        assert hasattr(axio, name), f"axio.{name} missing"


def test_no_private_names_in_all() -> None:
    for name in axio.__all__:
        assert not name.startswith("_"), f"private name in __all__: {name}"
