"""Tests for ModelRegistry.first() and ModelRegistry.last()."""

from __future__ import annotations

import pytest

from axio.models import ModelRegistry, ModelSpec

A = ModelSpec(id="a", input_cost=1.0)
B = ModelSpec(id="b", input_cost=2.0)
C = ModelSpec(id="c", input_cost=3.0)


def test_first_single() -> None:
    reg = ModelRegistry([A])
    assert reg.first() == A


def test_last_single() -> None:
    reg = ModelRegistry([A])
    assert reg.last() == A


def test_first_last_same_for_single_element() -> None:
    reg = ModelRegistry([A])
    assert reg.first() == reg.last()


def test_first_multi() -> None:
    reg = ModelRegistry([A, B, C])
    assert reg.first() == A


def test_last_multi() -> None:
    reg = ModelRegistry([A, B, C])
    assert reg.last() == C


def test_first_empty_raises() -> None:
    reg = ModelRegistry()
    with pytest.raises(IndexError, match="ModelRegistry is empty"):
        reg.first()


def test_last_empty_raises() -> None:
    reg = ModelRegistry()
    with pytest.raises(IndexError, match="ModelRegistry is empty"):
        reg.last()


def test_first_after_by_cost() -> None:
    reg = ModelRegistry([C, A, B])
    cheapest = reg.by_cost()
    assert cheapest.first() == A


def test_last_after_by_cost() -> None:
    reg = ModelRegistry([C, A, B])
    priciest = reg.by_cost(desc=True)
    assert priciest.first() == C
    assert priciest.last() == A
