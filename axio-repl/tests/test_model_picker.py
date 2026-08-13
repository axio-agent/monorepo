from typing import Any

from axio.models import ModelRegistry, ModelSpec

from axio_repl import _choose_model, _columnise


class _Transport:
    def __init__(self, ids: list[str]) -> None:
        self.models = ModelRegistry(ModelSpec(id=i) for i in ids)


def _pick(ids: list[str], arg: str) -> Any:
    return _choose_model(_Transport(ids), arg)


def test_full_id_wins() -> None:
    chosen = _pick(["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro-0813"], "deepseek/deepseek-v4-pro")
    assert chosen is not None and chosen.id == "deepseek/deepseek-v4-pro"


def test_bare_name_beats_a_longer_substring_match() -> None:
    # Both ids contain the typed text, but only one *is* it once the vendor
    # prefix is dropped — that is an exact choice, not an ambiguous one.
    chosen = _pick(["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro-0813"], "deepseek-v4-pro")
    assert chosen is not None and chosen.id == "deepseek/deepseek-v4-pro"


def test_single_substring_match_still_works() -> None:
    chosen = _pick(["openai/gpt-5.6", "anthropic/claude"], "gpt-5")
    assert chosen is not None and chosen.id == "openai/gpt-5.6"


def test_genuinely_ambiguous_returns_nothing() -> None:
    assert _pick(["a/x-mini", "a/x-nano"], "x-") is None


def test_no_match_returns_nothing() -> None:
    assert _pick(["a/x"], "zzz") is None


def test_columnise_fills_downwards() -> None:
    # ls-style: reading down the first column gives the first items.
    # Width 8 with 3-wide cells leaves room for two columns, so four items
    # become two rows read downwards.
    lines = _columnise(["a", "b", "c", "d"], width=8, gap=2)
    assert lines == ["a  c", "b  d"]


def test_columnise_falls_back_to_one_column_when_narrow() -> None:
    lines = _columnise(["aaaaaaaa", "bbbbbbbb"], width=4)
    assert lines == ["aaaaaaaa", "bbbbbbbb"]


def test_columnise_handles_empty() -> None:
    assert _columnise([], width=80) == []
