"""Tests for axio.field: FieldInfo, Field, StrictStr, bare_type, get_field_info."""

from __future__ import annotations

from typing import Annotated

import pytest

from axio.field import MISSING, Field, FieldInfo, StrictStr, bare_type, get_field_info


class TestMissingSentinel:
    def test_repr(self) -> None:
        assert repr(MISSING) == "MISSING"

    def test_bool_false(self) -> None:
        assert not MISSING

    def test_singleton(self) -> None:
        assert MISSING is MISSING


class TestFieldInfo:
    def test_defaults(self) -> None:
        fi = FieldInfo()
        assert fi.description == ""
        assert fi.default is MISSING
        assert fi.ge is None
        assert fi.le is None
        assert fi.strict is False

    def test_ge_passes(self) -> None:
        fi = FieldInfo(ge=0)
        fi.validate(0, "x", int)
        fi.validate(10, "x", int)

    def test_ge_fails(self) -> None:
        fi = FieldInfo(ge=5)
        with pytest.raises(ValueError, match="must be >= 5"):
            fi.validate(4, "x", int)

    def test_le_passes(self) -> None:
        fi = FieldInfo(le=100)
        fi.validate(100, "x", int)
        fi.validate(0, "x", int)

    def test_le_fails(self) -> None:
        fi = FieldInfo(le=10)
        with pytest.raises(ValueError, match="must be <= 10"):
            fi.validate(11, "x", int)

    def test_ge_and_le_range(self) -> None:
        fi = FieldInfo(ge=1, le=10)
        fi.validate(5, "x", int)
        with pytest.raises(ValueError):
            fi.validate(0, "x", int)
        with pytest.raises(ValueError):
            fi.validate(11, "x", int)

    def test_strict_rejects_wrong_type(self) -> None:
        fi = FieldInfo(strict=True)
        with pytest.raises(TypeError, match="requires str"):
            fi.validate(42, "name", str)

    def test_strict_accepts_correct_type(self) -> None:
        fi = FieldInfo(strict=True)
        fi.validate("hello", "name", str)

    def test_strict_with_annotated_hint(self) -> None:
        fi = FieldInfo(strict=True)
        hint = Annotated[str, FieldInfo(strict=True)]
        fi.validate("ok", "x", hint)
        with pytest.raises(TypeError):
            fi.validate(1, "x", hint)

    def test_frozen(self) -> None:
        fi = FieldInfo(description="x")
        with pytest.raises(Exception):
            fi.description = "y"  # type: ignore[misc]


class TestField:
    def test_returns_fieldinfo(self) -> None:
        fi = Field(description="desc", default=42, ge=1, le=100)
        assert isinstance(fi, FieldInfo)
        assert fi.description == "desc"
        assert fi.default == 42
        assert fi.ge == 1
        assert fi.le == 100

    def test_defaults(self) -> None:
        fi = Field()
        assert fi.description == ""
        assert fi.default is MISSING
        assert fi.ge is None
        assert fi.le is None


class TestStrictStr:
    def test_is_annotated_str(self) -> None:
        fi = get_field_info(StrictStr)
        assert fi is not None
        assert fi.strict is True

    def test_rejects_int_via_fieldinfo(self) -> None:
        fi = get_field_info(StrictStr)
        assert fi is not None
        with pytest.raises(TypeError):
            fi.validate(123, "x", StrictStr)

    def test_accepts_str(self) -> None:
        fi = get_field_info(StrictStr)
        assert fi is not None
        fi.validate("hello", "x", StrictStr)


class TestGetFieldInfo:
    def test_plain_type_returns_none(self) -> None:
        assert get_field_info(str) is None
        assert get_field_info(int) is None

    def test_annotated_with_fieldinfo(self) -> None:
        hint = Annotated[str, Field(description="test")]
        fi = get_field_info(hint)
        assert fi is not None
        assert fi.description == "test"

    def test_annotated_without_fieldinfo(self) -> None:
        hint = Annotated[str, "just a string"]
        assert get_field_info(hint) is None

    def test_annotated_multiple_metadata_picks_fieldinfo(self) -> None:
        hint = Annotated[str, "meta", Field(ge=0)]
        fi = get_field_info(hint)
        assert fi is not None
        assert fi.ge == 0


class TestBareType:
    def test_plain_type(self) -> None:
        assert bare_type(str) is str
        assert bare_type(int) is int

    def test_optional(self) -> None:
        assert bare_type(str | None) is str

    def test_annotated(self) -> None:
        assert bare_type(Annotated[int, Field(ge=0)]) is int

    def test_annotated_optional(self) -> None:
        assert bare_type(Annotated[str | None, Field()]) is str

    def test_union_multiple_returns_object(self) -> None:
        assert bare_type(int | str) is object

    def test_non_type_returns_object(self) -> None:
        assert bare_type(42) is object
