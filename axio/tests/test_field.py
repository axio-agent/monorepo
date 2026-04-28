"""Tests for axio.field: FieldInfo, Field, StrictStr, bare_type, get_field_info."""

from __future__ import annotations

from typing import Annotated, Literal

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

    def test_literal_valid(self) -> None:
        fi = FieldInfo()
        fi.validate("left", "dir", Literal["left", "right"])

    def test_literal_invalid(self) -> None:
        fi = FieldInfo()
        with pytest.raises(ValueError, match="must be one of"):
            fi.validate("up", "dir", Literal["left", "right"])

    def test_literal_in_annotated(self) -> None:
        fi = FieldInfo(description="direction")
        with pytest.raises(ValueError):
            fi.validate("up", "dir", Annotated[Literal["left", "right"], fi])

    def test_nonstrict_type_check_rejects_wrong_type(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires int"):
            fi.validate("oops", "count", int)

    def test_nonstrict_type_check_accepts_correct_type(self) -> None:
        fi = FieldInfo()
        fi.validate(42, "count", int)

    def test_nonstrict_none_allowed_for_optional(self) -> None:
        fi = FieldInfo()
        fi.validate(None, "value", str | None)  # None is valid for Optional

    def test_nonstrict_wrong_type_for_optional(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires str"):
            fi.validate(42, "value", str | None)


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

    def test_generic_list(self) -> None:
        assert bare_type(list[int]) is list

    def test_generic_dict(self) -> None:
        assert bare_type(dict[str, int]) is dict

    def test_generic_optional_list(self) -> None:
        assert bare_type(list[str] | None) is list


class TestValidateExtended:
    def test_list_type_accepts_list(self) -> None:
        fi = FieldInfo()
        fi.validate([1, 2, 3], "items", list[int])

    def test_list_type_rejects_non_list(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires list"):
            fi.validate("not a list", "items", list[int])

    def test_float_accepts_int(self) -> None:
        fi = FieldInfo()
        fi.validate(1, "value", float)  # int is a valid JSON "number"

    def test_float_accepts_float(self) -> None:
        fi = FieldInfo()
        fi.validate(1.5, "value", float)

    def test_strict_float_rejects_int(self) -> None:
        fi = FieldInfo(strict=True)
        with pytest.raises(TypeError, match="requires float"):
            fi.validate(1, "value", float)

    def test_optional_literal_accepts_none(self) -> None:
        fi = FieldInfo()
        fi.validate(None, "status", Literal["active", "inactive"] | None)

    def test_optional_literal_accepts_valid_value(self) -> None:
        fi = FieldInfo()
        fi.validate("active", "status", Literal["active", "inactive"] | None)

    def test_optional_literal_rejects_invalid_value(self) -> None:
        fi = FieldInfo()
        with pytest.raises(ValueError, match="must be one of"):
            fi.validate("deleted", "status", Literal["active", "inactive"] | None)

    def test_non_optional_literal_rejects_none(self) -> None:
        fi = FieldInfo()
        with pytest.raises(ValueError, match="must be one of"):
            fi.validate(None, "status", Literal["active", "inactive"])

    def test_bool_rejected_for_int_hint(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires int, got bool"):
            fi.validate(True, "count", int)

    def test_bool_rejected_for_float_hint(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires float, got bool"):
            fi.validate(False, "value", float)

    def test_bool_accepted_for_bool_hint(self) -> None:
        fi = FieldInfo()
        fi.validate(True, "flag", bool)

    def test_strict_bool_rejected_for_int(self) -> None:
        fi = FieldInfo(strict=True)
        with pytest.raises(TypeError, match="requires int, got bool"):
            fi.validate(True, "count", int)

    def test_list_str_rejects_int_element(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="element 1 requires str"):
            fi.validate(["ok", 42], "tags", list[str])

    def test_list_int_accepts_all_ints(self) -> None:
        fi = FieldInfo()
        fi.validate([1, 2, 3], "nums", list[int])

    def test_list_no_type_args_skips_element_check(self) -> None:
        fi = FieldInfo()
        fi.validate([1, "x", None], "items", list)  # bare list - no element check

    # list item consistency with scalar rules
    def test_list_float_accepts_int_element(self) -> None:
        fi = FieldInfo()
        fi.validate([1, 2.5], "values", list[float])  # int is valid JSON number

    def test_list_bool_rejected_for_int_element(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="element 0 requires int"):
            fi.validate([True], "flags", list[int])

    def test_list_optional_item_accepts_none(self) -> None:
        fi = FieldInfo()
        fi.validate(["ok", None], "tags", list[str | None])

    def test_list_optional_item_rejects_wrong_type(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="element 1 requires str"):
            fi.validate(["ok", 42], "tags", list[str | None])


class TestMultiBranchUnion:
    def test_str_or_int_accepts_str(self) -> None:
        fi = FieldInfo()
        fi.validate("hello", "x", str | int)

    def test_str_or_int_accepts_int(self) -> None:
        fi = FieldInfo()
        fi.validate(42, "x", str | int)

    def test_str_or_int_rejects_list(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires str | int"):
            fi.validate([], "x", str | int)

    def test_str_or_int_or_none_accepts_none(self) -> None:
        fi = FieldInfo()
        fi.validate(None, "x", str | int | None)

    def test_str_or_int_or_none_accepts_str(self) -> None:
        fi = FieldInfo()
        fi.validate("hi", "x", str | int | None)

    def test_str_or_int_or_none_rejects_list(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires str | int"):
            fi.validate([], "x", str | int | None)

    def test_bool_or_str_accepts_bool(self) -> None:
        fi = FieldInfo()
        fi.validate(True, "x", bool | str)

    def test_int_or_str_rejects_bool(self) -> None:
        fi = FieldInfo()
        with pytest.raises(TypeError, match="requires int | str"):
            fi.validate(True, "x", int | str)
