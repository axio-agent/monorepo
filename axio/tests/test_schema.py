"""Tests for axio.schema: property_schema and build_tool_schema."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Optional

from axio.field import Field, FieldInfo, StrictStr
from axio.schema import build_tool_schema, property_schema


class TestPropertySchemaPrimitives:
    def test_str(self) -> None:
        assert property_schema(str) == {"type": "string"}

    def test_int(self) -> None:
        assert property_schema(int) == {"type": "integer"}

    def test_float(self) -> None:
        assert property_schema(float) == {"type": "number"}

    def test_bool(self) -> None:
        assert property_schema(bool) == {"type": "boolean"}

    def test_unknown_type_returns_empty(self) -> None:
        class Custom:
            pass

        assert property_schema(Custom) == {}

    def test_none_type_returns_empty(self) -> None:
        assert property_schema(type(None)) == {}


class TestPropertySchemaCollections:
    def test_list_of_str(self) -> None:
        assert property_schema(list[str]) == {"type": "array", "items": {"type": "string"}}

    def test_list_of_int(self) -> None:
        assert property_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}

    def test_list_of_list(self) -> None:
        assert property_schema(list[list[str]]) == {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
        }

    def test_list_unparameterised(self) -> None:
        # bare `list` has no origin so it falls through to the unknown path
        assert property_schema(list) == {}

    def test_dict_bare(self) -> None:
        assert property_schema(dict) == {"type": "object"}

    def test_dict_parameterised(self) -> None:
        assert property_schema(dict[str, Any]) == {"type": "object"}

    def test_list_of_optional_str(self) -> None:
        # list[str | None] — items are nullable strings, emitted as anyOf
        assert property_schema(list[str | None]) == {
            "type": "array",
            "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        }

    def test_list_of_literal(self) -> None:
        assert property_schema(list[Literal["a", "b"]]) == {
            "type": "array",
            "items": {"enum": ["a", "b"]},
        }


class TestPropertySchemaOptional:
    def test_optional_str(self) -> None:
        assert property_schema(str | None) == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_optional_int(self) -> None:
        assert property_schema(int | None) == {"anyOf": [{"type": "integer"}, {"type": "null"}]}

    def test_typing_optional(self) -> None:
        assert property_schema(Optional[str]) == {"anyOf": [{"type": "string"}, {"type": "null"}]}  # noqa: UP045

    def test_union_multiple_non_none(self) -> None:
        result = property_schema(str | int)
        assert result == {"anyOf": [{"type": "string"}, {"type": "integer"}]}

    def test_union_none_first(self) -> None:
        # None | str — None is filtered out; str + null emitted as anyOf
        result = property_schema(None | str)
        assert result == {"anyOf": [{"type": "string"}, {"type": "null"}]}


class TestPropertySchemaLiteral:
    def test_literal_strings(self) -> None:
        assert property_schema(Literal["a", "b", "c"]) == {"enum": ["a", "b", "c"]}

    def test_literal_ints(self) -> None:
        assert property_schema(Literal[1, 2, 3]) == {"enum": [1, 2, 3]}

    def test_literal_single(self) -> None:
        assert property_schema(Literal["only"]) == {"enum": ["only"]}


class TestPropertySchemaAnnotated:
    def test_description_added(self) -> None:
        result = property_schema(Annotated[str, Field(description="a search term")])
        assert result == {"type": "string", "description": "a search term"}

    def test_empty_description_not_added(self) -> None:
        result = property_schema(Annotated[str, Field()])
        assert "description" not in result

    def test_ge_becomes_minimum(self) -> None:
        result = property_schema(Annotated[int, Field(ge=1)])
        assert result["minimum"] == 1
        assert "maximum" not in result

    def test_le_becomes_maximum(self) -> None:
        result = property_schema(Annotated[int, Field(le=100)])
        assert result["maximum"] == 100
        assert "minimum" not in result

    def test_ge_and_le(self) -> None:
        result = property_schema(Annotated[float, Field(ge=0.0, le=1.0)])
        assert result == {"type": "number", "minimum": 0.0, "maximum": 1.0}

    def test_annotated_optional(self) -> None:
        result = property_schema(Annotated[str | None, Field(description="optional")])
        assert result == {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "optional"}

    def test_annotated_list(self) -> None:
        result = property_schema(Annotated[list[int], Field(description="ids")])
        assert result == {"type": "array", "items": {"type": "integer"}, "description": "ids"}

    def test_annotated_optional_list(self) -> None:
        # Annotated[list[str] | None, Field(...)] — optional list with metadata; null variant preserved
        result = property_schema(Annotated[list[str] | None, Field(description="tags")])
        assert result == {
            "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}],
            "description": "tags",
        }

    def test_annotated_with_bounds_on_optional_int(self) -> None:
        result = property_schema(Annotated[int | None, Field(ge=0, le=100)])
        assert result == {"anyOf": [{"type": "integer"}, {"type": "null"}], "minimum": 0, "maximum": 100}

    def test_non_fieldinfo_annotation_ignored(self) -> None:
        # Annotated with a non-FieldInfo metadata item - should not crash
        result = property_schema(Annotated[str, "some doc string"])
        assert result == {"type": "string"}

    def test_strict_str_schema(self) -> None:
        result = property_schema(StrictStr)
        assert result == {"type": "string"}

    def test_strict_flag_not_in_schema(self) -> None:
        # strict=True is a runtime-only constraint; must not bleed into the JSON schema
        fi = FieldInfo(strict=True)
        result = property_schema(Annotated[str, fi])
        assert "strict" not in result


class TestPropertySchemaNoTitles:
    """No 'title' key must appear anywhere in a generated schema."""

    def test_primitive_no_title(self) -> None:
        assert "title" not in property_schema(str)

    def test_annotated_no_title(self) -> None:
        result = property_schema(Annotated[str, Field(description="x")])
        assert "title" not in result

    def test_list_no_title(self) -> None:
        result = property_schema(list[str])
        assert "title" not in result
        assert "title" not in result.get("items", {})


class TestBuildToolSchemaShape:
    def test_top_level_type_is_object(self) -> None:
        async def f(x: str) -> str:
            return x

        schema = build_tool_schema(f)
        assert schema["type"] == "object"

    def test_properties_present(self) -> None:
        async def f(a: str, b: int) -> str:
            return a

        schema = build_tool_schema(f)
        assert set(schema["properties"]) == {"a", "b"}

    def test_no_required_key_when_all_optional(self) -> None:
        async def f(x: str = "default") -> str:
            return x

        schema = build_tool_schema(f)
        assert "required" not in schema

    def test_no_required_key_for_no_params(self) -> None:
        async def f() -> str:
            return "ok"

        schema = build_tool_schema(f)
        assert "required" not in schema
        assert schema["properties"] == {}

    def test_no_title_at_top_level(self) -> None:
        async def f(x: str) -> str:
            return x

        assert "title" not in build_tool_schema(f)

    def test_no_title_in_property(self) -> None:
        async def f(x: str) -> str:
            return x

        schema = build_tool_schema(f)
        assert "title" not in schema["properties"]["x"]


class TestBuildToolSchemaRequired:
    def test_required_param_in_required(self) -> None:
        async def f(query: str) -> str:
            return query

        schema = build_tool_schema(f)
        assert "query" in schema["required"]

    def test_py_default_not_required(self) -> None:
        async def f(query: str, limit: int = 10) -> str:
            return query

        schema = build_tool_schema(f)
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_field_default_not_required(self) -> None:
        async def f(q: str, limit: Annotated[int, Field(default=10)]) -> str:
            return q

        schema = build_tool_schema(f)
        assert "q" in schema["required"]
        assert "limit" not in schema.get("required", [])

    def test_optional_type_still_required_without_default(self) -> None:
        # str | None with no default → still required (caller must pass it)
        async def f(value: str | None) -> str:
            return str(value)

        schema = build_tool_schema(f)
        assert "value" in schema["required"]

    def test_multiple_required(self) -> None:
        async def f(a: str, b: int, c: float) -> str:
            return a

        schema = build_tool_schema(f)
        assert set(schema["required"]) == {"a", "b", "c"}


class TestBuildToolSchemaExclusions:
    def test_return_annotation_excluded(self) -> None:
        async def f(x: str) -> str:
            return x

        schema = build_tool_schema(f)
        assert "return" not in schema["properties"]

    def test_private_param_excluded(self) -> None:
        # Parameters starting with _ should be excluded
        async def f(x: str) -> str:
            return x

        # Inject a private hint manually to test exclusion
        f.__annotations__["_internal"] = str
        schema = build_tool_schema(f)
        assert "_internal" not in schema["properties"]

    def test_classvar_excluded(self) -> None:
        class MyHandler:
            class_field: ClassVar[str] = "meta"
            name: str

            async def __call__(self) -> str:
                return self.name

        schema = build_tool_schema(MyHandler)
        assert "class_field" not in schema["properties"]
        assert "name" in schema["properties"]


class TestBuildToolSchemaTypeMapping:
    """Verify each Python type produces the correct JSON schema type."""

    def test_str_field(self) -> None:
        async def f(x: str) -> str:
            return x

        assert build_tool_schema(f)["properties"]["x"] == {"type": "string"}

    def test_int_field(self) -> None:
        async def f(x: int) -> str:
            return str(x)

        assert build_tool_schema(f)["properties"]["x"] == {"type": "integer"}

    def test_float_field(self) -> None:
        async def f(x: float) -> str:
            return str(x)

        assert build_tool_schema(f)["properties"]["x"] == {"type": "number"}

    def test_bool_field(self) -> None:
        async def f(x: bool) -> str:
            return str(x)

        assert build_tool_schema(f)["properties"]["x"] == {"type": "boolean"}

    def test_list_field(self) -> None:
        async def f(items: list[str]) -> str:
            return str(items)

        assert build_tool_schema(f)["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_dict_field(self) -> None:
        async def f(data: dict[str, Any]) -> str:
            return str(data)

        assert build_tool_schema(f)["properties"]["data"] == {"type": "object"}

    def test_optional_field_nullable(self) -> None:
        async def f(value: str | None) -> str:
            return str(value)

        assert build_tool_schema(f)["properties"]["value"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}

    def test_literal_field(self) -> None:
        async def f(direction: Literal["left", "right"]) -> str:
            return direction

        assert build_tool_schema(f)["properties"]["direction"] == {"enum": ["left", "right"]}

    def test_annotated_field_with_description(self) -> None:
        async def f(query: Annotated[str, Field(description="the search query")]) -> str:
            return query

        prop = build_tool_schema(f)["properties"]["query"]
        assert prop["type"] == "string"
        assert prop["description"] == "the search query"

    def test_annotated_field_with_bounds(self) -> None:
        async def f(count: Annotated[int, Field(ge=1, le=50)]) -> str:
            return str(count)

        prop = build_tool_schema(f)["properties"]["count"]
        assert prop == {"type": "integer", "minimum": 1, "maximum": 50}

    def test_strict_str_field(self) -> None:
        async def f(name: StrictStr) -> str:
            return name

        assert build_tool_schema(f)["properties"]["name"] == {"type": "string"}

    def test_precomputed_hints_used(self) -> None:
        async def f(x: str) -> str:
            return x

        hints = {"x": int}  # deliberately wrong to prove hints= is used
        schema = build_tool_schema(f, hints=hints)
        assert schema["properties"]["x"] == {"type": "integer"}

    def test_py_default_emitted_in_schema(self) -> None:
        async def f(limit: int = 10) -> str:
            return str(limit)

        prop = build_tool_schema(f)["properties"]["limit"]
        assert prop.get("default") == 10

    def test_field_default_emitted_in_schema(self) -> None:
        async def f(limit: Annotated[int, Field(default=20)]) -> str:
            return str(limit)

        prop = build_tool_schema(f)["properties"]["limit"]
        assert prop.get("default") == 20

    def test_no_default_key_when_no_default(self) -> None:
        async def f(query: str) -> str:
            return query

        prop = build_tool_schema(f)["properties"]["query"]
        assert "default" not in prop
