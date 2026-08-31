# tests/test_tool_registry.py

from collections.abc import Mapping

import pytest

from code_agent.kernel_types import ToolSpec
from code_agent.tool_registry import (
    DuplicateToolError,
    ToolDefinition,
    ToolRegistry,
    UnknownToolError,
)

def _accept_any_arguments(
    arguments: Mapping[str, object],
) -> None:
    _ = arguments


def _handler(arguments: Mapping[str, object]) -> str:
    return f"received: {arguments}"


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        spec=ToolSpec(
            name=name,
            description=f"Execute {name}.",
            input_schema={"type": "object"},
        ),
        validate_arguments=_accept_any_arguments,
        handler=_handler,
    )


def test_registry_exposes_specs_and_resolves_definition() -> None:
    definition = _make_tool("read_file")
    registry = ToolRegistry([definition])

    assert registry.specs == (definition.spec,)
    assert registry.resolve("read_file") is definition


def test_registry_rejects_duplicate_names() -> None:
    first = _make_tool("read_file")
    second = _make_tool("read_file")

    with pytest.raises(
        DuplicateToolError,
        match="duplicate tool name: read_file",
    ):
        ToolRegistry([first, second])


def test_registry_rejects_unknown_tool() -> None:
    registry = ToolRegistry([
        _make_tool("read_file"),
    ])

    with pytest.raises(
        UnknownToolError,
        match="unknown tool: delete_everything",
    ):
        registry.resolve("delete_everything")