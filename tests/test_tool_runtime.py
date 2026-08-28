# tests/test_tool_runtime.py
import pytest
from collections.abc import Mapping

from code_agent.kernel_types import (
    ToolCall,
    ToolResult,
    ToolSpec,
)
from code_agent.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)
from code_agent.tool_runtime import (
    ToolExecutionError,
    ToolRuntime,
)


def test_runtime_executes_registered_tool() -> None:
    received: list[Mapping[str, object]] = []

    def handler(arguments: Mapping[str, object]) -> str:
        received.append(arguments)
        return "README contents"

    definition = ToolDefinition(
        spec=ToolSpec(
            name="read_file",
            description="Read a workspace file.",
            input_schema={"type": "object"},
        ),
        handler=handler,
    )
    runtime = ToolRuntime(ToolRegistry([definition]))
    call = ToolCall(
        call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )

    result = runtime.execute(call)

    assert result == ToolResult(
        call_id="call_1",
        content="README contents",
        is_error=False,
    )
    assert received == [{"path": "README.md"}]


def test_runtime_returns_error_for_unknown_tool() -> None:
    runtime = ToolRuntime(ToolRegistry([]))
    call = ToolCall(
        call_id="call_2",
        tool_name="delete_everything",
        arguments={},
    )

    result = runtime.execute(call)

    assert result == ToolResult(
        call_id="call_2",
        content="unknown tool: delete_everything",
        is_error=True,
    )

def test_runtime_converts_expected_tool_error() -> None:
    def handler(
        arguments: Mapping[str, object],
    ) -> str:
        raise ToolExecutionError("file not found")

    definition = ToolDefinition(
        spec=ToolSpec(
            name="read_file",
            description="Read a workspace file.",
            input_schema={"type": "object"},
        ),
        handler=handler,
    )
    runtime = ToolRuntime(ToolRegistry([definition]))
    call = ToolCall(
        call_id="call_3",
        tool_name="read_file",
        arguments={"path": "missing.txt"},
    )

    result = runtime.execute(call)

    assert result == ToolResult(
        call_id="call_3",
        content="file not found",
        is_error=True,
    )


def test_runtime_propagates_unexpected_handler_error() -> None:
    original_error = AttributeError(
        "unexpected handler bug"
    )

    def handler(
        arguments: Mapping[str, object],
    ) -> str:
        raise original_error

    definition = ToolDefinition(
        spec=ToolSpec(
            name="read_file",
            description="Read a workspace file.",
            input_schema={"type": "object"},
        ),
        handler=handler,
    )
    runtime = ToolRuntime(ToolRegistry([definition]))
    call = ToolCall(
        call_id="call_4",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )

    with pytest.raises(AttributeError) as exc_info:
        runtime.execute(call)

    assert exc_info.value is original_error

