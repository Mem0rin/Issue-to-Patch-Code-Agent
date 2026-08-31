from dataclasses import dataclass
from typing import TypeAlias

from .kernel_types import ToolCall, ToolResult
from .tool_registry import (
    InvalidToolArgumentsError,
    ToolDefinition,
    ToolRegistry,
    UnknownToolError,
)


@dataclass(frozen=True)
class PreparedToolCall:
    """A known tool call whose arguments passed validation."""

    call: ToolCall
    definition: ToolDefinition


ToolPreparation: TypeAlias = (
    PreparedToolCall | ToolResult
)


class ToolExecutionError(RuntimeError):
    """An expected, model-visible tool failure."""


class ToolRuntime:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def prepare(
        self,
        call: ToolCall,
    ) -> ToolPreparation:
        try:
            definition = self._registry.resolve(
                call.tool_name
            )
        except UnknownToolError:
            return ToolResult(
                call_id=call.call_id,
                content=f"unknown tool: {call.tool_name}",
                is_error=True,
            )

        try:
            definition.validate_arguments(
                call.arguments
            )
        except InvalidToolArgumentsError as exc:
            return ToolResult(
                call_id=call.call_id,
                content=str(exc),
                is_error=True,
            )

        return PreparedToolCall(
            call=call,
            definition=definition,
        )

    def execute(
        self,
        prepared: PreparedToolCall,
    ) -> ToolResult:
        try:
            content = prepared.definition.handler(
                prepared.call.arguments
            )
        except ToolExecutionError as exc:
            return ToolResult(
                call_id=prepared.call.call_id,
                content=str(exc),
                is_error=True,
            )

        return ToolResult(
            call_id=prepared.call.call_id,
            content=content,
            is_error=False,
        )