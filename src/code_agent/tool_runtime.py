# src/code_agent/tool_runtime.py

from .kernel_types import ToolCall, ToolResult
from .tool_registry import (
    ToolRegistry,
    UnknownToolError,
)


class ToolExecutionError(RuntimeError):
    """An expected, model-visible tool failure."""


class ToolRuntime:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def execute(self, call: ToolCall) -> ToolResult:
        try:
            # TODO 1：按工具名称解析 ToolDefinition
            definition =self._registry.resolve(call.tool_name)
        except UnknownToolError:
            return ToolResult(
                # TODO 2：复制原工具调用的 call_id
                call_id= call.call_id,
                content=f"unknown tool: {call.tool_name}",
                is_error=True,
            )

        try:
            # TODO 3：调用 handler，并传入 arguments
            content = definition.handler(call.arguments)
        except ToolExecutionError as exc:
            return ToolResult(
                call_id=call.call_id,
                # ToolExecutionError 的消息必须是安全且可展示的
                content=str(exc),
                is_error=True,
            )

        return ToolResult(
            call_id=call.call_id,
            # TODO 4：填入 handler 的返回内容
            content=content,
            is_error=False,
        )