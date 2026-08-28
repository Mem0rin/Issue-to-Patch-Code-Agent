"""Adapter from JSON-text model responses to Kernel actions."""
"""Kernel - Adapter - Provider"""
import json
from collections.abc import Sequence
from dataclasses import dataclass

from .kernel_types import (
    HistoryItem,
    Message,
    ToolCall,
    ToolResult,
)

# 发给模型的消息，包含角色和内容。
@dataclass(frozen=True)
class ProviderMessage:
    role: str
    content: str


def _encode_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_provider_messages(
    history: Sequence[HistoryItem],
) -> tuple[ProviderMessage, ...]:
    messages: list[ProviderMessage] = []

    for item in history:
        if isinstance(item, Message):
            messages.append(
                ProviderMessage(
                    role=item.role.value,
                    content=item.content,
                )
            )
            continue

        if isinstance(item, ToolCall):
            messages.append(
                ProviderMessage(
                    # TODO 1：模型发出的行动使用什么角色？
                    role="assistant",
                    content=_encode_json({
                        "type": "tool_call",
                        "call_id": item.call_id,
                        "tool_name": item.tool_name,
                        "arguments": item.arguments,
                    }),
                )
            )
            continue

        if isinstance(item, ToolResult):
            messages.append(
                ProviderMessage(
                    # TODO 2：文本模型没有 tool 角色时，
                    # 环境反馈暂时使用什么角色？
                    role="user",
                    content=_encode_json({
                        "type": "tool_result",
                        "call_id": item.call_id,
                        "content": item.content,
                        "is_error": item.is_error,
                    }),
                )
            )
            continue

    return tuple(messages)