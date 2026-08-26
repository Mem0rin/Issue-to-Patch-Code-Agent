"""Provider-independent types used by the Agent Kernel."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


def _require_non_blank(value: str, field_name: str) -> None:
    """Reject an empty or whitespace-only string."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_non_blank(self.name, "name")
        _require_non_blank(self.description, "description")


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        _require_non_blank(self.content, "content")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_non_blank(self.call_id, "call_id")
        _require_non_blank(self.tool_name, "tool_name")


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool

    def __post_init__(self) -> None:
        _require_non_blank(self.call_id, "call_id")


@dataclass(frozen=True)
class FinalAnswer:
    content: str

    def __post_init__(self) -> None:
        _require_non_blank(self.content, "content")


Action: TypeAlias = ToolCall | FinalAnswer

HistoryItem: TypeAlias = Message | ToolCall | ToolResult