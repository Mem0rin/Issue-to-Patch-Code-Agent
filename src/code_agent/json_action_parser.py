"""Strict parsing of JSON text into Kernel actions."""

import json
from collections.abc import Mapping
from typing import cast

from .kernel_types import (
    Action,
    FinalAnswer,
    ToolCall,
)
from .model_adapter import InvalidModelOutputError


_TOOL_CALL_FIELDS = frozenset({
    "type",
    "call_id",
    "tool_name",
    "arguments",
})

_FINAL_ANSWER_FIELDS = frozenset({
    "type",
    "content",
})


def parse_json_action(raw_text: str) -> Action:
    """Parse untrusted JSON text into one Kernel action."""
    try:
        decoded: object = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise InvalidModelOutputError("model output must be valid JSON") from error

    if not isinstance(decoded, dict):
        raise InvalidModelOutputError("model output must be an object")

    payload = cast(dict[str, object], decoded)
    action_type = payload.get("type")

    if action_type == "tool_call":
        return _parse_tool_call(payload)

    if action_type == "final_answer":
        return _parse_final_answer(payload)

    raise InvalidModelOutputError("unsupported action type")


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    action_name: str,
) -> None:
    """Reject missing or additional action fields."""
    if set(payload) != expected:
        raise InvalidModelOutputError(
            f"{action_name} fields do not match the contract"
        )


def _parse_tool_call(
    payload: Mapping[str, object],
) -> ToolCall:
    """Parse one strict tool-call object."""
    _require_exact_fields(
        payload,
        _TOOL_CALL_FIELDS,
        "tool_call",
    )

    call_id = payload["call_id"]
    tool_name = payload["tool_name"]
    arguments = payload["arguments"]

    if not isinstance(call_id, str):
        raise InvalidModelOutputError("call_id must be a string")

    if not isinstance(tool_name, str):
        raise InvalidModelOutputError("tool_name must be a string")

    if not isinstance(arguments, dict):
        raise InvalidModelOutputError("arguments must be an object")

    arguments_mapping = cast(
        dict[str, object],
        arguments,
    )

    try:
        return ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments_mapping,
        )
    except ValueError as error:
        raise InvalidModelOutputError(str(error)) from error


def _parse_final_answer(
    payload: Mapping[str, object],
) -> FinalAnswer:
    """Parse one strict final-answer object."""
    _require_exact_fields(
        payload,
        _FINAL_ANSWER_FIELDS,
        "final_answer",
    )

    content = payload["content"]

    if not isinstance(content, str):
        raise InvalidModelOutputError("content must be a string")

    try:
        return FinalAnswer(content=content)
    except ValueError as error:
        raise InvalidModelOutputError(str(error)) from error
