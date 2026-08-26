import json

import pytest

from code_agent.json_action_parser import parse_json_action
from code_agent.kernel_types import FinalAnswer, ToolCall
from code_agent.model_adapter import InvalidModelOutputError


def test_parse_tool_call_without_validating_business_arguments() -> None:
    raw_text = json.dumps({
        "type": "tool_call",
        "call_id": "call_1",
        "tool_name": "read_file",
        "arguments": {
            "path": 123,
        },
    })

    action = parse_json_action(raw_text)

    assert action == ToolCall(
        call_id="call_1",
        tool_name="read_file",
        arguments={"path": 123},
    )


def test_parse_final_answer() -> None:
    raw_text = json.dumps({
        "type": "final_answer",
        "content": "finish the loop",
    })

    action = parse_json_action(raw_text)

    assert action == FinalAnswer(content="finish the loop")


@pytest.mark.parametrize(
    ("raw_text", "expected_message"),
    [
        ("not JSON", "model output must be valid JSON"),
        (json.dumps([1, 2]), "model output must be an object"),
        (json.dumps({"type": "unknown"}), "unsupported action type"),
        (
            json.dumps({
                "type": "tool_call",
                "call_id": "call_1",
                "tool_name": "read_file",
            }),
            "tool_call fields do not match the contract",
        ),
        (
            json.dumps({
                "type": "tool_call",
                "call_id": 123,
                "tool_name": "read_file",
                "arguments": {},
            }),
            "call_id must be a string",
        ),
        (
            json.dumps({
                "type": "tool_call",
                "call_id": "   ",
                "tool_name": "read_file",
                "arguments": {},
            }),
            "call_id must not be blank",
        ),
        (
            json.dumps({
                "type": "tool_call",
                "call_id": "call_1",
                "tool_name": "read_file",
                "arguments": {},
                "unexpected": True,
            }),
            "tool_call fields do not match the contract",
        ),
        (
            json.dumps({
                "type": "tool_call",
                "call_id": "call_1",
                "tool_name": 123,
                "arguments": {},
            }),
            "tool_name must be a string",
        ),
        (
            json.dumps({
                "type": "tool_call",
                "call_id": "call_1",
                "tool_name": "read_file",
                "arguments": [],
            }),
            "arguments must be an object",
        ),
        (
            json.dumps({
                "type": "final_answer",
                "content": 123,
            }),
            "content must be a string",
        ),
        (
            json.dumps({
                "type": "final_answer",
                "content": "   ",
            }),
            "content must not be blank",
        ),
    ],
)
def test_reject_invalid_model_output(
    raw_text: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        InvalidModelOutputError,
        match=expected_message,
    ):
        parse_json_action(raw_text)
