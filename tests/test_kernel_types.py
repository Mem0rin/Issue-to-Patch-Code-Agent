from collections.abc import Callable
from typing import get_args

import pytest

from code_agent.kernel_types import (
    Action,
    FinalAnswer,
    HistoryItem,
    Message,
    ModelFeedback,
    MessageRole,
    ToolCall,
    ToolResult,
)


def test_construct_valid_kernel_types() -> None:
    message = Message(
        role=MessageRole.USER,
        content="Fix the failing test.",
    )
    feedback = ModelFeedback(
        error_message="model output must be valid JSON",
    )
    call = ToolCall(
        call_id="call_1",
        tool_name="read_file",
        arguments={"path": "src/example.py"},
    )
    result = ToolResult(
        call_id="call_1",
        content="def example() -> None: ...",
        is_error=False,
    )
    final = FinalAnswer(content="The task is complete.")

    assert message.role is MessageRole.USER
    assert feedback.error_message == "model output must be valid JSON"
    assert call.arguments == {"path": "src/example.py"}
    assert result.call_id == call.call_id
    assert final.content == "The task is complete."


def test_tool_result_allows_empty_content() -> None:
    result = ToolResult(
        call_id="call_1",
        content="",
        is_error=False,
    )

    assert result.content == ""


@pytest.mark.parametrize(
    ("factory", "expected_message"),
    [
        (
            lambda: Message(
                role=MessageRole.USER,
                content="   ",
            ),
            "content must not be blank",
        ),
        (
            lambda: ModelFeedback(error_message="\t"),
            "error_message must not be blank",
        ),
        (
            lambda: ToolCall(
                call_id=" ",
                tool_name="read_file",
                arguments={},
            ),
            "call_id must not be blank",
        ),
        (
            lambda: ToolCall(
                call_id="call_1",
                tool_name=" ",
                arguments={},
            ),
            "tool_name must not be blank",
        ),
        (
            lambda: ToolResult(
                call_id="",
                content="result",
                is_error=False,
            ),
            "call_id must not be blank",
        ),
        (
            lambda: FinalAnswer(content="\t"),
            "content must not be blank",
        ),
    ],
)
def test_reject_blank_required_text(
    factory: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        factory()


def test_action_contains_model_proposals() -> None:
    assert set(get_args(Action)) == {
        ToolCall,
        FinalAnswer,
    }


def test_history_contains_model_context() -> None:
    assert set(get_args(HistoryItem)) == {
        Message,
        ModelFeedback,
        ToolCall,
        ToolResult,
    }
