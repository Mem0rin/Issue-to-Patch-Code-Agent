from collections.abc import Sequence

import pytest

from code_agent.kernel_types import (
    FinalAnswer,
    HistoryItem,
    Message,
    MessageRole,
    ModelFeedback,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from code_agent.model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    ModelAdapter,
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    Retryability,
    TokenUsage,
)
from code_agent.native_tool_adapter import (
    NativeProviderError,
    NativeProviderHistoryItem,
    NativeProviderMessage,
    NativeProviderResponse,
    NativeProviderToolCall,
    NativeProviderToolResult,
    NativeToolCallingModelAdapter,
)


class StubNativeProvider:
    def __init__(
        self,
        *,
        response: NativeProviderResponse | None = None,
        error: NativeProviderError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[
            tuple[
                tuple[NativeProviderHistoryItem, ...],
                tuple[ToolSpec, ...],
                int,
            ]
        ] = []

    def complete(
        self,
        history: Sequence[NativeProviderHistoryItem],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
    ) -> NativeProviderResponse:
        self.calls.append(
            (tuple(history), tuple(tools), max_output_tokens)
        )
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError(
                "stub requires a response or error"
            )
        return self.response


def make_response(
    *,
    tool_calls: tuple[NativeProviderToolCall, ...] = (),
    final_text: str | None = None,
) -> NativeProviderResponse:
    return NativeProviderResponse(
        tool_calls=tool_calls,
        final_text=final_text,
        input_tokens=20,
        output_tokens=5,
    )


def test_complete_preserves_native_history_and_call_id() -> None:
    call = ToolCall(
        call_id="call_previous",
        tool_name="calculator",
        arguments={"expression": "1+1"},
    )
    result = ToolResult(
        call_id="call_previous",
        content="2",
        is_error=False,
    )
    feedback = ModelFeedback(
        error_message="provider returned multiple tool calls",
    )
    history: list[HistoryItem] = [
        Message(
            role=MessageRole.USER,
            content="Calculate the answer.",
        ),
        call,
        result,
        feedback,
    ]
    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        input_schema={"type": "object"},
    )
    raw_call = NativeProviderToolCall(
        call_id="provider_call_42",
        tool_name="unknown_tool",
        arguments={"value": 42},
    )
    provider = StubNativeProvider(
        response=make_response(tool_calls=(raw_call,))
    )
    adapter = NativeToolCallingModelAdapter(provider)

    response = adapter.complete(
        history=history,
        tools=[spec],
        budget=ModelCallBudget(128),
    )

    assert response == ModelResponse(
        action=ToolCall(
            call_id="provider_call_42",
            tool_name="unknown_tool",
            arguments={"value": 42},
        ),
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )
    assert provider.calls == [
        (
            (
                NativeProviderMessage(
                    role="user",
                    content="Calculate the answer.",
                ),
                NativeProviderToolCall(
                    call_id="call_previous",
                    tool_name="calculator",
                    arguments={"expression": "1+1"},
                ),
                NativeProviderToolResult(
                    call_id="call_previous",
                    content="2",
                    is_error=False,
                ),
                NativeProviderMessage(
                    role="user",
                    content=(
                        "Previous model output was invalid: provider "
                        "returned multiple tool calls. Return exactly "
                        "one action matching the declared tool-calling "
                        "contract."
                    ),
                ),
            ),
            (spec,),
            128,
        )
    ]


def test_complete_converts_final_answer_and_usage() -> None:
    provider = StubNativeProvider(
        response=make_response(final_text="The answer is 42.")
    )
    adapter = NativeToolCallingModelAdapter(provider)

    response = adapter.complete(
        history=[],
        tools=[],
        budget=ModelCallBudget(128),
    )

    assert response == ModelResponse(
        action=FinalAnswer(content="The answer is 42."),
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (
            make_response(),
            "provider returned no action",
        ),
        (
            make_response(
                tool_calls=(
                    NativeProviderToolCall("call_1", "first", {}),
                    NativeProviderToolCall("call_2", "second", {}),
                )
            ),
            "provider returned multiple tool calls",
        ),
        (
            make_response(
                tool_calls=(
                    NativeProviderToolCall("call_1", "first", {}),
                ),
                final_text="also finished",
            ),
            "provider returned a tool call and final text",
        ),
        (
            make_response(final_text=" "),
            "content must not be blank",
        ),
    ],
)
def test_complete_rejects_ambiguous_or_invalid_actions(
    response: NativeProviderResponse,
    expected_message: str,
) -> None:
    adapter = NativeToolCallingModelAdapter(
        StubNativeProvider(response=response)
    )

    with pytest.raises(
        InvalidModelOutputError,
        match=expected_message,
    ) as exc_info:
        adapter.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert exc_info.value.usage == TokenUsage(
        input_tokens=20,
        output_tokens=5,
    )


def test_complete_normalizes_provider_error_semantics() -> None:
    usage = TokenUsage(input_tokens=12, output_tokens=3)
    original_error = NativeProviderError(
        "provider returned 503 after reporting usage",
        consumption_state=ConsumptionState.ACTUAL_USAGE,
        retryability=Retryability.RETRYABLE,
        usage=usage,
    )
    adapter = NativeToolCallingModelAdapter(
        StubNativeProvider(error=original_error)
    )

    with pytest.raises(ModelProviderError) as exc_info:
        adapter.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    error = exc_info.value
    assert error.__cause__ is original_error
    assert error.consumption_state is ConsumptionState.ACTUAL_USAGE
    assert error.retryability is Retryability.RETRYABLE
    assert error.usage is usage


def _accepts_model_adapter(adapter: ModelAdapter) -> None:
    _ = adapter


def test_adapter_satisfies_model_adapter_protocol() -> None:
    adapter = NativeToolCallingModelAdapter(
        StubNativeProvider(response=make_response(final_text="done"))
    )

    _accepts_model_adapter(adapter)
