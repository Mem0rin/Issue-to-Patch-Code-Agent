# tests/test_json_text_model_adapter.py

from collections.abc import Sequence

import pytest

from code_agent.json_text_messages import ProviderMessage

from code_agent.json_text_adapter import (
    JsonTextModelAdapter,
    ProviderError,
    ProviderResponse,
)
from code_agent.kernel_types import (
    FinalAnswer,
    HistoryItem,
    Message,
    MessageRole,
    ToolSpec,
    ToolCall,
)
from code_agent.model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    ModelCall,
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    Retryability,
    TokenUsage,
)


class StubProvider:
    def __init__(
        self,
        *,
        response: ProviderResponse | None = None,
        error: ProviderError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[
            tuple[
                tuple[ProviderMessage, ...],
                tuple[ToolSpec, ...],
                int,
            ]
        ] = []

    def complete(
        self,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
    ) -> ProviderResponse:
        self.calls.append(
            (
                tuple(messages),
                tuple(tools),
                max_output_tokens,
            )
        )

        if self.error is not None:
            raise self.error

        if self.response is None:
            raise AssertionError(
                "StubProvider requires a response or error"
            )

        return self.response


def test_complete_converts_final_answer_and_usage() -> None:
    provider = StubProvider(
        response=ProviderResponse(
            text=(
                '{"type":"final_answer",'
                '"content":"42"}'
            ),
            input_tokens=120,
            output_tokens=15,
        )
    )
    adapter = JsonTextModelAdapter(provider)

    message = Message(
        role=MessageRole.USER,
        content="Calculate the answer.",
    )
    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        input_schema={"type": "object"},
    )
    history: list[HistoryItem] = [message]
    tools = [spec]
    budget = ModelCallBudget(max_output_tokens=128)

    result = adapter.complete(history, tools, budget)

    assert result == ModelResponse(
        action=FinalAnswer(content="42"),
        usage=TokenUsage(
            input_tokens=120,
            output_tokens=15,
        ),
    )
    assert provider.calls == [
        (
            (
                ProviderMessage(
                    role="user",
                    content="Calculate the answer.",
                ),
            ),
            (spec,),
            128,
        )
    ]


def test_complete_normalizes_provider_error() -> None:
    original_error = ProviderError(
        "provider returned HTTP 503"
    )
    provider = StubProvider(error=original_error)
    adapter = JsonTextModelAdapter(provider)
    budget = ModelCallBudget(max_output_tokens=128)

    with pytest.raises(
        ModelProviderError,
        match="provider request failed",
    ) as exc_info:
        adapter.complete(
            history=[],
            tools=[],
            budget=budget,
        )

    assert exc_info.value.__cause__ is original_error
    assert (
        exc_info.value.consumption_state
        is ConsumptionState.UNKNOWN_CONSUMPTION
    )
    assert exc_info.value.retryability is Retryability.NON_RETRYABLE
    assert exc_info.value.usage is None


def test_complete_preserves_confirmed_no_consumption() -> None:
    original_error = ProviderError(
        "local validation failed",
        consumption_state=ConsumptionState.NO_CONSUMPTION,
    )
    adapter = JsonTextModelAdapter(
        StubProvider(error=original_error)
    )

    with pytest.raises(ModelProviderError) as exc_info:
        adapter.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert (
        exc_info.value.consumption_state
        is ConsumptionState.NO_CONSUMPTION
    )
    assert exc_info.value.usage is None


def test_complete_preserves_provider_retryability() -> None:
    original_error = ProviderError(
        "provider returned HTTP 503",
        consumption_state=ConsumptionState.NO_CONSUMPTION,
        retryability=Retryability.RETRYABLE,
    )
    adapter = JsonTextModelAdapter(
        StubProvider(error=original_error)
    )

    with pytest.raises(ModelProviderError) as exc_info:
        adapter.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert exc_info.value.retryability is Retryability.RETRYABLE


def test_complete_preserves_provider_error_usage() -> None:
    usage = TokenUsage(input_tokens=20, output_tokens=5)
    original_error = ProviderError(
        "provider returned an error with usage",
        consumption_state=ConsumptionState.ACTUAL_USAGE,
        usage=usage,
    )
    adapter = JsonTextModelAdapter(
        StubProvider(error=original_error)
    )

    with pytest.raises(ModelProviderError) as exc_info:
        adapter.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert (
        exc_info.value.consumption_state
        is ConsumptionState.ACTUAL_USAGE
    )
    assert exc_info.value.usage is usage


def test_complete_preserves_invalid_model_output_error() -> None:
    provider = StubProvider(
        response=ProviderResponse(
            text="this is not JSON",
            input_tokens=20,
            output_tokens=5,
        )
    )
    adapter = JsonTextModelAdapter(provider)

    with pytest.raises(
        InvalidModelOutputError,
        match="model output must be valid JSON",
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


def test_complete_converts_tool_call() -> None:
    provider = StubProvider(
        response=ProviderResponse(
            text=(
                '{"type":"tool_call",'
                '"call_id":"call_1",'
                '"tool_name":"read_file",'
                '"arguments":{"path":"README.md"}}'
            ),
            input_tokens=30,
            output_tokens=20,
        )
    )
    adapter = JsonTextModelAdapter(provider)

    result = adapter.complete(
        history=[],
        tools=[],
        budget=ModelCallBudget(128),
    )

    assert result.action == ToolCall(
        call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )
