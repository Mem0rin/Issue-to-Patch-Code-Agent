import pytest

from code_agent.kernel_types import (
    FinalAnswer,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolSpec,
    HistoryItem,
)
from code_agent.model_adapter import (
    ConsumptionState,
    FakeModelAdapter,
    FakeModelExhaustedError,
    InvalidModelOutputError,
    ModelAdapter,
    ModelCall,
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    Retryability,
    TokenUsage,
)


def _accepts_model_adapter(adapter: ModelAdapter) -> None:
    """Let mypy verify structural compatibility."""
    _ = adapter


def test_model_call_budget_requires_positive_limit() -> None:
    with pytest.raises(
        ValueError,
        match="max_output_tokens must be positive",
    ):
        ModelCallBudget(max_output_tokens=0)


def test_token_usage_reports_total() -> None:
    usage = TokenUsage(
        input_tokens=120,
        output_tokens=30,
    )

    assert usage.total_tokens == 150


def test_token_usage_rejects_negative_input_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="input_tokens must not be negative",
    ):
        TokenUsage(
            input_tokens=-1,
            output_tokens=0,
        )


def test_token_usage_rejects_negative_output_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="output_tokens must not be negative",
    ):
        TokenUsage(
            input_tokens=0,
            output_tokens=-1,
        )


def test_provider_error_defaults_to_unknown_consumption() -> None:
    error = ModelProviderError("provider timed out")

    assert (
        error.consumption_state
        is ConsumptionState.UNKNOWN_CONSUMPTION
    )
    assert error.retryability is Retryability.NON_RETRYABLE
    assert error.usage is None


def test_provider_error_can_be_explicitly_retryable() -> None:
    error = ModelProviderError(
        "provider returned 503",
        retryability=Retryability.RETRYABLE,
    )

    assert error.retryability is Retryability.RETRYABLE


def test_actual_consumption_requires_usage() -> None:
    with pytest.raises(ValueError, match="usage is required"):
        ModelProviderError(
            "provider failed after billing",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
        )


@pytest.mark.parametrize(
    "consumption_state",
    [
        ConsumptionState.NO_CONSUMPTION,
        ConsumptionState.UNKNOWN_CONSUMPTION,
    ],
)
def test_non_actual_consumption_rejects_usage(
    consumption_state: ConsumptionState,
) -> None:
    with pytest.raises(ValueError, match="usage must be absent"):
        ModelProviderError(
            "provider failed",
            consumption_state=consumption_state,
            usage=TokenUsage(input_tokens=10, output_tokens=2),
        )


@pytest.mark.parametrize(
    ("name", "description", "expected_message"),
    [
        (" ", "Read a file", "name must not be blank"),
        ("read_file", "\t", "description must not be blank"),
    ],
)
def test_tool_spec_rejects_blank_required_text(
    name: str,
    description: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        ToolSpec(
            name=name,
            description=description,
            input_schema={},
        )


def test_fake_returns_outcomes_in_order_and_records_calls() -> None:
    message = Message(
        role=MessageRole.USER,
        content="Read README.md",
    )
    spec = ToolSpec(
        name="read_file",
        description="Read one workspace file.",
        input_schema={"type": "object"},
    )
    budget = ModelCallBudget(max_output_tokens=128)

    tool_call = ToolCall(
        call_id="call_1",
        tool_name="read_file",
        arguments={"path": "README.md"},
    )
    tool_response = ModelResponse(
        action=tool_call,
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
        ),
    )
    final_response = ModelResponse(
        action=FinalAnswer(content="README inspected."),
        usage=TokenUsage(
            input_tokens=20,
            output_tokens=4,
        ),
    )
    fake = FakeModelAdapter([
        tool_response,
        final_response,
    ])

    history: list[HistoryItem] = [message]
    tools = [spec]

    assert fake.complete(history, tools, budget) is tool_response

    result = ToolResult(
        call_id="call_1",
        content="README contents",
        is_error=False,
    )
    history.extend([tool_call, result])

    assert fake.complete(history, tools, budget) is final_response

    assert fake.calls == [
        ModelCall(
            history=(message,),
            tools=(spec,),
            budget=budget,
        ),
        ModelCall(
            history=(message, tool_call, result),
            tools=(spec,),
            budget=budget,
        ),
    ]


def test_fake_reraises_provider_error() -> None:
    error = ModelProviderError("provider returned 503")
    fake = FakeModelAdapter([error])

    with pytest.raises(ModelProviderError) as exc_info:
        fake.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert exc_info.value is error


def test_fake_reraises_invalid_model_output() -> None:
    error = InvalidModelOutputError("arguments must be an object")
    fake = FakeModelAdapter([error])

    with pytest.raises(InvalidModelOutputError) as exc_info:
        fake.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert exc_info.value is error


def test_fake_reports_exhaustion() -> None:
    fake = FakeModelAdapter([])

    with pytest.raises(
        FakeModelExhaustedError,
        match="fake model has no configured outcome left",
    ):
        fake.complete(
            history=[],
            tools=[],
            budget=ModelCallBudget(128),
        )

    assert len(fake.calls) == 1


def test_fake_satisfies_model_adapter_protocol() -> None:
    fake = FakeModelAdapter([])

    _accepts_model_adapter(fake)
