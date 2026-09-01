from collections.abc import Sequence

import pytest

from code_agent.kernel_types import (
    FinalAnswer,
    HistoryItem,
    Message,
    MessageRole,
    ModelFeedback,
    ToolSpec,
)
from code_agent.model_adapter import (
    ConsumptionState,
    FakeModelAdapter,
    InvalidModelOutputError,
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    Retryability,
    TokenUsage,
)
from code_agent.model_call import (
    complete_model_call,
    complete_model_with_retry,
)
from code_agent.run_budget import (
    BudgetStopReason,
    BudgetTracker,
    RunBudget,
    RunUsage,
)


def make_tracker() -> BudgetTracker:
    return BudgetTracker(
        RunBudget(
            max_model_calls=3,
            max_action_steps=3,
            max_total_tokens=100,
            max_elapsed_seconds=10,
        )
    )


def test_budget_stop_prevents_model_call() -> None:
    response = ModelResponse(
        action=FinalAnswer(content="unused"),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    adapter = FakeModelAdapter([response])
    tracker = make_tracker()

    result = complete_model_call(
        adapter,
        tracker,
        history=[],
        tools=[],
        budget=ModelCallBudget(128),
        reserved_tokens=101,
    )

    assert result is BudgetStopReason.TOKENS_EXHAUSTED
    assert adapter.calls == []
    assert tracker.usage == RunUsage()


def test_successful_call_settles_actual_usage() -> None:
    response = ModelResponse(
        action=FinalAnswer(content="42"),
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )
    adapter = FakeModelAdapter([response])
    tracker = make_tracker()

    result = complete_model_call(
        adapter,
        tracker,
        history=[],
        tools=[],
        budget=ModelCallBudget(40),
        reserved_tokens=80,
    )

    assert result is response
    assert tracker.usage == RunUsage(
        model_calls=1,
        action_steps=1,
        input_tokens=20,
        output_tokens=5,
    )


def test_invalid_output_is_recorded_before_reraising() -> None:
    error = InvalidModelOutputError(
        "invalid JSON",
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )
    tracker = make_tracker()

    with pytest.raises(InvalidModelOutputError) as exc_info:
        complete_model_call(
            FakeModelAdapter([error]),
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=80,
        )

    assert exc_info.value is error
    assert tracker.usage == RunUsage(
        model_calls=1,
        input_tokens=20,
        output_tokens=5,
    )


@pytest.mark.parametrize(
    ("error", "expected_usage"),
    [
        (
            ModelProviderError(
                "request was not sent",
                consumption_state=(
                    ConsumptionState.NO_CONSUMPTION
                ),
            ),
            RunUsage(model_calls=1),
        ),
        (
            ModelProviderError(
                "provider reported usage",
                consumption_state=ConsumptionState.ACTUAL_USAGE,
                usage=TokenUsage(input_tokens=20, output_tokens=5),
            ),
            RunUsage(
                model_calls=1,
                input_tokens=20,
                output_tokens=5,
            ),
        ),
        (
            ModelProviderError("provider timed out"),
            RunUsage(model_calls=1, reserved_tokens=80),
        ),
    ],
)
def test_provider_error_is_recorded_before_reraising(
    error: ModelProviderError,
    expected_usage: RunUsage,
) -> None:
    tracker = make_tracker()

    with pytest.raises(ModelProviderError) as exc_info:
        complete_model_call(
            FakeModelAdapter([error]),
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=80,
        )

    assert exc_info.value is error
    assert tracker.usage == expected_usage


class BrokenModelAdapter:
    def complete(
        self,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        raise RuntimeError("unexpected adapter failure")


def test_unexpected_error_retains_reservation() -> None:
    tracker = make_tracker()

    with pytest.raises(RuntimeError, match="unexpected adapter"):
        complete_model_call(
            BrokenModelAdapter(),
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=80,
        )

    assert tracker.usage == RunUsage(
        model_calls=1,
        reserved_tokens=80,
    )


def test_retryable_provider_error_retries_with_new_reservation() -> None:
    error = ModelProviderError(
        "provider returned 503",
        consumption_state=ConsumptionState.NO_CONSUMPTION,
        retryability=Retryability.RETRYABLE,
    )
    response = ModelResponse(
        action=FinalAnswer(content="recovered"),
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )
    adapter = FakeModelAdapter([error, response])
    tracker = make_tracker()

    result = complete_model_with_retry(
        adapter,
        tracker,
        history=[],
        tools=[],
        budget=ModelCallBudget(40),
        reserved_tokens=40,
        max_provider_retries=1,
    )

    assert result is response
    assert len(adapter.calls) == 2
    assert tracker.usage == RunUsage(
        model_calls=2,
        action_steps=1,
        input_tokens=20,
        output_tokens=5,
    )


def test_non_retryable_provider_error_is_reraised() -> None:
    error = ModelProviderError(
        "invalid API key",
        consumption_state=ConsumptionState.NO_CONSUMPTION,
    )
    adapter = FakeModelAdapter([error])
    tracker = make_tracker()

    with pytest.raises(ModelProviderError) as exc_info:
        complete_model_with_retry(
            adapter,
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=40,
            max_provider_retries=1,
        )

    assert exc_info.value is error
    assert len(adapter.calls) == 1
    assert tracker.usage == RunUsage(model_calls=1)


def test_retry_stops_when_token_budget_cannot_reserve_again() -> None:
    error = ModelProviderError(
        "provider timed out",
        retryability=Retryability.RETRYABLE,
    )
    response = ModelResponse(
        action=FinalAnswer(content="unused"),
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )
    adapter = FakeModelAdapter([error, response])
    tracker = make_tracker()

    result = complete_model_with_retry(
        adapter,
        tracker,
        history=[],
        tools=[],
        budget=ModelCallBudget(40),
        reserved_tokens=80,
        max_provider_retries=1,
    )

    assert result is BudgetStopReason.TOKENS_EXHAUSTED
    assert len(adapter.calls) == 1
    assert tracker.usage == RunUsage(
        model_calls=1,
        reserved_tokens=80,
    )


def test_provider_retry_limit_is_enforced() -> None:
    first_error = ModelProviderError(
        "provider returned 503",
        consumption_state=ConsumptionState.NO_CONSUMPTION,
        retryability=Retryability.RETRYABLE,
    )
    second_error = ModelProviderError(
        "provider returned 503 again",
        consumption_state=ConsumptionState.NO_CONSUMPTION,
        retryability=Retryability.RETRYABLE,
    )
    adapter = FakeModelAdapter([first_error, second_error])
    tracker = make_tracker()

    with pytest.raises(ModelProviderError) as exc_info:
        complete_model_with_retry(
            adapter,
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=40,
            max_provider_retries=1,
        )

    assert exc_info.value is second_error
    assert len(adapter.calls) == 2
    assert tracker.usage == RunUsage(model_calls=2)


def test_provider_retry_limit_must_not_be_negative() -> None:
    adapter = FakeModelAdapter([])
    tracker = make_tracker()

    with pytest.raises(ValueError, match="must not be negative"):
        complete_model_with_retry(
            adapter,
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=40,
            max_provider_retries=-1,
        )

    assert adapter.calls == []
    assert tracker.usage == RunUsage()


def test_invalid_output_retry_adds_feedback_without_mutating_history() -> None:
    message = Message(
        role=MessageRole.USER,
        content="Fix the failing test.",
    )
    history: list[HistoryItem] = [message]
    error = InvalidModelOutputError(
        "model output must be valid JSON",
        usage=TokenUsage(input_tokens=10, output_tokens=2),
    )
    response = ModelResponse(
        action=FinalAnswer(content="recovered"),
        usage=TokenUsage(input_tokens=20, output_tokens=5),
    )
    adapter = FakeModelAdapter([error, response])
    tracker = make_tracker()

    result = complete_model_with_retry(
        adapter,
        tracker,
        history=history,
        tools=[],
        budget=ModelCallBudget(40),
        reserved_tokens=40,
        max_provider_retries=0,
        max_invalid_output_retries=1,
    )

    feedback = ModelFeedback(
        error_message="model output must be valid JSON",
    )
    assert result is response
    assert history == [message]
    assert adapter.calls[0].history == (message,)
    assert adapter.calls[1].history == (message, feedback)
    assert tracker.usage == RunUsage(
        model_calls=2,
        action_steps=1,
        input_tokens=30,
        output_tokens=7,
    )


def test_invalid_output_retry_limit_is_enforced() -> None:
    first_error = InvalidModelOutputError(
        "model output must be valid JSON",
        usage=TokenUsage(input_tokens=10, output_tokens=1),
    )
    second_error = InvalidModelOutputError(
        "unsupported action type",
        usage=TokenUsage(input_tokens=10, output_tokens=1),
    )
    adapter = FakeModelAdapter([first_error, second_error])
    tracker = make_tracker()

    with pytest.raises(InvalidModelOutputError) as exc_info:
        complete_model_with_retry(
            adapter,
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=40,
            max_provider_retries=0,
            max_invalid_output_retries=1,
        )

    assert exc_info.value is second_error
    assert len(adapter.calls) == 2
    assert tracker.usage == RunUsage(
        model_calls=2,
        input_tokens=20,
        output_tokens=2,
    )


def test_invalid_output_retry_limit_must_not_be_negative() -> None:
    adapter = FakeModelAdapter([])
    tracker = make_tracker()

    with pytest.raises(ValueError, match="must not be negative"):
        complete_model_with_retry(
            adapter,
            tracker,
            history=[],
            tools=[],
            budget=ModelCallBudget(40),
            reserved_tokens=40,
            max_provider_retries=0,
            max_invalid_output_retries=-1,
        )

    assert adapter.calls == []
    assert tracker.usage == RunUsage()
