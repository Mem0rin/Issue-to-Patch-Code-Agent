import pytest

from code_agent.kernel_types import FinalAnswer
from code_agent.model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    ModelProviderError,
    ModelResponse,
    TokenUsage,
)
from code_agent.run_budget import (
    BudgetStopReason,
    BudgetTracker,
    RunBudget,
    RunUsage,
)


def test_run_usage_calculates_total_tokens() -> None:
    usage = RunUsage(
        model_calls=2,
        action_steps=1,
        input_tokens=30,
        output_tokens=10,
        reserved_tokens=20,
    )

    assert usage.total_tokens == 60


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_model_calls", 0),
        ("max_action_steps", 0),
        ("max_total_tokens", 0),
        ("max_elapsed_seconds", 0),
    ],
)
def test_run_budget_rejects_non_positive_limits(
    field_name: str,
    value: int,
) -> None:
    arguments: dict[str, int | float] = {
        "max_model_calls": 3,
        "max_action_steps": 5,
        "max_total_tokens": 1_000,
        "max_elapsed_seconds": 30,
    }
    arguments[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        RunBudget(**arguments)  # type: ignore[arg-type]


def test_run_usage_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="model_calls"):
        RunUsage(model_calls=-1)


def make_budget(
    *,
    max_model_calls: int = 3,
    max_action_steps: int = 3,
    max_total_tokens: int = 100,
    max_elapsed_seconds: float = 10,
) -> RunBudget:
    return RunBudget(
        max_model_calls=max_model_calls,
        max_action_steps=max_action_steps,
        max_total_tokens=max_total_tokens,
        max_elapsed_seconds=max_elapsed_seconds,
    )


def test_reserving_model_call_increases_usage() -> None:
    tracker = BudgetTracker(make_budget())

    assert tracker.reserve_model_call(40) is None
    assert tracker.usage.model_calls == 1
    assert tracker.usage.reserved_tokens == 40


@pytest.mark.parametrize("reserved_tokens", [0, -1])
def test_token_reservation_must_be_positive(
    reserved_tokens: int,
) -> None:
    tracker = BudgetTracker(make_budget())

    with pytest.raises(ValueError, match="reserved_tokens"):
        tracker.reserve_model_call(reserved_tokens)


def test_cannot_reserve_two_model_calls_at_once() -> None:
    tracker = BudgetTracker(make_budget())
    tracker.reserve_model_call(40)

    with pytest.raises(RuntimeError, match="already"):
        tracker.reserve_model_call(10)

    assert tracker.usage == RunUsage(
        model_calls=1,
        reserved_tokens=40,
    )


def test_model_call_limit_prevents_another_reservation() -> None:
    tracker = BudgetTracker(
        make_budget(max_model_calls=1)
    )

    assert tracker.reserve_model_call(40) is None

    tracker.retain_model_call_reservation()
    reason = tracker.reserve_model_call(40)

    assert reason is BudgetStopReason.MODEL_CALLS_EXHAUSTED
    assert tracker.usage.model_calls == 1


def test_model_response_settles_reserved_tokens_to_actual_usage() -> None:
    tracker = BudgetTracker(make_budget())
    tracker.reserve_model_call(80)

    tracker.record_model_response(
        ModelResponse(
            action=FinalAnswer(content="42"),
            usage=TokenUsage(
                input_tokens=30,
                output_tokens=10,
            ),
        )
    )

    assert tracker.usage == RunUsage(
        model_calls=1,
        action_steps=1,
        input_tokens=30,
        output_tokens=10,
        reserved_tokens=0,
    )


def test_timeout_retains_reserved_tokens() -> None:
    tracker = BudgetTracker(make_budget())
    tracker.reserve_model_call(80)

    tracker.record_provider_error(
        ModelProviderError("provider timed out")
    )

    assert tracker.usage == RunUsage(
        model_calls=1,
        action_steps=0,
        input_tokens=0,
        output_tokens=0,
        reserved_tokens=80,
    )
    assert (
        tracker.reserve_model_call(30)
        is BudgetStopReason.TOKENS_EXHAUSTED
    )
    assert tracker.usage.model_calls == 1


def test_provider_error_releases_when_request_was_not_sent() -> None:
    tracker = BudgetTracker(make_budget())
    tracker.reserve_model_call(80)

    tracker.record_provider_error(
        ModelProviderError(
            "local request validation failed",
            consumption_state=ConsumptionState.NO_CONSUMPTION,
        )
    )

    assert tracker.usage == RunUsage(model_calls=1)


def test_provider_error_settles_reported_usage() -> None:
    tracker = BudgetTracker(make_budget())
    tracker.reserve_model_call(80)

    tracker.record_provider_error(
        ModelProviderError(
            "provider returned an error with usage",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=TokenUsage(
                input_tokens=20,
                output_tokens=5,
            ),
        )
    )

    assert tracker.usage == RunUsage(
        model_calls=1,
        input_tokens=20,
        output_tokens=5,
    )


def test_action_step_limit_prevents_another_model_call() -> None:
    tracker = BudgetTracker(
        make_budget(max_action_steps=1)
    )
    tracker.reserve_model_call(2)
    tracker.record_model_response(
        ModelResponse(
            action=FinalAnswer(content="42"),
            usage=TokenUsage(
                input_tokens=1,
                output_tokens=1,
            ),
        )
    )

    reason = tracker.reserve_model_call(2)

    assert reason is BudgetStopReason.ACTION_STEPS_EXHAUSTED
    assert tracker.usage.model_calls == 1


def test_token_limit_prevents_another_model_call() -> None:
    tracker = BudgetTracker(
        make_budget(max_total_tokens=40)
    )
    tracker.reserve_model_call(40)
    tracker.record_model_response(
        ModelResponse(
            action=FinalAnswer(content="42"),
            usage=TokenUsage(
                input_tokens=30,
                output_tokens=10,
            ),
        )
    )

    reason = tracker.reserve_model_call(1)

    assert reason is BudgetStopReason.TOKENS_EXHAUSTED
    assert tracker.usage.model_calls == 1


def test_elapsed_time_stops_model_call() -> None:
    current_time = [100.0]
    tracker = BudgetTracker(
        make_budget(max_elapsed_seconds=10),
        clock=lambda: current_time[0],
    )
    current_time[0] = 110.0

    reason = tracker.reserve_model_call(10)

    assert reason is BudgetStopReason.TIME_EXHAUSTED
    assert tracker.usage.model_calls == 0


def test_invalid_model_output_records_tokens_without_action() -> None:
    tracker = BudgetTracker(make_budget())
    tracker.reserve_model_call(50)

    error = InvalidModelOutputError(
        "invalid JSON",
        usage=TokenUsage(
            input_tokens=20,
            output_tokens=5,
        ),
    )
    tracker.record_invalid_model_output(error)

    assert tracker.usage == RunUsage(
        model_calls=1,
        action_steps=0,
        input_tokens=20,
        output_tokens=5,
        reserved_tokens=0,
    )
