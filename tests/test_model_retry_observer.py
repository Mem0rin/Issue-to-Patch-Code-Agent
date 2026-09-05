"""Retry observation and accounting use deterministic model/clock substitutes."""
from collections.abc import Callable
from typing import cast

import pytest

from code_agent.kernel_types import FinalAnswer, ModelFeedback
from code_agent.model_adapter import (
    ConsumptionState, FakeModelAdapter, FakeOutcome, InvalidModelOutputError,
    ModelCallBudget, ModelProviderError, ModelResponse, Retryability, TokenUsage,
)
from code_agent.model_call import complete_model_with_retry
from code_agent.run_budget import BudgetStopReason, BudgetTracker, RunBudget, RunUsage

Observed = tuple[int, object, ModelFeedback | None, RunUsage]
Retry = Callable[..., ModelResponse | BudgetStopReason]


def response() -> ModelResponse:
    return ModelResponse(FinalAnswer("done"), TokenUsage(1, 1))


def provider_error(
    state: ConsumptionState = ConsumptionState.NO_CONSUMPTION,
) -> ModelProviderError:
    return ModelProviderError(
        "provider failure", consumption_state=state,
        retryability=Retryability.RETRYABLE,
        usage=TokenUsage(2, 1) if state is ConsumptionState.ACTUAL_USAGE else None,
    )


class Harness:
    def __init__(self, outcomes: list[FakeOutcome]) -> None:
        self.adapter = FakeModelAdapter(outcomes)
        self.tracker = BudgetTracker(RunBudget(20, 20, 1000, 10), clock=lambda: 0.0)
        self.events: list[Observed] = []

    def observe(
        self, index: int, kind: object, feedback: ModelFeedback | None, usage: RunUsage,
    ) -> None:
        assert len(self.adapter.calls) == index
        assert usage == self.tracker.usage
        self.events.append((index, kind, feedback, usage))

    def run(
        self, *, observer: object = None, provider_limit: object = 2,
        invalid_limit: object = 2,
    ) -> ModelResponse | BudgetStopReason:
        return cast(Retry, complete_model_with_retry)(
            self.adapter, self.tracker, history=(), tools=(),
            budget=ModelCallBudget(10), reserved_tokens=10,
            max_provider_retries=provider_limit,
            max_invalid_output_retries=invalid_limit, on_retry=observer,
        )


@pytest.mark.parametrize("state", list(ConsumptionState))
def test_provider_observation_has_settled_usage_before_next_attempt(
    state: ConsumptionState,
) -> None:
    harness = Harness([provider_error(state), response()])
    result = harness.run(observer=harness.observe)
    assert result == response()
    expected = RunUsage(
        model_calls=1,
        input_tokens=2 if state is ConsumptionState.ACTUAL_USAGE else 0,
        output_tokens=1 if state is ConsumptionState.ACTUAL_USAGE else 0,
        reserved_tokens=10 if state is ConsumptionState.UNKNOWN_CONSUMPTION else 0,
    )
    assert harness.events == [(1, "provider", None, expected)]
    assert len(harness.adapter.calls) == 2


@pytest.mark.parametrize("known", [False, True])
def test_invalid_output_observation_precedes_single_feedback_insertion(known: bool) -> None:
    error = InvalidModelOutputError(" fix format ", usage=TokenUsage(2, 1) if known else None)
    harness = Harness([error, response()])
    harness.run(observer=harness.observe)
    feedback = ModelFeedback("fix format")
    expected = RunUsage(
        model_calls=1, input_tokens=2 if known else 0, output_tokens=1 if known else 0,
        reserved_tokens=0 if known else 10,
    )
    assert harness.events == [(1, "invalid_output", feedback, expected)]
    assert harness.adapter.calls[0].history == ()
    assert harness.adapter.calls[1].history == (feedback,)
    assert harness.events[0][2] is not harness.adapter.calls[1].history[0]
    assert harness.events[0][3] is not harness.tracker.usage


def test_blank_invalid_output_message_uses_nonblank_feedback() -> None:
    harness = Harness([InvalidModelOutputError(" \t "), response()])
    harness.run(observer=harness.observe)
    assert harness.events[0][2] == ModelFeedback(
        "model output did not satisfy the action contract",
    )


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_provider_retry_limit_includes_exactly_limit_retries(limit: int) -> None:
    failures = [provider_error() for _ in range(limit + 1)]
    harness = Harness(list(failures))
    with pytest.raises(ModelProviderError) as caught:
        harness.run(observer=harness.observe, provider_limit=limit)
    assert caught.value is failures[-1]
    assert len(harness.adapter.calls) == limit + 1
    assert [entry[0] for entry in harness.events] == list(range(1, limit + 1))


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_invalid_output_retry_limit_includes_exactly_limit_feedbacks(limit: int) -> None:
    failures = [InvalidModelOutputError("invalid", usage=TokenUsage(0, 0)) for _ in range(limit + 1)]
    harness = Harness(list(failures))
    with pytest.raises(InvalidModelOutputError) as caught:
        harness.run(observer=harness.observe, invalid_limit=limit)
    assert caught.value is failures[-1]
    assert len(harness.adapter.calls) == limit + 1
    assert len(harness.events) == limit
    assert len(harness.adapter.calls[-1].history) == limit


def test_non_retryable_failure_does_not_emit_retry() -> None:
    error = ModelProviderError("permanent", consumption_state=ConsumptionState.NO_CONSUMPTION)
    harness = Harness([error])
    with pytest.raises(ModelProviderError) as caught:
        harness.run(observer=harness.observe)
    assert caught.value is error
    assert harness.events == []
    assert len(harness.adapter.calls) == 1


def test_separate_retry_counters_emit_monotonic_attempt_indexes() -> None:
    harness = Harness([
        provider_error(), InvalidModelOutputError("first"),
        provider_error(), InvalidModelOutputError("second"), response(),
    ])
    harness.run(observer=harness.observe)
    assert [(index, kind) for index, kind, _, _ in harness.events] == [
        (1, "provider"), (2, "invalid_output"), (3, "provider"), (4, "invalid_output"),
    ]
    assert harness.adapter.calls[-1].history == (
        ModelFeedback("first"), ModelFeedback("second"),
    )


def test_attempt_index_continues_across_model_turns() -> None:
    harness = Harness([provider_error(), response(), provider_error(), response()])
    harness.run(observer=harness.observe)
    harness.run(observer=harness.observe)
    assert [event[0] for event in harness.events] == [1, 3]
    assert harness.tracker.usage.model_calls == 4


def test_retry_scheduled_does_not_claim_next_attempt_was_prepared() -> None:
    harness = Harness([InvalidModelOutputError("retry")])
    harness.tracker = BudgetTracker(RunBudget(1, 2, 100, 10), clock=lambda: 0.0)
    result = harness.run(observer=harness.observe)
    assert result is BudgetStopReason.MODEL_CALLS_EXHAUSTED
    assert len(harness.events) == len(harness.adapter.calls) == 1
    assert harness.tracker.usage == RunUsage(model_calls=1, reserved_tokens=10)


@pytest.mark.parametrize("field", ["provider_limit", "invalid_limit"])
@pytest.mark.parametrize("bad", [True, 1.0, None, "1"])
def test_invalid_retry_limit_type_is_rejected_before_model_and_observer(
    field: str, bad: object,
) -> None:
    harness = Harness([response()])
    limits = {field: bad}
    with pytest.raises(TypeError, match="max_.*_retries.*integer"):
        harness.run(observer=harness.observe, **limits)
    assert harness.adapter.calls == [] and harness.events == []
    assert harness.tracker.usage == RunUsage()


@pytest.mark.parametrize("field", ["provider_limit", "invalid_limit"])
def test_negative_retry_limit_is_rejected_without_side_effects(field: str) -> None:
    harness = Harness([response()])
    with pytest.raises(ValueError, match="must not be negative"):
        harness.run(observer=harness.observe, **{field: -1})
    assert harness.adapter.calls == [] and harness.events == []
    assert harness.tracker.usage == RunUsage()


@pytest.mark.parametrize("bad", [False, 0, "", []])
def test_noncallable_observer_is_rejected_before_reservation(bad: object) -> None:
    harness = Harness([response()])
    with pytest.raises(TypeError, match="on_retry must be callable or None"):
        harness.run(observer=bad)
    assert harness.adapter.calls == []
    assert harness.tracker.usage == RunUsage()


def test_falsey_callable_observer_is_still_invoked() -> None:
    harness = Harness([provider_error(), response()])

    class Observer:
        def __bool__(self) -> bool:
            return False

        def __call__(
            self, index: int, kind: object, feedback: ModelFeedback | None, usage: RunUsage,
        ) -> None:
            harness.observe(index, kind, feedback, usage)

    harness.run(observer=Observer())
    assert len(harness.events) == 1


@pytest.mark.parametrize("failure", [
    OSError("trace unavailable"),
    ModelProviderError("observer failure", retryability=Retryability.RETRYABLE),
    InvalidModelOutputError("observer formatting failure"),
])
def test_observer_exception_is_not_reclassified_or_retried(failure: Exception) -> None:
    harness = Harness([provider_error(), response()])

    def fail(index: int, kind: object, feedback: ModelFeedback | None, usage: RunUsage) -> None:
        raise failure

    with pytest.raises(RuntimeError, match="observe model retry after attempt 1 failed") as caught:
        harness.run(observer=fail)
    assert type(caught.value).__name__ == "ModelObservationError"
    assert caught.value.__cause__ is failure
    assert len(harness.adapter.calls) == 1
    assert harness.tracker.usage == RunUsage(model_calls=1)
    assert harness.tracker.reserve_model_call(10) is None


def test_non_none_observer_result_is_a_contract_error_without_next_attempt() -> None:
    harness = Harness([provider_error(), response()])

    def wrong(index: int, kind: object, feedback: ModelFeedback | None, usage: RunUsage) -> str:
        return "unexpected"

    with pytest.raises(RuntimeError, match="observe model retry") as caught:
        harness.run(observer=wrong)
    assert isinstance(caught.value.__cause__, TypeError)
    assert "on_retry must return None" in str(caught.value.__cause__)
    assert len(harness.adapter.calls) == 1


def test_none_observer_preserves_existing_retry_behavior() -> None:
    harness = Harness([provider_error(), response()])
    result = harness.run(observer=None)
    assert result == response()
    assert len(harness.adapter.calls) == 2
