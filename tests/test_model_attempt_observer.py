"""Attempt observation occurs outside model error classification and settlement."""
from collections.abc import Callable, Sequence
from typing import cast

import pytest

from code_agent.kernel_types import FinalAnswer, HistoryItem, ToolSpec
from code_agent.model_adapter import (
    ConsumptionState, FakeModelAdapter, InvalidModelOutputError, ModelCallBudget,
    ModelProviderError, ModelResponse, Retryability, TokenUsage,
)
from code_agent.model_call import (
    ModelCallObserver, ModelObservationError, complete_model_call, complete_model_with_retry,
)
from code_agent.run_budget import BudgetStopReason, BudgetTracker, RunBudget, RunUsage


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, RunUsage]] = []
        self.fail_at: str | None = None
        self.failure: Exception = OSError("observer failed")
        self.latencies: list[float] = []

    def prepared(self, attempt_index: int, reserved_tokens: int, usage: RunUsage) -> None:
        assert reserved_tokens == 10
        self.events.append(("prepared", attempt_index, usage))
        if self.fail_at == "prepared":
            raise self.failure

    def completed(self, attempt_index: int, response: ModelResponse, usage: RunUsage, latency: float) -> None:
        self.events.append(("completed", attempt_index, usage))
        self.latencies.append(latency)
        if self.fail_at == "completed":
            raise self.failure

    def failed(self, attempt_index: int, error: Exception, usage: RunUsage, latency: float) -> None:
        self.events.append(("failed", attempt_index, usage))
        self.latencies.append(latency)
        if self.fail_at == "failed":
            raise self.failure


def tracker() -> BudgetTracker:
    return BudgetTracker(RunBudget(4, 4, 100, 5), clock=lambda: 0.0)


def run(adapter: FakeModelAdapter, budget: BudgetTracker, observer: object) -> ModelResponse | BudgetStopReason:
    return complete_model_with_retry(
        adapter, budget, history=(), tools=(), budget=ModelCallBudget(10),
        reserved_tokens=10, max_provider_retries=2, max_invalid_output_retries=2,
        observer=cast(ModelCallObserver, observer),
    )


def test_success_observes_reservation_then_settled_usage() -> None:
    budget = tracker()
    observer = Recorder()
    response = ModelResponse(FinalAnswer("done"), TokenUsage(2, 1))
    adapter = FakeModelAdapter([response])
    assert run(adapter, budget, observer) is response
    assert observer.events == [
        ("prepared", 1, RunUsage(model_calls=1, reserved_tokens=10)),
        ("completed", 1, RunUsage(model_calls=1, action_steps=1, input_tokens=2, output_tokens=1)),
    ]
    assert observer.latencies == [0.0]


@pytest.mark.parametrize(("error", "reserved", "tokens"), [
    (ModelProviderError("unknown"), 10, 0),
    (ModelProviderError("not sent", consumption_state=ConsumptionState.NO_CONSUMPTION), 0, 0),
    (ModelProviderError("known", consumption_state=ConsumptionState.ACTUAL_USAGE, usage=TokenUsage(2, 1)), 0, 3),
    (InvalidModelOutputError("unknown"), 10, 0),
    (InvalidModelOutputError("known", usage=TokenUsage(2, 1)), 0, 3),
])
def test_failure_observation_sees_exactly_once_settlement(
    error: ModelProviderError | InvalidModelOutputError, reserved: int, tokens: int,
) -> None:
    budget = tracker()
    observer = Recorder()
    adapter = FakeModelAdapter([error])
    with pytest.raises(type(error)) as caught:
        complete_model_call(
            adapter, budget, history=(), tools=(), budget=ModelCallBudget(10),
            reserved_tokens=10, observer=observer,
        )
    assert caught.value is error
    assert observer.events[-1][0] == "failed"
    assert observer.events[-1][2].reserved_tokens == reserved
    assert observer.events[-1][2].input_tokens + observer.events[-1][2].output_tokens == tokens
    assert budget.usage.action_steps == 0


@pytest.mark.parametrize("stage", ["prepared", "completed", "failed"])
@pytest.mark.parametrize("failure", [
    OSError("write failed"),
    ModelProviderError("callback provider error", retryability=Retryability.RETRYABLE),
    InvalidModelOutputError("callback invalid output"),
])
def test_observer_failure_is_never_retried_or_double_settled(stage: str, failure: Exception) -> None:
    budget = tracker()
    observer = Recorder()
    observer.fail_at, observer.failure = stage, failure
    first = ModelProviderError("model failed", consumption_state=ConsumptionState.NO_CONSUMPTION, retryability=Retryability.RETRYABLE)
    response = ModelResponse(FinalAnswer("done"), TokenUsage(2, 1))
    adapter = FakeModelAdapter([first if stage == "failed" else response, response])
    with pytest.raises(ModelObservationError, match=f"model {stage}") as caught:
        run(adapter, budget, observer)
    assert caught.value.__cause__ is failure
    assert len(adapter.calls) == (0 if stage == "prepared" else 1)
    assert budget.usage.model_calls == 1 and budget.usage.reserved_tokens == 0
    assert budget.usage.action_steps == (1 if stage == "completed" else 0)
    assert budget.reserve_model_call(10) is None


@pytest.mark.parametrize("method", ["prepared", "completed", "failed"])
def test_invalid_observer_method_rejected_before_reservation(method: str) -> None:
    budget = tracker()
    observer = Recorder()
    setattr(observer, method, None)
    adapter = FakeModelAdapter([])
    with pytest.raises(TypeError, match=f"observer.{method} must be callable"):
        run(adapter, budget, observer)
    assert adapter.calls == [] and budget.usage == RunUsage()


@pytest.mark.parametrize("method", ["prepared", "completed", "failed"])
def test_non_none_observer_return_is_rejected(method: str) -> None:
    budget = tracker()
    observer = Recorder()
    setattr(observer, method, lambda *args: "bad")
    outcome = ModelProviderError("failure") if method == "failed" else ModelResponse(FinalAnswer("done"), TokenUsage(1, 1))
    adapter = FakeModelAdapter([outcome])
    with pytest.raises(ModelObservationError) as caught:
        run(adapter, budget, observer)
    assert isinstance(caught.value.__cause__, TypeError)
    assert f"observer.{method} must return None" in str(caught.value.__cause__)


def test_budget_refusal_does_not_notify_observer() -> None:
    budget = BudgetTracker(RunBudget(1, 1, 9, 5), clock=lambda: 0.0)
    observer = Recorder()
    adapter = FakeModelAdapter([])
    assert run(adapter, budget, observer) is BudgetStopReason.TOKENS_EXHAUSTED
    assert observer.events == [] and adapter.calls == []


def test_none_observer_preserves_direct_call_result() -> None:
    budget = tracker()
    response = ModelResponse(FinalAnswer("done"), TokenUsage(1, 1))
    assert run(FakeModelAdapter([response]), budget, None) is response


def test_failure_snapshot_cannot_mutate_retry_classification() -> None:
    budget = tracker()
    failure = ModelProviderError("original", retryability=Retryability.RETRYABLE)
    response = ModelResponse(FinalAnswer("done"), TokenUsage(1, 1))
    adapter = FakeModelAdapter([failure, response])

    class Mutating(Recorder):
        def failed(self, attempt_index: int, error: Exception, usage: RunUsage, latency: float) -> None:
            assert isinstance(error, ModelProviderError)
            error.retryability = Retryability.NON_RETRYABLE

    assert run(adapter, budget, Mutating()) is response
    assert failure.retryability is Retryability.RETRYABLE


def test_unexpected_error_identity_survives_observation() -> None:
    class Failure(RuntimeError):
        def __init__(self, first: str, second: str) -> None:
            super().__init__(first)
            self.second = second

    failure = Failure("private", "detail")

    class Adapter:
        def complete(self, history: Sequence[HistoryItem], tools: Sequence[ToolSpec], budget: ModelCallBudget) -> ModelResponse:
            raise failure

    budget = tracker()
    observer = Recorder()
    with pytest.raises(Failure) as caught:
        complete_model_call(Adapter(), budget, history=(), tools=(), budget=ModelCallBudget(1), reserved_tokens=10, observer=observer)
    assert caught.value is failure
    assert observer.events[-1][0] == "failed"
    assert budget.usage.reserved_tokens == 10


def test_latency_uses_shared_clock_and_cannot_be_negative() -> None:
    readings = iter([0.0, 0.0, 2.0, 2.0, 1.0])
    budget = BudgetTracker(RunBudget(4, 4, 100, 10), clock=lambda: next(readings))
    adapter = FakeModelAdapter([ModelResponse(FinalAnswer("done"), TokenUsage(1, 1))])
    with pytest.raises(ValueError, match="latency"):
        run(adapter, budget, Recorder())
    assert budget.usage.action_steps == 1 and budget.usage.reserved_tokens == 0


@pytest.mark.parametrize(("elapsed", "expired"), [(0.0, False), (4.999, False), (5.0, True), (5.001, True)])
def test_time_exhausted_property_includes_deadline(elapsed: float, expired: bool) -> None:
    now = [0.0]
    budget = BudgetTracker(RunBudget(1, 1, 10, 5), clock=lambda: now[0])
    now[0] = elapsed
    assert budget.time_exhausted is expired
