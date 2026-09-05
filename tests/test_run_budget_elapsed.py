from importlib import import_module
from typing import Callable, cast

import pytest

from code_agent.run_budget import BudgetStopReason, BudgetTracker, RunBudget


class FakeClock:
    def __init__(self) -> None:
        self.value: object = 10.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return cast(float, self.value)


def _tracker(clock: Callable[[], float]) -> BudgetTracker:
    return BudgetTracker(RunBudget(3, 3, 100, 5), clock=clock)


def _elapsed(tracker: BudgetTracker) -> float:
    value: object = getattr(tracker, "elapsed_seconds")
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def test_elapsed_uses_tracker_start_without_restarting_timer() -> None:
    clock = FakeClock()
    tracker = _tracker(clock)
    assert clock.calls == 1
    assert _elapsed(tracker) == 0
    clock.value = 13.0
    assert _elapsed(tracker) == 3
    assert tracker.reserve_model_call(1) is None
    clock.value = 14.0
    assert _elapsed(tracker) == 4
    assert tracker.usage.model_calls == 1


@pytest.mark.parametrize(("elapsed", "reason"), [
    (4.999, None),
    (5, BudgetStopReason.TIME_EXHAUSTED),
    (5.001, BudgetStopReason.TIME_EXHAUSTED),
])
def test_stop_reason_shares_elapsed_boundary(
    elapsed: float, reason: BudgetStopReason | None,
) -> None:
    clock = FakeClock()
    tracker = _tracker(clock)
    clock.value = 10 + elapsed
    assert _elapsed(tracker) == pytest.approx(elapsed)
    assert tracker.stop_reason() is reason


def test_elapsed_is_read_only() -> None:
    clock = FakeClock()
    tracker = _tracker(clock)
    assert _elapsed(tracker) == 0
    with pytest.raises(AttributeError):
        setattr(tracker, "elapsed_seconds", 5)
    assert _elapsed(tracker) == 0


@pytest.mark.parametrize(("value", "error"), [
    (True, TypeError), (None, TypeError), ("12", TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError),
    (-float("inf"), ValueError), (10**400, ValueError),
])
def test_elapsed_rejects_invalid_clock_reading(
    value: object, error: type[Exception],
) -> None:
    clock = FakeClock()
    tracker = _tracker(clock)
    before = tracker.usage
    clock.value = value
    with pytest.raises(error, match="clock"):
        _elapsed(tracker)
    assert tracker.usage == before


@pytest.mark.parametrize("start", [True, float("nan")])
def test_elapsed_rejects_invalid_existing_start_before_reading_clock(
    start: object,
) -> None:
    clock = FakeClock()
    clock.value = start
    tracker = _tracker(clock)
    clock.value = 12
    error = TypeError if isinstance(start, bool) else ValueError
    with pytest.raises(error, match="clock"):
        _elapsed(tracker)
    assert clock.calls == 1


@pytest.mark.parametrize(("start", "end"), [(10.0, 9.999), (-1e308, 1e308)])
def test_elapsed_rejects_backwards_or_overflowed_duration(
    start: float, end: float,
) -> None:
    clock = FakeClock()
    clock.value = start
    tracker = _tracker(clock)
    clock.value = end
    with pytest.raises(ValueError, match="elapsed_seconds"):
        _elapsed(tracker)


def test_elapsed_propagates_unexpected_clock_failure() -> None:
    failure = RuntimeError("clock broke")
    reads = 0

    def clock() -> float:
        nonlocal reads
        reads += 1
        if reads > 1:
            raise failure
        return 1.0

    tracker = _tracker(clock)
    with pytest.raises(RuntimeError) as caught:
        _elapsed(tracker)
    assert caught.value is failure


def test_stop_reason_uses_the_elapsed_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("code_agent.run_budget")
    tracker = _tracker(lambda: 0.0)
    monkeypatch.setattr(
        module.BudgetTracker, "elapsed_seconds", property(lambda self: 5.0),
        raising=False,
    )
    assert tracker.stop_reason() is BudgetStopReason.TIME_EXHAUSTED
