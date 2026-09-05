"""Coordinate model attempts, budget settlement and process observation."""

import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
from enum import StrEnum
from typing import Protocol, cast

from .kernel_types import HistoryItem, ModelFeedback, ToolSpec
from .model_adapter import (
    ConsumptionState, InvalidModelOutputError, ModelAdapter, ModelCallBudget,
    ModelProviderError, ModelResponse, Retryability,
)
from .run_budget import BudgetStopReason, BudgetTracker, RunUsage


class ModelRetryKind(StrEnum):
    PROVIDER = "provider"
    INVALID_OUTPUT = "invalid_output"


RetryObserver = Callable[
    [int, ModelRetryKind, ModelFeedback | None, RunUsage], None
]


class ModelObservationError(RuntimeError):
    pass


def _require_retry_limit(value: object, field: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field} must be an integer (bool is excluded)")
    if value < 0:
        raise ValueError(f"{field} must not be negative")


def _notify_model_retry(
    observer: RetryObserver | None,
    attempt_index: int,
    kind: ModelRetryKind,
    feedback: ModelFeedback | None,
    usage: RunUsage,
) -> None:
    if observer is None:
        return
    try:
        result = cast(Callable[..., object], observer)(
            attempt_index, kind, deepcopy(feedback), replace(usage),
        )
        if result is not None:
            raise TypeError("on_retry must return None")
    except Exception as error:
        failure = ModelObservationError(
            f"observe model retry after attempt {attempt_index} failed"
        )
        raise failure from error


class ModelCallObserver(Protocol):
    def prepared(self, attempt_index: int, reserved_tokens: int, usage: RunUsage) -> None: ...

    def completed(
        self, attempt_index: int, response: ModelResponse, usage: RunUsage, latency: float,
    ) -> None: ...

    def failed(
        self, attempt_index: int, error: Exception, usage: RunUsage, latency: float,
    ) -> None: ...


def _validate_observer(observer: ModelCallObserver | None) -> None:
    if observer is not None:
        for name in ("prepared", "completed", "failed"):
            if not callable(getattr(observer, name, None)):
                raise TypeError(f"observer.{name} must be callable")


def _notify_model_attempt(
    observer: ModelCallObserver, stage: str, attempt_index: int, *values: object,
) -> None:
    try:
        callback = cast(Callable[..., object], getattr(observer, stage))
        if callback(attempt_index, *deepcopy(values)) is not None:
            raise TypeError(f"observer.{stage} must return None")
    except Exception as error:
        raise ModelObservationError(
            f"observe model {stage} for attempt {attempt_index} failed"
        ) from error


def _failure_snapshot(error: Exception) -> Exception:
    if isinstance(error, ModelProviderError):
        return ModelProviderError(
            "provider request failed", consumption_state=error.consumption_state,
            retryability=error.retryability, usage=error.usage,
        )
    if isinstance(error, InvalidModelOutputError):
        return InvalidModelOutputError("invalid model output", usage=error.usage)
    return RuntimeError("unexpected model failure")


def _latency(tracker: BudgetTracker, started_at: float) -> float:
    duration = tracker.elapsed_seconds - started_at
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("model latency must be finite and >= 0")
    return duration


def complete_model_call(
    adapter: ModelAdapter,
    tracker: BudgetTracker,
    *,
    history: Sequence[HistoryItem],
    tools: Sequence[ToolSpec],
    budget: ModelCallBudget,
    reserved_tokens: int,
    observer: ModelCallObserver | None = None,
) -> ModelResponse | BudgetStopReason:
    """Complete one model call and settle its reservation exactly once."""
    _validate_observer(observer)
    stop_reason = tracker.reserve_model_call(reserved_tokens)
    if stop_reason is not None:
        return stop_reason

    attempt_index = tracker.usage.model_calls
    started_at = 0.0
    if observer is not None:
        try:
            started_at = tracker.elapsed_seconds
            _notify_model_attempt(
                observer, "prepared", attempt_index, reserved_tokens, tracker.usage,
            )
            if tracker.time_exhausted:
                tracker.release_model_call_reservation()
                return BudgetStopReason.TIME_EXHAUSTED
        except Exception:
            tracker.release_model_call_reservation()
            raise

    try:
        response = adapter.complete(history=history, tools=tools, budget=budget)
    except InvalidModelOutputError as error:
        tracker.record_invalid_model_output(error)
        if observer is not None:
            _notify_model_attempt(
                observer, "failed", attempt_index,
                _failure_snapshot(error), tracker.usage, _latency(tracker, started_at),
            )
        raise
    except ModelProviderError as error:
        tracker.record_provider_error(error)
        if observer is not None:
            _notify_model_attempt(
                observer, "failed", attempt_index,
                _failure_snapshot(error), tracker.usage, _latency(tracker, started_at),
            )
        raise
    except Exception as error:
        tracker.retain_model_call_reservation()
        if observer is not None:
            _notify_model_attempt(
                observer, "failed", attempt_index,
                _failure_snapshot(error), tracker.usage, _latency(tracker, started_at),
            )
        raise

    tracker.record_model_response(response)
    if observer is not None:
        _notify_model_attempt(
            observer, "completed", attempt_index,
            response, tracker.usage, _latency(tracker, started_at),
        )
    return response


def complete_model_with_retry(
    adapter: ModelAdapter,
    tracker: BudgetTracker,
    *,
    history: Sequence[HistoryItem],
    tools: Sequence[ToolSpec],
    budget: ModelCallBudget,
    reserved_tokens: int,
    max_provider_retries: int,
    max_invalid_output_retries: int = 0,
    on_retry: RetryObserver | None = None,
    observer: ModelCallObserver | None = None,
) -> ModelResponse | BudgetStopReason:
    """Retry classified failures within the run budget."""

    _require_retry_limit(max_provider_retries, "max_provider_retries")
    _require_retry_limit(max_invalid_output_retries, "max_invalid_output_retries")
    if on_retry is not None and not callable(on_retry):
        raise TypeError("on_retry must be callable or None")

    _validate_observer(observer)

    provider_retries_used = 0
    invalid_output_retries_used = 0
    retry_history: tuple[HistoryItem, ...] = tuple(history)
    while True:
        feedback: ModelFeedback | None = None
        try:
            return complete_model_call(
                adapter,
                tracker,
                history=retry_history,
                tools=tools,
                budget=budget,
                reserved_tokens=reserved_tokens,
                observer=observer,
            )
        except InvalidModelOutputError as error:
            if invalid_output_retries_used >= max_invalid_output_retries:
                raise
            feedback = ModelFeedback(
                error_message=str(error).strip()
                or "model output did not satisfy the action contract",
            )
            kind = ModelRetryKind.INVALID_OUTPUT
            invalid_output_retries_used += 1
        except ModelProviderError as error:
            should_retry = (
                error.retryability is Retryability.RETRYABLE
                and provider_retries_used < max_provider_retries
            )
            if not should_retry:
                raise
            kind = ModelRetryKind.PROVIDER
            provider_retries_used += 1

        _notify_model_retry(
            on_retry, tracker.usage.model_calls, kind, feedback, tracker.usage,
        )
        if feedback is not None:
            retry_history = (*retry_history, feedback)
