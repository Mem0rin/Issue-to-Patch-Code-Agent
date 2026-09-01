"""Coordinate one model call with run-budget accounting."""

from collections.abc import Sequence

from .kernel_types import HistoryItem, ModelFeedback, ToolSpec
from .model_adapter import (
    InvalidModelOutputError,
    ModelAdapter,
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    Retryability,
)
from .run_budget import BudgetStopReason, BudgetTracker


def complete_model_call(
    adapter: ModelAdapter,
    tracker: BudgetTracker,
    *,
    history: Sequence[HistoryItem],
    tools: Sequence[ToolSpec],
    budget: ModelCallBudget,
    reserved_tokens: int,
) -> ModelResponse | BudgetStopReason:
    """Complete one model call and settle its reservation exactly once."""

    stop_reason = tracker.reserve_model_call(reserved_tokens)
    if stop_reason is not None:
        return stop_reason

    try:
        response = adapter.complete(
            history=history,
            tools=tools,
            budget=budget,
        )
    except InvalidModelOutputError as error:
        tracker.record_invalid_model_output(error)
        raise
    except ModelProviderError as error:
        tracker.record_provider_error(error)
        raise
    except Exception:
        tracker.retain_model_call_reservation()
        raise

    tracker.record_model_response(response)
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
) -> ModelResponse | BudgetStopReason:
    """Retry classified failures within the run budget."""

    if max_provider_retries < 0:
        raise ValueError(
            "max_provider_retries must not be negative"
        )
    if max_invalid_output_retries < 0:
        raise ValueError(
            "max_invalid_output_retries must not be negative"
        )

    provider_retries_used = 0
    invalid_output_retries_used = 0
    retry_history: tuple[HistoryItem, ...] = tuple(history)
    while True:
        try:
            return complete_model_call(
                adapter,
                tracker,
                history=retry_history,
                tools=tools,
                budget=budget,
                reserved_tokens=reserved_tokens,
            )
        except InvalidModelOutputError as error:
            if (
                invalid_output_retries_used
                >= max_invalid_output_retries
            ):
                raise

            error_message = str(error).strip()
            if not error_message:
                error_message = (
                    "model output did not satisfy the action contract"
                )
            retry_history = (
                *retry_history,
                ModelFeedback(error_message=error_message),
            )
            invalid_output_retries_used += 1
        except ModelProviderError as error:
            should_retry = (
                error.retryability is Retryability.RETRYABLE
                and provider_retries_used < max_provider_retries
            )
            if not should_retry:
                raise
            provider_retries_used += 1
