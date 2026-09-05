import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from time import monotonic

from .model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    ModelProviderError,
    ModelResponse,
    TokenUsage,
)


def _require_positive(
    value: int | float,
    field_name: str,
) -> None:
    if value <= 0:
        raise ValueError(
            f"{field_name} must be positive"
        )


def _require_non_negative(
    value: int,
    field_name: str,
) -> None:
    if value < 0:
        raise ValueError(
            f"{field_name} must not be negative"
        )


@dataclass(frozen=True)
class RunBudget:
    """Limits for one complete Agent run."""

    max_model_calls: int
    max_action_steps: int
    max_total_tokens: int
    max_elapsed_seconds: float

    def __post_init__(self) -> None:
        _require_positive(
            self.max_model_calls,
            "max_model_calls",
        )
        _require_positive(
            self.max_action_steps,
            "max_action_steps",
        )
        _require_positive(
            self.max_total_tokens,
            "max_total_tokens",
        )
        _require_positive(
            self.max_elapsed_seconds,
            "max_elapsed_seconds",
        )


@dataclass(frozen=True)
class RunUsage:
    """Consumed resources at one point in a run."""

    model_calls: int = 0
    action_steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reserved_tokens: int = 0

    def __post_init__(self) -> None:
        _require_non_negative(
            self.model_calls,
            "model_calls",
        )
        _require_non_negative(
            self.action_steps,
            "action_steps",
        )
        _require_non_negative(
            self.input_tokens,
            "input_tokens",
        )
        _require_non_negative(
            self.output_tokens,
            "output_tokens",
        )
        _require_non_negative(
            self.reserved_tokens,
            "reserved_tokens",
        )

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reserved_tokens
        )


class BudgetStopReason(StrEnum):
    MODEL_CALLS_EXHAUSTED = (
        "model_calls_exhausted"
    )
    ACTION_STEPS_EXHAUSTED = (
        "action_steps_exhausted"
    )
    TOKENS_EXHAUSTED = "tokens_exhausted"
    TIME_EXHAUSTED = "time_exhausted"


Clock = Callable[[], float]


class BudgetTracker:
    """Tracks resource consumption for one Agent run."""

    def __init__(
        self,
        budget: RunBudget,
        *,
        clock: Clock = monotonic,
    ) -> None:
        self._budget = budget
        self._clock = clock
        self._started_at = clock()
        self._usage = RunUsage()
        self._pending_reserved_tokens: int | None = None

    @property
    def usage(self) -> RunUsage:
        return self._usage

    @property
    def elapsed_seconds(self) -> float:
        started_at = _finite_clock_value(self._started_at)
        current = _finite_clock_value(self._clock())
        elapsed = current - started_at
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed_seconds must be finite and >= 0")
        return elapsed

    @property
    def time_exhausted(self) -> bool:
        return self.elapsed_seconds >= self._budget.max_elapsed_seconds

    def stop_reason(self) -> BudgetStopReason | None:
        elapsed_seconds = self.elapsed_seconds

        # TODO：依次检查：
        # 1. elapsed_seconds
        if elapsed_seconds >= self._budget.max_elapsed_seconds:
            return BudgetStopReason.TIME_EXHAUSTED
        # 2. model_calls\
        if self._usage.model_calls >= self._budget.max_model_calls:
            return BudgetStopReason.MODEL_CALLS_EXHAUSTED
        # 3. action_steps
        if self._usage.action_steps >= self._budget.max_action_steps:
            return BudgetStopReason.ACTION_STEPS_EXHAUSTED
        # 4. total_tokens
        if self._usage.total_tokens >= self._budget.max_total_tokens:
            return BudgetStopReason.TOKENS_EXHAUSTED
        #
        # 达到限制时使用 >=，返回对应的 BudgetStopReason。
        # 尚未达到任何限制则返回 None。
        return None

    def reserve_model_call(
        self,
        reserved_tokens: int,
    ) -> BudgetStopReason | None:
        """Reserve one model call and its possible token consumption."""

        _require_positive(
            reserved_tokens,
            "reserved_tokens",
        )
        if self._pending_reserved_tokens is not None:
            raise RuntimeError(
                "a model call already has a pending reservation"
            )

        reason = self.stop_reason()
        if reason is not None:
            return reason

        projected_tokens = (
            self._usage.total_tokens
            + reserved_tokens
        )
        if projected_tokens > self._budget.max_total_tokens:
            return BudgetStopReason.TOKENS_EXHAUSTED

        self._usage = replace(
            self._usage,
            model_calls=self._usage.model_calls + 1,
            reserved_tokens=(
                self._usage.reserved_tokens
                + reserved_tokens
            ),
        )
        self._pending_reserved_tokens = reserved_tokens
        return None

    def _settle_token_reservation(self) -> None:
        reserved_tokens = self._pending_reserved_tokens
        if reserved_tokens is None:
            raise RuntimeError(
                "no pending model call reservation"
            )

        self._usage = replace(
            self._usage,
            reserved_tokens=(
                self._usage.reserved_tokens
                - reserved_tokens
            ),
        )
        self._pending_reserved_tokens = None

    def retain_model_call_reservation(self) -> None:
        """Keep reserved tokens when actual consumption is unknown."""
        if self._pending_reserved_tokens is None:
            raise RuntimeError(
                "no pending model call reservation"
            )
        self._pending_reserved_tokens = None

    def release_model_call_reservation(self) -> None:
        """Release reserved tokens when no consumption occurred."""
        self._settle_token_reservation()

    def _record_token_usage(
        self,
        usage: TokenUsage,
    ) -> None:
        self._usage = replace(
            self._usage,
            input_tokens=(
                self._usage.input_tokens
                + usage.input_tokens
            ),
            output_tokens=(
                self._usage.output_tokens
                + usage.output_tokens
            ),
        )

    def record_model_response(
        self,
        response: ModelResponse,
    ) -> None:
        self._settle_token_reservation()
        self._record_token_usage(response.usage)
        self._usage = replace(
            self._usage,
            action_steps=self._usage.action_steps + 1,
        )

    def record_invalid_model_output(
        self,
        error: InvalidModelOutputError,
    ) -> None:
        if error.usage is not None:
            self._settle_token_reservation()
            self._record_token_usage(error.usage)
            return

        self.retain_model_call_reservation()

    def record_provider_error(
        self,
        error: ModelProviderError,
    ) -> None:
        state = error.consumption_state

        if state is ConsumptionState.NO_CONSUMPTION:
            self.release_model_call_reservation()
            return

        if state is ConsumptionState.UNKNOWN_CONSUMPTION:
            self.retain_model_call_reservation()
            return

        if state is ConsumptionState.ACTUAL_USAGE:
            usage = error.usage
            if usage is None:
                raise AssertionError(
                    "actual consumption must contain usage"
                )
            self._settle_token_reservation()
            self._record_token_usage(usage)
            return

        raise AssertionError(
            f"unsupported consumption state: {state}"
        )


def _finite_clock_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("clock must return a number (bool is excluded)")
    try:
        seconds = float(value)
    except OverflowError as error:
        raise ValueError("clock must return a finite number") from error
    if not math.isfinite(seconds):
        raise ValueError("clock must return a finite number")
    return seconds
