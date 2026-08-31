from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from time import monotonic

from .model_adapter import (
    InvalidModelOutputError,
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

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
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

    @property
    def usage(self) -> RunUsage:
        return self._usage

    def stop_reason(self) -> BudgetStopReason | None:
        elapsed_seconds = (
            self._clock() - self._started_at
        )

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
    ) -> BudgetStopReason | None:
        """Reserve one model call before contacting the provider."""

        # TODO：
        # 1. 调用 stop_reason()
        reason = self.stop_reason()
        # 2. 如果已经耗尽预算，直接返回停止原因
        if reason is not None:
            return reason
        # 3. 否则使用 replace() 将 model_calls 加一

        self._usage = replace(
            self._usage,
            model_calls=self._usage.model_calls + 1,
        )
        # 4. 返回 None
        return None

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
        self._record_token_usage(response.usage)
        self._usage = replace(
            self._usage,
            action_steps=self._usage.action_steps + 1,
        )

    def record_invalid_model_output(
        self,
        error: InvalidModelOutputError,
    ) -> None:
        # TODO：
        # 如果 error.usage 存在，就调用 _record_token_usage。
        # 不增加 action_steps。
        if error.usage is not None:
            self._record_token_usage(error.usage)
