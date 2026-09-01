"""Model Adapter interface and deterministic fake."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias

from .kernel_types import (
    Action,
    HistoryItem,
    ToolSpec,
)


class FakeModelExhaustedError(RuntimeError):
    """The fake has no configured outcome left."""


@dataclass(frozen=True)
class ModelCallBudget:
    max_output_tokens: int

    def __post_init__(self) -> None:
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens must not be negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must not be negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ConsumptionState(StrEnum):
    """How much provider consumption is known after a failure."""

    NO_CONSUMPTION = "no_consumption"
    ACTUAL_USAGE = "actual_usage"
    UNKNOWN_CONSUMPTION = "unknown_consumption"


class Retryability(StrEnum):
    """Whether the caller may repeat a failed provider request."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"


class ModelProviderError(RuntimeError):
    """The provider request failed with explicit usage semantics."""

    def __init__(
        self,
        message: str,
        *,
        consumption_state: ConsumptionState = (
            ConsumptionState.UNKNOWN_CONSUMPTION
        ),
        retryability: Retryability = Retryability.NON_RETRYABLE,
        usage: TokenUsage | None = None,
    ) -> None:
        if (
            consumption_state is ConsumptionState.ACTUAL_USAGE
            and usage is None
        ):
            raise ValueError(
                "usage is required for actual consumption"
            )
        if (
            consumption_state is not ConsumptionState.ACTUAL_USAGE
            and usage is not None
        ):
            raise ValueError(
                "usage must be absent unless consumption is actual"
            )

        super().__init__(message)
        self.consumption_state = consumption_state
        self.retryability = retryability
        self.usage = usage


class InvalidModelOutputError(ValueError):
    """The provider response cannot become a valid Action."""

    def __init__(
        self,
        message: str,
        *,
        usage: TokenUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class ModelResponse:
    action: Action
    usage: TokenUsage


@dataclass(frozen=True)
class ModelCall:
    history: tuple[HistoryItem, ...]
    tools: tuple[ToolSpec, ...]
    budget: ModelCallBudget


class ModelAdapter(Protocol):
    def complete(
        self,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        """Return one normalized model response."""
        ...


FakeOutcome: TypeAlias = (
    ModelResponse
    | ModelProviderError
    | InvalidModelOutputError
)


class FakeModelAdapter:
    def __init__(
        self,
        outcomes: Sequence[FakeOutcome],
    ) -> None:
        self._outcomes = tuple(outcomes)
        self._index = 0
        self.calls: list[ModelCall] = []

    def complete(
        self,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        self.calls.append(
            ModelCall(
                history=tuple(history),
                tools=tuple(tools),
                budget=budget,
            )
        )
        if self._index >= len(self._outcomes):
            raise FakeModelExhaustedError(
                "fake model has no configured outcome left"
            )
        outcome = self._outcomes[self._index]
        self._index += 1
        # 否则返回 ModelResponse
        if isinstance(outcome, ModelProviderError):
            raise outcome
        if isinstance(outcome, InvalidModelOutputError):
            raise outcome

        return outcome
