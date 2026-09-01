"""Adapter for providers with native tool-calling messages."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from .kernel_types import (
    FinalAnswer,
    HistoryItem,
    Message,
    ModelFeedback,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    Retryability,
    TokenUsage,
)


@dataclass(frozen=True)
class NativeProviderMessage:
    role: str
    content: str


@dataclass(frozen=True)
class NativeProviderToolCall:
    call_id: str
    tool_name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class NativeProviderToolResult:
    call_id: str
    content: str
    is_error: bool


NativeProviderHistoryItem: TypeAlias = (
    NativeProviderMessage
    | NativeProviderToolCall
    | NativeProviderToolResult
)


@dataclass(frozen=True)
class NativeProviderResponse:
    tool_calls: tuple[NativeProviderToolCall, ...]
    final_text: str | None
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None


class NativeProviderError(RuntimeError):
    """A native tool-calling provider request failed."""

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


class NativeToolProvider(Protocol):
    def complete(
        self,
        history: Sequence[NativeProviderHistoryItem],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
    ) -> NativeProviderResponse:
        ...


def build_native_provider_history(
    history: Sequence[HistoryItem],
) -> tuple[NativeProviderHistoryItem, ...]:
    """Preserve native tool items while translating Kernel history."""

    provider_history: list[NativeProviderHistoryItem] = []
    for item in history:
        if isinstance(item, Message):
            provider_history.append(
                NativeProviderMessage(
                    role=item.role.value,
                    content=item.content,
                )
            )
            continue

        if isinstance(item, ModelFeedback):
            provider_history.append(
                NativeProviderMessage(
                    role="user",
                    content=(
                        "Previous model output was invalid: "
                        f"{item.error_message}. Return exactly one "
                        "action matching the declared tool-calling "
                        "contract."
                    ),
                )
            )
            continue

        if isinstance(item, ToolCall):
            provider_history.append(
                NativeProviderToolCall(
                    call_id=item.call_id,
                    tool_name=item.tool_name,
                    arguments=item.arguments,
                )
            )
            continue

        if isinstance(item, ToolResult):
            provider_history.append(
                NativeProviderToolResult(
                    call_id=item.call_id,
                    content=item.content,
                    is_error=item.is_error,
                )
            )
            continue

    return tuple(provider_history)


class NativeToolCallingModelAdapter:
    """Normalize one native provider response into one Kernel action."""

    def __init__(self, provider: NativeToolProvider) -> None:
        self._provider = provider

    def complete(
        self,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        provider_history = build_native_provider_history(history)

        try:
            raw_response = self._provider.complete(
                history=provider_history,
                tools=tools,
                max_output_tokens=budget.max_output_tokens,
            )
        except NativeProviderError as error:
            raise ModelProviderError(
                "provider request failed",
                consumption_state=error.consumption_state,
                retryability=error.retryability,
                usage=error.usage,
            ) from error

        usage = TokenUsage(
            input_tokens=raw_response.input_tokens,
            output_tokens=raw_response.output_tokens,
        )
        action = _normalize_native_action(raw_response, usage)
        return ModelResponse(action=action, usage=usage)


def _normalize_native_action(
    response: NativeProviderResponse,
    usage: TokenUsage,
) -> ToolCall | FinalAnswer:
    if response.finish_reason not in {None, "stop", "tool_calls"}:
        raise InvalidModelOutputError(
            "provider stopped without completing an action: "
            f"{response.finish_reason}",
            usage=usage,
        )

    tool_call_count = len(response.tool_calls)
    has_final_text = response.final_text is not None

    if tool_call_count > 1:
        raise InvalidModelOutputError(
            "provider returned multiple tool calls",
            usage=usage,
        )

    if tool_call_count == 1 and has_final_text:
        raise InvalidModelOutputError(
            "provider returned a tool call and final text",
            usage=usage,
        )

    if tool_call_count == 0 and not has_final_text:
        raise InvalidModelOutputError(
            "provider returned no action",
            usage=usage,
        )

    try:
        if tool_call_count == 1:
            raw_call = response.tool_calls[0]
            return ToolCall(
                call_id=raw_call.call_id,
                tool_name=raw_call.tool_name,
                arguments=raw_call.arguments,
            )

        final_text = response.final_text
        if final_text is None:
            raise AssertionError("final text must be present")
        return FinalAnswer(content=final_text)
    except ValueError as error:
        raise InvalidModelOutputError(
            str(error),
            usage=usage,
        ) from error
