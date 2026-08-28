"""Adapter from JSON-text model responses to Kernel actions."""
'''Model - Adapter - Provider'''
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .json_text_messages import (
    ProviderMessage,
    build_provider_messages,
)

from .json_action_parser import parse_json_action
from .kernel_types import HistoryItem, ToolSpec
from .model_adapter import (
    ModelCallBudget,
    ModelProviderError,
    ModelResponse,
    TokenUsage,
)


class ProviderError(RuntimeError):
    """The underlying provider request failed."""


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int
    output_tokens: int


class TextProvider(Protocol):
    def complete(
        self,
        messages: Sequence[ProviderMessage],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
    ) -> ProviderResponse:
        ...


class JsonTextModelAdapter:
    def __init__(self, provider: TextProvider) -> None:
        self._provider = provider

    def complete(
        self,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        provider_messages = build_provider_messages(history)
        
        try:
            raw_response = self._provider.complete(
                messages=provider_messages,
                tools=tools,
                max_output_tokens=budget.max_output_tokens,
            )
        except ProviderError as exc:
            raise ModelProviderError(
                "provider request failed"
            ) from exc

        action = parse_json_action(raw_response.text)

        usage = TokenUsage(
            input_tokens=raw_response.input_tokens,
            output_tokens=raw_response.output_tokens,
        )

        return ModelResponse(
            action=action,
            usage=usage,
        )