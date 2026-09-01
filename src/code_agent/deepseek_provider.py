"""DeepSeek Chat Completions provider for native tool calling."""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .kernel_types import ToolSpec
from .model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    Retryability,
    TokenUsage,
)
from .native_tool_adapter import (
    NativeProviderError,
    NativeProviderHistoryItem,
    NativeProviderMessage,
    NativeProviderResponse,
    NativeProviderToolCall,
    NativeProviderToolResult,
)


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_MAX_RESPONSE_BYTES = 2_000_000


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str = field(repr=False)
    model: str = DEFAULT_DEEPSEEK_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key must not be blank")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if not self.base_url.startswith("https://"):
            raise ValueError("base_url must use https")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        timeout_seconds: float = 60.0,
    ) -> "DeepSeekConfig":
        source = os.environ if environ is None else environ
        api_key = source.get("DEEPSEEK_API_KEY")
        if api_key is None or not api_key.strip():
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is required"
            )
        return cls(
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )


@dataclass(frozen=True)
class DeepSeekHttpResponse:
    status_code: int
    body: bytes


class DeepSeekTransportError(RuntimeError):
    """The HTTP result is unknown because transport failed."""


class DeepSeekTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> DeepSeekHttpResponse:
        ...


class UrllibDeepSeekTransport:
    """Small synchronous HTTPS transport with a response-size limit."""

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> DeepSeekHttpResponse:
        encoded_payload = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            url=url,
            data=encoded_payload,
            headers=dict(headers),
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                body = _read_limited_body(response)
                return DeepSeekHttpResponse(
                    status_code=response.status,
                    body=body,
                )
        except HTTPError as error:
            return DeepSeekHttpResponse(
                status_code=error.code,
                body=_read_limited_body(error),
            )
        except (URLError, TimeoutError, OSError) as error:
            raise DeepSeekTransportError(
                "DeepSeek HTTP transport failed"
            ) from error


def _read_limited_body(response: object) -> bytes:
    read = getattr(response, "read", None)
    if not callable(read):
        raise DeepSeekTransportError(
            "DeepSeek HTTP response is not readable"
        )
    body = cast(bytes, read(_MAX_RESPONSE_BYTES + 1))
    if len(body) > _MAX_RESPONSE_BYTES:
        raise DeepSeekTransportError(
            "DeepSeek HTTP response exceeded size limit"
        )
    return body


class DeepSeekProvider:
    """Map DeepSeek Chat Completions to native provider types."""

    def __init__(
        self,
        config: DeepSeekConfig,
        *,
        transport: DeepSeekTransport | None = None,
    ) -> None:
        self._config = config
        self._transport = (
            UrllibDeepSeekTransport()
            if transport is None
            else transport
        )

    def complete(
        self,
        history: Sequence[NativeProviderHistoryItem],
        tools: Sequence[ToolSpec],
        max_output_tokens: int,
    ) -> NativeProviderResponse:
        if not history:
            raise NativeProviderError(
                "DeepSeek request requires non-empty history",
                consumption_state=ConsumptionState.NO_CONSUMPTION,
            )

        payload = _build_request_payload(
            model=self._config.model,
            history=history,
            tools=tools,
            max_output_tokens=max_output_tokens,
        )
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._transport.post_json(
                url=(
                    self._config.base_url.rstrip("/")
                    + "/chat/completions"
                ),
                headers=headers,
                payload=payload,
                timeout_seconds=self._config.timeout_seconds,
            )
        except DeepSeekTransportError as error:
            raise NativeProviderError(
                "DeepSeek request outcome is unknown",
                consumption_state=(
                    ConsumptionState.UNKNOWN_CONSUMPTION
                ),
                retryability=Retryability.RETRYABLE,
            ) from error

        if response.status_code < 200 or response.status_code >= 300:
            retryability = (
                Retryability.RETRYABLE
                if response.status_code == 429
                or response.status_code >= 500
                else Retryability.NON_RETRYABLE
            )
            raise NativeProviderError(
                "DeepSeek API request failed with status "
                f"{response.status_code}",
                consumption_state=(
                    ConsumptionState.UNKNOWN_CONSUMPTION
                ),
                retryability=retryability,
            )

        try:
            decoded: object = json.loads(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise NativeProviderError(
                "DeepSeek API returned invalid JSON",
                consumption_state=(
                    ConsumptionState.UNKNOWN_CONSUMPTION
                ),
            ) from error

        if not isinstance(decoded, dict):
            raise NativeProviderError(
                "DeepSeek API response must be an object",
                consumption_state=(
                    ConsumptionState.UNKNOWN_CONSUMPTION
                ),
            )

        response_object = cast(dict[str, object], decoded)
        return _parse_response(response_object)


def _build_request_payload(
    *,
    model: str,
    history: Sequence[NativeProviderHistoryItem],
    tools: Sequence[ToolSpec],
    max_output_tokens: int,
) -> dict[str, object]:
    messages: list[object] = []
    for item in history:
        if isinstance(item, NativeProviderMessage):
            messages.append({
                "role": item.role,
                "content": item.content,
            })
            continue

        if isinstance(item, NativeProviderToolCall):
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.call_id,
                    "type": "function",
                    "function": {
                        "name": item.tool_name,
                        "arguments": json.dumps(
                            dict(item.arguments),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }],
            })
            continue

        if isinstance(item, NativeProviderToolResult):
            messages.append({
                "role": "tool",
                "tool_call_id": item.call_id,
                "content": json.dumps(
                    {
                        "content": item.content,
                        "is_error": item.is_error,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            })
            continue

    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": dict(spec.input_schema),
                },
            }
            for spec in tools
        ]
        payload["tool_choice"] = "auto"
    return payload


def _parse_response(
    response: Mapping[str, object],
) -> NativeProviderResponse:
    usage = _parse_usage(response)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise NativeProviderError(
            "DeepSeek API must return exactly one choice",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )

    choice = _require_object(
        choices[0],
        "DeepSeek choice must be an object",
        usage,
    )
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str):
        raise NativeProviderError(
            "DeepSeek finish_reason must be a string",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )
    if finish_reason == "insufficient_system_resource":
        raise NativeProviderError(
            "DeepSeek inference resources were insufficient",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            retryability=Retryability.RETRYABLE,
            usage=usage,
        )

    message = _require_object(
        choice.get("message"),
        "DeepSeek choice message must be an object",
        usage,
    )
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise NativeProviderError(
            "DeepSeek message content must be text or null",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )

    raw_tool_calls = message.get("tool_calls", [])
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise NativeProviderError(
            "DeepSeek tool_calls must be a list",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )

    tool_calls = tuple(
        _parse_tool_call(item, usage)
        for item in raw_tool_calls
    )
    return NativeProviderResponse(
        tool_calls=tool_calls,
        final_text=content,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        finish_reason=finish_reason,
    )


def _parse_usage(response: Mapping[str, object]) -> TokenUsage:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise NativeProviderError(
            "DeepSeek response did not include usage",
            consumption_state=(
                ConsumptionState.UNKNOWN_CONSUMPTION
            ),
        )

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
    ):
        raise NativeProviderError(
            "DeepSeek usage tokens must be integers",
            consumption_state=(
                ConsumptionState.UNKNOWN_CONSUMPTION
            ),
        )
    try:
        return TokenUsage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        )
    except ValueError as error:
        raise NativeProviderError(
            "DeepSeek usage tokens must not be negative",
            consumption_state=(
                ConsumptionState.UNKNOWN_CONSUMPTION
            ),
        ) from error


def _require_object(
    value: object,
    message: str,
    usage: TokenUsage,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise NativeProviderError(
            message,
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )
    return cast(dict[str, object], value)


def _parse_tool_call(
    value: object,
    usage: TokenUsage,
) -> NativeProviderToolCall:
    tool_call = _require_object(
        value,
        "DeepSeek tool call must be an object",
        usage,
    )
    call_id = tool_call.get("id")
    if not isinstance(call_id, str):
        raise NativeProviderError(
            "DeepSeek tool call id must be a string",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )
    if tool_call.get("type") != "function":
        raise NativeProviderError(
            "DeepSeek tool call type must be function",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )

    function = _require_object(
        tool_call.get("function"),
        "DeepSeek tool call function must be an object",
        usage,
    )
    name = function.get("name")
    arguments_text = function.get("arguments")
    if not isinstance(name, str):
        raise NativeProviderError(
            "DeepSeek tool name must be a string",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )
    if not isinstance(arguments_text, str):
        raise NativeProviderError(
            "DeepSeek tool arguments must be JSON text",
            consumption_state=ConsumptionState.ACTUAL_USAGE,
            usage=usage,
        )

    try:
        decoded_arguments: object = json.loads(arguments_text)
    except json.JSONDecodeError as error:
        raise InvalidModelOutputError(
            "DeepSeek tool arguments must be valid JSON",
            usage=usage,
        ) from error
    if not isinstance(decoded_arguments, dict):
        raise InvalidModelOutputError(
            "DeepSeek tool arguments must be an object",
            usage=usage,
        )

    arguments = cast(dict[str, object], decoded_arguments)
    return NativeProviderToolCall(
        call_id=call_id,
        tool_name=name,
        arguments=arguments,
    )
