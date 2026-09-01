import json
from collections.abc import Mapping

import pytest

from code_agent.deepseek_provider import (
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekConfig,
    DeepSeekHttpResponse,
    DeepSeekProvider,
    DeepSeekTransportError,
)
from code_agent.kernel_types import ToolSpec
from code_agent.model_adapter import (
    ConsumptionState,
    InvalidModelOutputError,
    Retryability,
    TokenUsage,
)
from code_agent.native_tool_adapter import (
    NativeProviderError,
    NativeProviderMessage,
    NativeProviderResponse,
    NativeProviderToolCall,
    NativeProviderToolResult,
)


class StubTransport:
    def __init__(
        self,
        *,
        response: DeepSeekHttpResponse | None = None,
        error: DeepSeekTransportError | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[
            tuple[
                str,
                dict[str, str],
                dict[str, object],
                float,
            ]
        ] = []

    def post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        timeout_seconds: float,
    ) -> DeepSeekHttpResponse:
        self.calls.append(
            (
                url,
                dict(headers),
                dict(payload),
                timeout_seconds,
            )
        )
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise AssertionError("stub transport requires an outcome")
        return self.response


def make_http_response(
    *,
    content: str | None = None,
    tool_calls: list[object] | None = None,
    finish_reason: str = "stop",
) -> DeepSeekHttpResponse:
    return DeepSeekHttpResponse(
        status_code=200,
        body=json.dumps({
            "choices": [{
                "finish_reason": finish_reason,
                "message": {
                    "content": content,
                    "tool_calls": tool_calls,
                },
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
            },
        }).encode("utf-8"),
    )


def make_config() -> DeepSeekConfig:
    return DeepSeekConfig(
        api_key="test-api-key",
        timeout_seconds=12.5,
    )


def test_config_loads_key_from_environment_without_repr_leak() -> None:
    config = DeepSeekConfig.from_environment({
        "DEEPSEEK_API_KEY": "test-secret-key",
    })

    assert config.api_key == "test-secret-key"
    assert config.model == "deepseek-v4-pro"
    assert "test-secret-key" not in repr(config)


def test_config_requires_environment_key() -> None:
    with pytest.raises(
        ValueError,
        match="DEEPSEEK_API_KEY environment variable is required",
    ):
        DeepSeekConfig.from_environment({})


def test_provider_builds_non_thinking_native_tool_request() -> None:
    transport = StubTransport(
        response=make_http_response(
            tool_calls=[{
                "id": "provider_call_1",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression":"1+1"}',
                },
            }],
            finish_reason="tool_calls",
        )
    )
    provider = DeepSeekProvider(
        make_config(),
        transport=transport,
    )
    spec = ToolSpec(
        name="calculator",
        description="Evaluate an arithmetic expression.",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
        },
    )

    response = provider.complete(
        history=[
            NativeProviderMessage(
                role="user",
                content="Calculate 1+1.",
            ),
            NativeProviderToolCall(
                call_id="previous_call",
                tool_name="calculator",
                arguments={"expression": "2+2"},
            ),
            NativeProviderToolResult(
                call_id="previous_call",
                content="4",
                is_error=False,
            ),
        ],
        tools=[spec],
        max_output_tokens=128,
    )

    assert response == NativeProviderResponse(
        tool_calls=(
            NativeProviderToolCall(
                call_id="provider_call_1",
                tool_name="calculator",
                arguments={"expression": "1+1"},
            ),
        ),
        final_text=None,
        input_tokens=20,
        output_tokens=5,
        finish_reason="tool_calls",
    )
    assert len(transport.calls) == 1
    url, headers, payload, timeout_seconds = transport.calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers == {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json",
    }
    assert payload["model"] == DEFAULT_DEEPSEEK_MODEL
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["stream"] is False
    assert payload["max_tokens"] == 128
    assert payload["tool_choice"] == "auto"
    assert timeout_seconds == 12.5

    messages = payload["messages"]
    assert isinstance(messages, list)
    assert messages[1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "previous_call",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"2+2"}',
            },
        }],
    }
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "previous_call",
        "content": '{"content":"4","is_error":false}',
    }


def test_provider_parses_final_answer_and_usage() -> None:
    provider = DeepSeekProvider(
        make_config(),
        transport=StubTransport(
            response=make_http_response(content="The answer is 42.")
        ),
    )

    response = provider.complete(
        history=[NativeProviderMessage("user", "Answer.")],
        tools=[],
        max_output_tokens=128,
    )

    assert response == NativeProviderResponse(
        tool_calls=(),
        final_text="The answer is 42.",
        input_tokens=20,
        output_tokens=5,
        finish_reason="stop",
    )


def test_invalid_tool_arguments_preserve_actual_usage() -> None:
    provider = DeepSeekProvider(
        make_config(),
        transport=StubTransport(
            response=make_http_response(
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": "not JSON",
                    },
                }],
                finish_reason="tool_calls",
            )
        ),
    )

    with pytest.raises(InvalidModelOutputError) as exc_info:
        provider.complete(
            history=[NativeProviderMessage("user", "Calculate.")],
            tools=[],
            max_output_tokens=128,
        )

    assert exc_info.value.usage == TokenUsage(20, 5)


@pytest.mark.parametrize(
    ("status_code", "expected_retryability"),
    [
        (401, Retryability.NON_RETRYABLE),
        (429, Retryability.RETRYABLE),
        (503, Retryability.RETRYABLE),
    ],
)
def test_http_errors_are_fail_closed(
    status_code: int,
    expected_retryability: Retryability,
) -> None:
    provider = DeepSeekProvider(
        make_config(),
        transport=StubTransport(
            response=DeepSeekHttpResponse(
                status_code=status_code,
                body=b'{"error":{"message":"not exposed"}}',
            )
        ),
    )

    with pytest.raises(NativeProviderError) as exc_info:
        provider.complete(
            history=[NativeProviderMessage("user", "Hello")],
            tools=[],
            max_output_tokens=128,
        )

    error = exc_info.value
    assert (
        error.consumption_state
        is ConsumptionState.UNKNOWN_CONSUMPTION
    )
    assert error.retryability is expected_retryability
    assert "not exposed" not in str(error)


def test_transport_failure_is_unknown_and_retryable() -> None:
    provider = DeepSeekProvider(
        make_config(),
        transport=StubTransport(
            error=DeepSeekTransportError("timed out"),
        ),
    )

    with pytest.raises(NativeProviderError) as exc_info:
        provider.complete(
            history=[NativeProviderMessage("user", "Hello")],
            tools=[],
            max_output_tokens=128,
        )

    error = exc_info.value
    assert (
        error.consumption_state
        is ConsumptionState.UNKNOWN_CONSUMPTION
    )
    assert error.retryability is Retryability.RETRYABLE
    assert "test-api-key" not in str(error)


def test_empty_history_fails_before_transport() -> None:
    transport = StubTransport(
        response=make_http_response(content="unused")
    )
    provider = DeepSeekProvider(
        make_config(),
        transport=transport,
    )

    with pytest.raises(NativeProviderError) as exc_info:
        provider.complete(
            history=[],
            tools=[],
            max_output_tokens=128,
        )

    assert (
        exc_info.value.consumption_state
        is ConsumptionState.NO_CONSUMPTION
    )
    assert transport.calls == []
