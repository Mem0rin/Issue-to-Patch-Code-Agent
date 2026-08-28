# tests/test_json_text_messages.py

from code_agent.json_text_messages import (
    ProviderMessage,
    build_provider_messages,
)
from code_agent.kernel_types import (
    HistoryItem,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


def test_build_provider_messages_preserves_history() -> None:
    history: list[HistoryItem] = [
        Message(
            role=MessageRole.USER,
            content="读取 README.md",
        ),
        ToolCall(
            call_id="call_1",
            tool_name="read_file",
            arguments={"path": "README.md"},
        ),
        ToolResult(
            call_id="call_1",
            content="README contents",
            is_error=False,
        ),
    ]

    result = build_provider_messages(history)

    assert result == (
        ProviderMessage(
            role="user",
            content="读取 README.md",
        ),
        ProviderMessage(
            role="assistant",
            content=(
                '{"arguments":{"path":"README.md"},'
                '"call_id":"call_1",'
                '"tool_name":"read_file",'
                '"type":"tool_call"}'
            ),
        ),
        ProviderMessage(
            role="user",
            content=(
                '{"call_id":"call_1",'
                '"content":"README contents",'
                '"is_error":false,'
                '"type":"tool_result"}'
            ),
        ),
    )