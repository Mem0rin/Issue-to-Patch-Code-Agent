"""Run one safe DeepSeek tool-call round trip."""

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from time import monotonic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_agent.deepseek_provider import (  # noqa: E402
    DeepSeekConfig,
    DeepSeekProvider,
)
from code_agent.kernel_types import (  # noqa: E402
    FinalAnswer,
    HistoryItem,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from code_agent.model_adapter import ModelCallBudget  # noqa: E402
from code_agent.model_call import complete_model_with_retry  # noqa: E402
from code_agent.native_tool_adapter import (  # noqa: E402
    NativeToolCallingModelAdapter,
)
from code_agent.run_budget import (  # noqa: E402
    BudgetStopReason,
    BudgetTracker,
    RunBudget,
)
from code_agent.tool_approval import PendingApprovals  # noqa: E402
from code_agent.tool_controller import ToolController  # noqa: E402
from code_agent.tool_policy import ToolPolicy  # noqa: E402
from code_agent.tool_registry import (  # noqa: E402
    InvalidToolArgumentsError,
    ToolDefinition,
    ToolRegistry,
)
from code_agent.tool_runtime import ToolRuntime  # noqa: E402


def _validate_calculator(arguments: Mapping[str, object]) -> None:
    if set(arguments) != {"a", "b"}:
        raise InvalidToolArgumentsError(
            "calculator requires exactly a and b"
        )
    for name in ("a", "b"):
        value = arguments[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise InvalidToolArgumentsError(
                f"calculator {name} must be an integer"
            )


def _calculate(arguments: Mapping[str, object]) -> str:
    _validate_calculator(arguments)
    left = arguments["a"]
    right = arguments["b"]
    if not isinstance(left, int) or isinstance(left, bool):
        raise AssertionError("validated a must be an integer")
    if not isinstance(right, int) or isinstance(right, bool):
        raise AssertionError("validated b must be an integer")
    return str(left + right)


def _make_controller() -> tuple[ToolController, tuple[ToolSpec, ...]]:
    definition = ToolDefinition(
        spec=ToolSpec(
            name="calculator",
            description="Add exactly two integers named a and b.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        ),
        validate_arguments=_validate_calculator,
        handler=_calculate,
    )
    registry = ToolRegistry([definition])
    controller = ToolController(
        runtime=ToolRuntime(registry),
        policy=ToolPolicy(
            allowed_tools={"calculator"},
            confirmation_required_tools=set(),
        ),
        pending_approvals=PendingApprovals(),
        approval_id_factory=lambda: "unused_approval",
    )
    return controller, registry.specs


def _complete(
    adapter: NativeToolCallingModelAdapter,
    tracker: BudgetTracker,
    *,
    history: list[HistoryItem],
    tools: tuple[ToolSpec, ...],
) -> ToolCall | FinalAnswer:
    result = complete_model_with_retry(
        adapter,
        tracker,
        history=history,
        tools=tools,
        budget=ModelCallBudget(max_output_tokens=128),
        reserved_tokens=1024,
        max_provider_retries=1,
        max_invalid_output_retries=1,
    )
    if isinstance(result, BudgetStopReason):
        raise RuntimeError(f"budget stopped the smoke test: {result}")
    return result.action


def main() -> None:
    config = DeepSeekConfig.from_environment()
    provider = DeepSeekProvider(config)
    adapter = NativeToolCallingModelAdapter(provider)
    controller, tools = _make_controller()
    tracker = BudgetTracker(
        RunBudget(
            max_model_calls=4,
            max_action_steps=2,
            max_total_tokens=4096,
            max_elapsed_seconds=180,
        )
    )
    history: list[HistoryItem] = [
        Message(
            role=MessageRole.SYSTEM,
            content=(
                "You are validating a tool-calling Agent Kernel. "
                "Call the calculator tool exactly once, then use its "
                "result to answer the user."
            ),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                "Use the calculator tool to add 17 and 25. Do not "
                "calculate it yourself."
            ),
        ),
    ]

    started_at = monotonic()
    first_action = _complete(
        adapter,
        tracker,
        history=history,
        tools=tools,
    )
    if not isinstance(first_action, ToolCall):
        raise RuntimeError(
            "DeepSeek returned a final answer before calling the tool"
        )

    outcome = controller.handle(first_action)
    if not isinstance(outcome, ToolResult):
        raise RuntimeError("calculator unexpectedly required approval")
    history.extend([first_action, outcome])

    second_action = _complete(
        adapter,
        tracker,
        history=history,
        tools=tools,
    )
    if not isinstance(second_action, FinalAnswer):
        raise RuntimeError(
            "DeepSeek called a tool again instead of finishing"
        )

    elapsed_seconds = monotonic() - started_at
    usage = tracker.usage
    print(json.dumps(
        {
            "status": "ok",
            "model": config.model,
            "call_id": first_action.call_id,
            "tool_name": first_action.tool_name,
            "tool_result": outcome.content,
            "final_answer": second_action.content,
            "model_calls": usage.model_calls,
            "action_steps": usage.action_steps,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "reserved_tokens": usage.reserved_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
        },
        ensure_ascii=False,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
