"""Exercise the public agent loop; --live explicitly enables DeepSeek requests."""
import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from code_agent.agent_loop import RunRequest, RunResult, RunStatus, run_agent
from code_agent.deepseek_provider import DeepSeekConfig, DeepSeekProvider
from code_agent.event_store import JsonlEventStore
from code_agent.kernel_types import FinalAnswer, Message, MessageRole, ToolCall, ToolResult
from code_agent.model_adapter import (
    FakeModelAdapter, InvalidModelOutputError, ModelAdapter, ModelCallBudget,
    ModelProviderError, ModelResponse, TokenUsage,
)
from code_agent.native_tool_adapter import NativeToolCallingModelAdapter
from code_agent.run_budget import RunBudget
from code_agent.tool_approval import ApprovalRequest, ApprovalResponse
from deepseek_tool_smoke import _make_controller


def _reject_unexpected_approval(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(request.approval_id, False)


def _request(run_id: str) -> RunRequest:
    return RunRequest(
        run_id,
        (
            Message(MessageRole.SYSTEM, "Call calculator exactly once, then use its result to answer."),
            Message(MessageRole.USER, "Use the calculator tool to add 17 and 25. Do not calculate it yourself."),
        ),
        RunBudget(4, 2, 4096, 180.0), ModelCallBudget(128), 1024,
        max_provider_retries=1, max_invalid_output_retries=1,
    )


def _summary(result: RunResult, trace: Path) -> dict[str, object]:
    return {
        "run_id": result.run_id, "status": result.status.value,
        "reason": None if result.reason is None else result.reason.value,
        "final_answer": None if result.final_answer is None else result.final_answer.content,
        "usage": asdict(result.usage), "elapsed_seconds": result.elapsed_seconds,
        "trace": str(trace.resolve()),
    }


def run_smoke(output_dir: Path, *, live: bool = False) -> dict[str, object]:
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if type(live) is not bool:
        raise TypeError("live must be a bool")
    if output_dir.exists():
        raise ValueError("output_dir must not already exist")
    config = DeepSeekConfig.from_environment() if live else None
    adapter: ModelAdapter = (
        NativeToolCallingModelAdapter(DeepSeekProvider(config)) if config is not None
        else FakeModelAdapter([
            ModelResponse(ToolCall("calculator-1", "calculator", {"a": 17, "b": 25}), TokenUsage(10, 5)),
            ModelResponse(FinalAnswer("17 + 25 = 42"), TokenUsage(20, 5)),
        ])
    )
    output_dir.mkdir(parents=True)
    fixed_time = datetime(2026, 9, 3, tzinfo=UTC)
    success_path = output_dir / "success.jsonl"
    success_id = ("deepseek-" if live else "fake-success-") + uuid4().hex
    store = JsonlEventStore(
        success_path, run_id=success_id,
        clock=(lambda: datetime.now(UTC)) if live else (lambda: fixed_time),
    )
    controller, tools = _make_controller()
    result = run_agent(
        _request(success_id), adapter=adapter, controller=controller, tools=tools,
        event_store=store, approval_handler=_reject_unexpected_approval,
        clock=monotonic if live else lambda: 0.0,
    )
    summary: dict[str, object] = {
        "mode": "live" if live else "fake",
        "model": config.model if config is not None else "fake",
        "success": _summary(result, success_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tool_results = [item for item in result.history if isinstance(item, ToolResult)]
    if (
        result.status is not RunStatus.COMPLETED
        or len(tool_results) != 1
        or tool_results[0].content != "42"
        or tool_results[0].is_error
        or result.final_answer is None
        or "42" not in result.final_answer.content
        or result.usage.action_steps != 2
        or result.usage.reserved_tokens != 0
    ):
        raise RuntimeError("loop smoke did not complete one calculator round trip; inspect summary.json")

    failure_path = output_dir / "failure.jsonl"
    failure_id = "fake-failure-" + uuid4().hex
    failure_store = JsonlEventStore(failure_path, run_id=failure_id, clock=lambda: fixed_time)
    failed_adapter = FakeModelAdapter([
        InvalidModelOutputError("repair the output", usage=TokenUsage(3, 1)),
        ModelProviderError("simulated unknown consumption"),
    ])
    failure_controller, failure_tools = _make_controller()
    failed = run_agent(
        _request(failure_id), adapter=failed_adapter, controller=failure_controller,
        tools=failure_tools, event_store=failure_store,
        approval_handler=_reject_unexpected_approval, clock=lambda: 0.0,
    )
    if failed.status is not RunStatus.MODEL_FAILED or failed.usage.reserved_tokens != 1024:
        raise RuntimeError("fake failure smoke did not retain unknown consumption")
    summary["failure"] = _summary(failed, failure_path)
    summary["success_events"] = len(store.read_all())
    summary["failure_events"] = len(failure_store.read_all())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Enable real DeepSeek requests.")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run_smoke(arguments.output_dir, live=arguments.live), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
