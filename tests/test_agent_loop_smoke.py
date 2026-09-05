"""Smoke script is verified offline with deterministic providers."""
import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from code_agent.agent_loop import RunResult, RunStatus
from code_agent.event_store import JsonlEventStore
from code_agent.kernel_types import FinalAnswer, Message, MessageRole
from code_agent.run_budget import RunUsage

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def smoke(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("agent_loop_smoke", SCRIPTS / "agent_loop_smoke.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_smoke_writes_success_and_failure_traces(smoke: Any, tmp_path: Path) -> None:
    summary = smoke.run_smoke(tmp_path / "output")
    assert summary["mode"] == "fake"
    assert summary["success"]["status"] == "completed"
    assert summary["success"]["usage"]["model_calls"] == 2
    assert summary["success"]["usage"]["reserved_tokens"] == 0
    assert summary["failure"]["status"] == "model_failed"
    assert summary["failure"]["usage"]["reserved_tokens"] == 1024
    assert summary["success_events"] == 9
    assert summary["failure_events"] == 7
    for name in ("success", "failure"):
        data = summary[name]
        events = JsonlEventStore(Path(data["trace"]), run_id=data["run_id"]).read_all()
        assert events[0].event_type == "run_started"
        assert events[-1].event_type == "run_finished"


@pytest.mark.parametrize(("field", "bad", "expected"), [
    ("output_dir", None, TypeError), ("output_dir", "", TypeError),
    ("live", None, TypeError), ("live", 1, TypeError),
])
def test_smoke_rejects_invalid_inputs_before_files(
    smoke: Any, tmp_path: Path, field: str, bad: object, expected: type[Exception],
) -> None:
    kwargs: dict[str, object] = {"output_dir": tmp_path / "output", "live": False}
    kwargs[field] = bad
    with pytest.raises(expected, match=field):
        smoke.run_smoke(**kwargs)
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_not_overwritten(smoke: Any, tmp_path: Path) -> None:
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        smoke.run_smoke(tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_missing_live_config_is_rejected_before_creating_output(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        smoke.run_smoke(tmp_path / "output", live=True)
    assert list(tmp_path.iterdir()) == []


def test_live_mode_uses_provider_through_loop_without_network(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from code_agent.model_adapter import FakeModelAdapter, ModelResponse, TokenUsage
    from code_agent.kernel_types import ToolCall
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-only")
    monkeypatch.setattr(smoke, "NativeToolCallingModelAdapter", lambda provider: FakeModelAdapter([
        ModelResponse(ToolCall("live-fake", "calculator", {"a": 17, "b": 25}), TokenUsage(1, 1)),
        ModelResponse(FinalAnswer("42"), TokenUsage(1, 1)),
    ]))
    monkeypatch.setattr(smoke, "monotonic", lambda: 0.0)
    summary = smoke.run_smoke(tmp_path / "live-fake", live=True)
    assert summary["mode"] == "live" and summary["success"]["status"] == "completed"


def test_cli_default_is_offline(smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["agent_loop_smoke.py", "--output-dir", str(tmp_path / "cli")])
    smoke.main()
    assert '"mode": "fake"' in capsys.readouterr().out


def test_unexpected_approval_is_rejected(smoke: Any) -> None:
    from code_agent.kernel_types import ToolCall
    from code_agent.tool_approval import ApprovalRequest, ApprovalResponse
    request = ApprovalRequest("a1", ToolCall("c1", "calculator", {"a": 17, "b": 25}), "confirm")
    assert smoke._reject_unexpected_approval(request) == ApprovalResponse("a1", False)


def test_bad_round_trip_preserves_summary_for_diagnosis(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from code_agent.model_adapter import FakeModelAdapter, ModelResponse, TokenUsage
    monkeypatch.setattr(smoke, "FakeModelAdapter", lambda outcomes: FakeModelAdapter([
        ModelResponse(FinalAnswer("42"), TokenUsage(1, 1)),
    ]))
    output = tmp_path / "bad"
    with pytest.raises(RuntimeError, match="calculator round trip"):
        smoke.run_smoke(output)
    assert (output / "summary.json").exists()
    assert (output / "success.jsonl").exists()
    assert not (output / "failure.jsonl").exists()


def test_failure_fixture_must_retain_unknown_consumption(
    smoke: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = smoke.run_agent
    count = 0

    def fake(*args: Any, **kwargs: Any) -> RunResult:
        nonlocal count
        count += 1
        if count == 1:
            return cast(RunResult, original(*args, **kwargs))
        return RunResult(
            "wrong-failure", RunStatus.COMPLETED, FinalAnswer("42"), None,
            (Message(MessageRole.USER, "task"), Message(MessageRole.ASSISTANT, "42")),
            RunUsage(1, 1, 1, 1), 0.0,
        )

    monkeypatch.setattr(smoke, "run_agent", fake)
    with pytest.raises(RuntimeError, match="retain unknown consumption"):
        smoke.run_smoke(tmp_path / "bad-failure")
