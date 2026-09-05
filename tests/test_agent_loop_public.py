"""Public run contracts, complete traces and persistence failure boundaries."""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from code_agent.agent_loop import RunResult, RunStatus, run_agent
from code_agent.event_store import EventDraft, JsonlEventStore, StoredEvent
from code_agent.kernel_types import FinalAnswer, HistoryItem, ModelFeedback, ToolCall, ToolResult, ToolSpec
from code_agent.model_adapter import (
    ConsumptionState, FakeOutcome, InvalidModelOutputError, ModelCallBudget,
    ModelProviderError, ModelResponse, Retryability, TokenUsage,
)
from code_agent.run_budget import BudgetStopReason, RunBudget
from code_agent.tool_approval import ApprovalRequest, ApprovalResponse
from code_agent.tool_policy import ToolPolicy
from test_agent_loop_core import Harness, _call, _response


class RecordingStore(JsonlEventStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path, run_id="core-run", clock=lambda: datetime(2026, 9, 3, tzinfo=UTC))
        self.reads = 0
        self.attempts: list[str] = []
        self.fail_at: str | None = None
        self.failure = OSError("trace unavailable")
        self.after_write: Callable[[EventDraft], None] = lambda draft: None

    def read_all(self) -> tuple[StoredEvent, ...]:
        self.reads += 1
        return super().read_all()

    def append(self, draft: EventDraft) -> StoredEvent:
        self.attempts.append(draft.event_type)
        if draft.event_type == self.fail_at:
            raise self.failure
        event = super().append(draft)
        self.after_write(draft)
        return event


class PublicHarness(Harness):
    def __init__(self, path: Path, outcomes: Sequence[FakeOutcome], *, policy: str = "allow") -> None:
        super().__init__(outcomes, policy=policy)
        self.path = path
        self.store = RecordingStore(path)
        self.clock_calls = 0
        self.approvals: list[ApprovalRequest] = []

    def clock(self) -> float:
        self.clock_calls += 1
        return self.now

    def approve(self, request: ApprovalRequest) -> ApprovalResponse:
        self.approvals.append(request)
        return ApprovalResponse(request.approval_id, True)

    def run_public(self, **changes: object) -> RunResult:
        args: dict[str, object] = dict(
            request=self.request, adapter=self.adapter, controller=self.controller,
            tools=(self.spec,), event_store=self.store, approval_handler=self.approve,
            clock=self.clock,
        )
        args.update(changes)
        return cast(Callable[..., RunResult], run_agent)(**args)

    def events(self) -> tuple[StoredEvent, ...]:
        return self.store.read_all()


@pytest.fixture
def harness(tmp_path: Path) -> PublicHarness:
    return PublicHarness(tmp_path / "trace.jsonl", [_response(_call()), _response(FinalAnswer("done"))])


def test_success_trace_pairs_actions_and_reconciles_usage(harness: PublicHarness) -> None:
    result = harness.run_public()
    events = harness.events()
    assert [event.event_type for event in events] == [
        "run_started", "model_call_prepared", "model_call_completed",
        "tool_call_requested", "tool_policy_decided", "tool_result_recorded",
        "model_call_prepared", "model_call_completed", "run_finished",
    ]
    assert [event.sequence for event in events] == list(range(1, 10))
    completed = [event.payload for event in events if event.event_type == "model_call_completed"]
    assert [event["attempt_index"] for event in completed] == [1, 2]
    assert sum(cast(dict[str, int], event["actual_usage"])["input_tokens"] for event in completed) == result.usage.input_tokens
    assert result.usage.model_calls == result.usage.action_steps == 2
    assert result.usage.reserved_tokens == 0
    assert result.history[-2] == ToolResult("c1", "42", False)
    assert events[-1].payload["status"] == "completed"
    assert events[-1].payload["final_answer"] == "done"
    assert events[3].payload["call_id"] == events[5].payload["call_id"] == "c1"
    assert events[4].payload["action"] == "allow"


def test_empty_tool_set_and_single_message_can_finish(tmp_path: Path) -> None:
    harness = PublicHarness(tmp_path / "empty.jsonl", [_response(FinalAnswer("done"))])
    result = harness.run_public(tools=())
    assert result.status is RunStatus.COMPLETED
    assert harness.adapter.calls[0].tools == ()
    assert harness.events()[0].payload["tools"] == []
    assert harness.executed == []


@pytest.mark.parametrize(("field", "bad"), [
    ("request", None), ("adapter", None), ("controller", None),
    ("tools", None), ("tools", []), ("tools", (None,)),
    ("event_store", None), ("approval_handler", None), ("clock", None),
])
def test_invalid_input_is_rejected_before_all_dependencies(
    harness: PublicHarness, field: str, bad: object,
) -> None:
    with pytest.raises(TypeError) as caught:
        harness.run_public(**{field: bad})
    assert "prepare run_agent" in caught.value.__notes__
    assert harness.store.reads == harness.clock_calls == 0
    assert harness.store.attempts == [] and harness.adapter.calls == []
    assert harness.executed == [] and harness.approvals == []
    assert not harness.path.exists()


@pytest.mark.parametrize(("field", "value", "expected"), [
    ("name", None, TypeError), ("name", "", ValueError), ("name", " ", ValueError),
    ("description", None, TypeError), ("description", "", ValueError),
    ("input_schema", None, TypeError), ("input_schema", [], TypeError),
    ("input_schema", {"bad": float("inf")}, ValueError),
    ("input_schema", {"bad": object()}, TypeError),
    ("input_schema", {1: "bad key"}, TypeError),
])
def test_invalid_tool_metadata_cannot_reach_store_or_model(
    harness: PublicHarness, field: str, value: object, expected: type[Exception],
) -> None:
    object.__setattr__(harness.spec, field, value)
    with pytest.raises(expected):
        harness.run_public()
    assert harness.store.reads == harness.clock_calls == 0
    assert harness.store.attempts == [] and harness.adapter.calls == []
    assert harness.executed == []


def test_duplicate_tool_names_are_rejected_before_store(harness: PublicHarness) -> None:
    with pytest.raises(ValueError, match="name must be unique"):
        harness.run_public(tools=(harness.spec, harness.spec))
    assert harness.store.reads == 0 and harness.adapter.calls == []


def test_cycle_in_tool_schema_is_rejected_before_store(harness: PublicHarness) -> None:
    schema: dict[str, object] = {}
    schema["self"] = schema
    object.__setattr__(harness.spec, "input_schema", schema)
    with pytest.raises(ValueError):
        harness.run_public()
    assert harness.store.reads == 0 and harness.adapter.calls == []


def test_generic_mapping_schema_is_normalized(harness: PublicHarness) -> None:
    schema = MappingProxyType({"properties": MappingProxyType({"n": {"type": "integer"}})})
    spec = ToolSpec("calc", "Calculate.", schema)
    result = harness.run_public(tools=(spec,))
    assert result.status is RunStatus.COMPLETED
    assert harness.adapter.calls[0].tools[0].input_schema == dict(schema)


@pytest.mark.parametrize("field", ["reserved_tokens", "max_provider_retries", "max_invalid_output_retries"])
def test_mutated_request_is_revalidated_before_dependencies(harness: PublicHarness, field: str) -> None:
    object.__setattr__(harness.request, field, True)
    with pytest.raises(TypeError, match=field):
        harness.run_public()
    assert harness.store.reads == harness.clock_calls == 0
    assert harness.adapter.calls == []


@pytest.mark.parametrize(("owner", "method"), [
    ("adapter", "complete"), ("controller", "handle"), ("controller", "resolve_approval"),
    ("store", "read_all"), ("store", "append"),
])
def test_noncallable_dependency_method_is_rejected_without_side_effects(
    harness: PublicHarness, monkeypatch: pytest.MonkeyPatch, owner: str, method: str,
) -> None:
    monkeypatch.setattr(getattr(harness, owner), method, None)
    with pytest.raises(TypeError, match=method):
        harness.run_public()
    assert harness.clock_calls == 0
    assert harness.adapter.calls == []


def test_store_identity_is_read_only_and_mismatch_is_rejected(harness: PublicHarness) -> None:
    assert harness.store.run_id == "core-run"
    with pytest.raises(AttributeError):
        setattr(harness.store, "run_id", "other")
    other = JsonlEventStore(harness.path, run_id="other")
    with pytest.raises(ValueError, match="run_id must match"):
        harness.run_public(event_store=other)
    assert harness.clock_calls == 0 and not harness.path.exists()


def test_existing_trace_cannot_be_resumed_or_modified(harness: PublicHarness) -> None:
    harness.store.append(EventDraft("previous", {}))
    before = harness.path.read_bytes()
    with pytest.raises(ValueError, match="empty trace"):
        harness.run_public()
    assert harness.path.read_bytes() == before
    assert harness.adapter.calls == [] and harness.clock_calls == 0


def test_corrupt_trace_read_preserves_file_and_prevents_clock(harness: PublicHarness) -> None:
    harness.path.write_bytes(b"bad json\n")
    with pytest.raises(ValueError, match="line 1"):
        harness.run_public()
    assert harness.path.read_bytes() == b"bad json\n"
    assert harness.adapter.calls == [] and harness.clock_calls == 0


def test_store_read_exception_preserves_identity_and_prepare_context(
    harness: PublicHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("read failed")

    def fail() -> tuple[StoredEvent, ...]:
        raise failure

    monkeypatch.setattr(harness.store, "read_all", fail)
    with pytest.raises(OSError) as caught:
        harness.run_public()
    assert caught.value is failure
    assert "prepare run_agent" in failure.__notes__
    assert harness.clock_calls == 0 and harness.adapter.calls == []


def test_non_tuple_store_result_is_rejected_before_clock(
    harness: PublicHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness.store, "read_all", lambda: [])
    with pytest.raises(TypeError, match="read_all must return a tuple"):
        harness.run_public()
    assert harness.clock_calls == 0 and harness.adapter.calls == []


@pytest.mark.parametrize(("value", "expected"), [
    (True, TypeError), (None, TypeError), ("0", TypeError),
    (float("nan"), ValueError), (float("inf"), ValueError), (10**400, ValueError),
])
def test_invalid_initial_clock_cannot_write_started(
    harness: PublicHarness, value: object, expected: type[Exception],
) -> None:
    with pytest.raises(expected, match="clock"):
        harness.run_public(clock=lambda: value)
    assert harness.store.attempts == [] and harness.adapter.calls == []
    assert not harness.path.exists()


def test_negative_clock_origin_is_valid(harness: PublicHarness) -> None:
    result = harness.run_public(clock=lambda: -10.0)
    assert result.elapsed_seconds == 0 and result.status is RunStatus.COMPLETED


def test_backward_clock_aborts_without_retrying_clock_for_abort(
    harness: PublicHarness,
) -> None:
    readings = iter([0.0, -1.0])
    with pytest.raises(ValueError, match="nondecreasing"):
        harness.run_public(clock=lambda: next(readings))
    events = harness.events()
    assert [event.event_type for event in events] == ["run_started", "run_aborted"]
    assert events[-1].payload["elapsed_seconds"] == 0
    assert harness.adapter.calls == []


@pytest.mark.parametrize("stage", ["model_call_prepared", "tool_policy_decided", "approval_resolved"])
def test_time_spent_recording_intent_prevents_next_external_action(
    tmp_path: Path, stage: str,
) -> None:
    harness = PublicHarness(
        tmp_path / "deadline.jsonl", [_response(_call())],
        policy="ask" if stage == "approval_resolved" else "allow",
    )

    def advance(draft: EventDraft) -> None:
        if draft.event_type == stage:
            harness.now = 5.0

    harness.store.after_write = advance
    result = harness.run_public()
    assert result.reason is BudgetStopReason.TIME_EXHAUSTED
    assert harness.executed == []
    assert len(harness.adapter.calls) == (0 if stage == "model_call_prepared" else 1)
    assert result.usage.reserved_tokens == 0
    assert harness.events()[-1].event_type == "run_finished"


@pytest.mark.parametrize("approved", [False, True])
def test_approval_trace_records_actual_decision_and_recheck(tmp_path: Path, approved: bool) -> None:
    harness = PublicHarness(
        tmp_path / "approval.jsonl", [_response(_call()), _response(FinalAnswer("done"))],
        policy="ask",
    )
    result = harness.run_public(
        approval_handler=lambda request: ApprovalResponse(request.approval_id, approved),
    )
    events = harness.events()
    policies = [event.payload for event in events if event.event_type == "tool_policy_decided"]
    assert [policy["phase"] for policy in policies] == (["initial", "approval"] if approved else ["initial"])
    resolved = next(event for event in events if event.event_type == "approval_resolved")
    assert resolved.payload["approved"] is approved
    assert len(harness.executed) == int(approved)
    assert result.status is RunStatus.COMPLETED


def test_approval_cannot_override_new_hard_deny(tmp_path: Path) -> None:
    harness = PublicHarness(
        tmp_path / "deny.jsonl", [_response(_call()), _response(FinalAnswer("denied"))],
        policy="ask",
    )

    def approve(request: ApprovalRequest) -> ApprovalResponse:
        harness.controller._policy = ToolPolicy(allowed_tools=set(), confirmation_required_tools=set())
        return ApprovalResponse(request.approval_id, True)

    result = harness.run_public(approval_handler=approve)
    policies = [event.payload for event in harness.events() if event.event_type == "tool_policy_decided"]
    assert [policy["action"] for policy in policies] == ["ask", "deny"]
    assert result.history[-2] == ToolResult("c1", "tool is denied by default policy", True)
    assert harness.executed == []


@pytest.mark.parametrize("approved", [None, 1])
def test_invalid_approval_aborts_before_resolution(tmp_path: Path, approved: object) -> None:
    harness = PublicHarness(tmp_path / "bad-approval.jsonl", [_response(_call())], policy="ask")
    bad = ApprovalResponse("approval-1", cast(bool, approved))
    with pytest.raises(TypeError, match="approved"):
        harness.run_public(approval_handler=lambda request: bad)
    assert "approval_resolved" not in harness.store.attempts
    assert harness.store.attempts[-1] == "run_aborted"
    assert harness.executed == []


@pytest.mark.parametrize(("stage", "calls", "tools"), [
    ("run_started", 0, 0), ("model_call_prepared", 0, 0),
    ("model_call_completed", 1, 0), ("tool_call_requested", 1, 0),
    ("tool_policy_decided", 1, 0), ("approval_requested", 1, 0),
    ("approval_resolved", 1, 0), ("tool_result_recorded", 1, 1),
    ("run_finished", 2, 1),
])
def test_trace_failure_stops_without_error_write_retry_or_next_action(
    tmp_path: Path, stage: str, calls: int, tools: int,
) -> None:
    harness = PublicHarness(
        tmp_path / "failure.jsonl", [_response(_call()), _response(FinalAnswer("done"))],
        policy="ask",
    )
    harness.store.fail_at = stage
    with pytest.raises(OSError) as caught:
        harness.run_public()
    assert caught.value is harness.store.failure
    assert harness.store.attempts[-1] == stage
    assert "run_aborted" not in harness.store.attempts
    assert len(harness.adapter.calls) == calls and len(harness.executed) == tools


@pytest.mark.parametrize("stage", ["model_call_failed", "model_retry_scheduled"])
def test_failed_attempt_trace_error_does_not_issue_another_model_call(tmp_path: Path, stage: str) -> None:
    harness = PublicHarness(
        tmp_path / "retry-write.jsonl", [InvalidModelOutputError("bad"), _response(FinalAnswer("unused"))],
    )
    harness.request = replace(harness.request, max_invalid_output_retries=1)
    harness.store.fail_at = stage
    with pytest.raises(OSError) as caught:
        harness.run_public()
    assert caught.value is harness.store.failure
    assert len(harness.adapter.calls) == 1
    assert harness.store.attempts[-1] == stage and harness.executed == []


def test_abort_write_failure_keeps_original_tool_failure_as_cause(
    harness: PublicHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = RuntimeError("tool failed")

    def fail(*args: object, **kwargs: object) -> ToolResult:
        raise original

    monkeypatch.setattr(harness.controller, "handle", fail)
    harness.store.fail_at = "run_aborted"
    with pytest.raises(OSError) as caught:
        harness.run_public()
    assert caught.value is harness.store.failure
    assert caught.value.__cause__ is original
    assert harness.store.attempts.count("run_aborted") == 1
    assert len(harness.adapter.calls) == 1


def test_unexpected_adapter_failure_has_failed_attempt_and_abort(
    harness: PublicHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("private provider detail")

    def fail(history: Sequence[HistoryItem], tools: Sequence[ToolSpec], budget: ModelCallBudget) -> ModelResponse:
        raise failure

    monkeypatch.setattr(harness.adapter, "complete", fail)
    with pytest.raises(RuntimeError) as caught:
        harness.run_public()
    assert caught.value is failure
    events = harness.events()
    assert [event.event_type for event in events] == [
        "run_started", "model_call_prepared", "model_call_failed", "run_aborted",
    ]
    assert events[2].payload["consumption_state"] == "unknown_consumption"
    assert cast(dict[str, int], events[-1].payload["usage"])["reserved_tokens"] == 10
    assert "private provider detail" not in harness.path.read_text(encoding="utf-8")


def test_mixed_retries_have_complete_attempts_and_feedback_history(tmp_path: Path) -> None:
    harness = PublicHarness(tmp_path / "mixed.jsonl", [
        ModelProviderError("transient", consumption_state=ConsumptionState.NO_CONSUMPTION, retryability=Retryability.RETRYABLE),
        InvalidModelOutputError("repair", usage=TokenUsage(2, 1)),
        _response(_call()), _response(FinalAnswer("done")),
    ])
    harness.request = replace(harness.request, max_provider_retries=1, max_invalid_output_retries=1)
    result = harness.run_public()
    events = harness.events()
    prepared = [event.payload["attempt_index"] for event in events if event.event_type == "model_call_prepared"]
    outcomes = [event.payload["attempt_index"] for event in events if event.event_type in {"model_call_completed", "model_call_failed"}]
    assert prepared == outcomes == [1, 2, 3, 4]
    retries = [event.payload for event in events if event.event_type == "model_retry_scheduled"]
    assert [event["kind"] for event in retries] == ["provider", "invalid_output"]
    assert retries[0]["feedback"] is None
    assert result.history[1] == ModelFeedback("repair")
    assert result.usage.input_tokens == 4 and result.usage.output_tokens == 3
    assert result.usage.action_steps == 2 and result.usage.reserved_tokens == 0


@pytest.mark.parametrize("state", list(ConsumptionState))
def test_provider_failure_trace_preserves_consumption_state(tmp_path: Path, state: ConsumptionState) -> None:
    error = ModelProviderError(
        "private detail", consumption_state=state,
        usage=TokenUsage(2, 1) if state is ConsumptionState.ACTUAL_USAGE else None,
    )
    harness = PublicHarness(tmp_path / "provider.jsonl", [error])
    result = harness.run_public()
    failure = next(event for event in harness.events() if event.event_type == "model_call_failed")
    assert result.status is RunStatus.MODEL_FAILED
    assert failure.payload["consumption_state"] == state.value
    assert (failure.payload["actual_usage"] is not None) is (state is ConsumptionState.ACTUAL_USAGE)
    assert result.usage.reserved_tokens == (10 if state is ConsumptionState.UNKNOWN_CONSUMPTION else 0)
    assert harness.events()[-1].event_type == "run_finished"


def test_trace_redaction_does_not_rewrite_model_context(harness: PublicHarness) -> None:
    call = ToolCall("c1", "calc", {"api_key": "secret-value", "value": 42})
    harness.adapter._outcomes = (_response(call), _response(FinalAnswer("done")))
    result = harness.run_public()
    assert cast(ToolCall, result.history[1]).arguments["api_key"] == "secret-value"
    assert "secret-value" not in harness.path.read_text(encoding="utf-8")
    assert harness.executed[0]["api_key"] == "secret-value"


def test_input_snapshots_survive_caller_mutation(
    harness: PublicHarness, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = harness.adapter.complete

    def mutate(history: Sequence[HistoryItem], tools: Sequence[ToolSpec], budget: ModelCallBudget) -> ModelResponse:
        object.__setattr__(harness.request.budget, "max_model_calls", 0)
        cast(dict[str, object], harness.spec.input_schema)["changed"] = True
        return original(history, tools, budget)

    monkeypatch.setattr(harness.adapter, "complete", mutate)
    result = harness.run_public()
    assert result.status is RunStatus.COMPLETED and result.usage.model_calls == 2
    assert all("changed" not in call.tools[0].input_schema for call in harness.adapter.calls)


def test_pre_reservation_budget_stop_has_only_start_and_finish(harness: PublicHarness) -> None:
    harness.request = replace(harness.request, reserved_tokens=1001)
    result = harness.run_public()
    assert result.reason is BudgetStopReason.TOKENS_EXHAUSTED
    assert [event.event_type for event in harness.events()] == ["run_started", "run_finished"]
    assert result.usage.model_calls == 0 and harness.adapter.calls == []


def test_duplicate_call_id_finishes_without_duplicate_tool_intent(harness: PublicHarness) -> None:
    harness.adapter._outcomes = (_response(_call()), _response(_call()))
    result = harness.run_public()
    assert str(result.reason) == "duplicate_call_id"
    assert harness.store.attempts.count("tool_call_requested") == 1
    assert len(harness.executed) == 1
    assert harness.events()[-1].payload["status"] == "model_failed"


def test_elapsed_overflow_aborts_without_reusing_invalid_clock(harness: PublicHarness) -> None:
    readings = iter([-1e308, 1e308])
    with pytest.raises(ValueError, match="elapsed time must be finite"):
        harness.run_public(clock=lambda: next(readings))
    assert harness.adapter.calls == []
    assert harness.events()[-1].event_type == "run_aborted"
    assert harness.events()[-1].payload["elapsed_seconds"] == 0.0


def test_approval_policy_recording_deadline_prevents_execution(tmp_path: Path) -> None:
    harness = PublicHarness(tmp_path / "approval-deadline.jsonl", [_response(_call())], policy="ask")

    def advance(draft: EventDraft) -> None:
        if draft.event_type == "tool_policy_decided" and draft.payload["phase"] == "approval":
            harness.now = 5.0

    harness.store.after_write = advance
    result = harness.run_public()
    assert result.reason is BudgetStopReason.TIME_EXHAUSTED
    assert harness.executed == []
    assert harness.store.attempts.count("tool_policy_decided") == 2


def test_tool_removed_during_approval_is_rechecked_before_execution(tmp_path: Path) -> None:
    from code_agent.tool_registry import ToolRegistry
    from code_agent.tool_runtime import ToolRuntime
    harness = PublicHarness(
        tmp_path / "removed.jsonl", [_response(_call()), _response(FinalAnswer("unavailable"))],
        policy="ask",
    )

    def approve(request: ApprovalRequest) -> ApprovalResponse:
        harness.controller._runtime = ToolRuntime(ToolRegistry([]))
        return ApprovalResponse(request.approval_id, True)

    result = harness.run_public(approval_handler=approve)
    assert cast(ToolResult, result.history[-2]).is_error
    assert harness.store.attempts.count("tool_policy_decided") == 1
    assert harness.executed == []
