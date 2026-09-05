"""Policy observation owns decisions before execution and approval consumption."""
from collections.abc import Callable, Mapping
from typing import cast

import pytest

from code_agent.kernel_types import ToolCall, ToolResult
from code_agent.tool_approval import ApprovalRequest, ApprovalResponse, UnknownApprovalError
from code_agent.tool_controller import PolicyObserver, ToolController
from code_agent.tool_policy import PolicyAction, PolicyDecision
from test_agent_loop_core import Harness, _call


@pytest.mark.parametrize("policy", ["allow", "ask", "deny"])
def test_policy_observer_sees_actual_decision_before_execution(policy: str) -> None:
    harness = Harness([], policy=policy)
    observed: list[tuple[str, PolicyAction, bool]] = []

    def observe(call: ToolCall, decision: PolicyDecision, after_approval: bool) -> None:
        assert harness.executed == []
        observed.append((call.call_id, decision.action, after_approval))

    result = harness.controller.handle(_call(), on_decision=observe)
    assert observed == [("c1", PolicyAction(policy), False)]
    assert len(harness.executed) == int(policy == "allow")
    assert isinstance(result, ApprovalRequest if policy == "ask" else ToolResult)


@pytest.mark.parametrize("owner", ["runtime", "policy", "pending_approvals", "approval_id_factory"])
def test_controller_constructor_rejects_invalid_dependency(owner: str) -> None:
    harness = Harness([])
    kwargs: dict[str, object] = {
        "runtime": harness.controller._runtime, "policy": harness.controller._policy,
        "pending_approvals": harness.pending, "approval_id_factory": lambda: "a1",
    }
    kwargs[owner] = None
    with pytest.raises(TypeError, match=owner):
        cast(Callable[..., ToolController], ToolController)(**kwargs)
    assert harness.executed == []


@pytest.mark.parametrize(("field", "bad", "expected"), [
    ("call", None, TypeError), ("call_id", "", ValueError),
    ("tool_name", None, TypeError), ("arguments", [], TypeError),
])
def test_invalid_tool_call_rejected_before_runtime(
    monkeypatch: pytest.MonkeyPatch, field: str, bad: object, expected: type[Exception],
) -> None:
    harness = Harness([])
    prepared: list[ToolCall] = []

    def prepare(call: ToolCall) -> ToolResult:
        prepared.append(call)
        return ToolResult(call.call_id, "unexpected", True)

    monkeypatch.setattr(harness.controller._runtime, "prepare", prepare)
    call = _call()
    if field != "call":
        object.__setattr__(call, field, bad)
    with pytest.raises(expected, match="call"):
        harness.controller.handle(cast(ToolCall, bad) if field == "call" else call)
    assert prepared == [] and harness.executed == []


@pytest.mark.parametrize("bad", [False, "", 0])
def test_invalid_observer_is_rejected_before_preparation_or_approval_consumption(bad: object) -> None:
    harness = Harness([], policy="ask")
    with pytest.raises(TypeError, match="on_decision"):
        harness.controller.handle(_call(), on_decision=cast(PolicyObserver, bad))
    request = harness.controller.handle(_call())
    assert isinstance(request, ApprovalRequest)
    response = ApprovalResponse(request.approval_id, False)
    with pytest.raises(TypeError, match="on_decision"):
        harness.controller.resolve_approval(response, on_decision=cast(PolicyObserver, bad))
    assert harness.pending.resolve(response).request == request
    assert harness.executed == []


@pytest.mark.parametrize(("field", "value", "expected"), [
    ("decision", None, TypeError), ("action", "allow", TypeError),
    ("reason", None, TypeError), ("reason", " ", ValueError),
])
def test_invalid_policy_output_prevents_observer_and_handler(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, expected: type[Exception],
) -> None:
    harness = Harness([])
    decision = PolicyDecision(PolicyAction.ALLOW, "allowed")
    if field != "decision":
        object.__setattr__(decision, field, value)
    monkeypatch.setattr(harness.controller._policy, "decide", lambda call: value if field == "decision" else decision)
    observed: list[object] = []

    def observe(*values: object) -> None:
        observed.extend(values)

    with pytest.raises(expected, match="policy"):
        harness.controller.handle(_call(), on_decision=observe)
    assert observed == [] and harness.executed == []


@pytest.mark.parametrize("bad", [None, ""])
def test_invalid_approval_id_cannot_create_pending_request(
    monkeypatch: pytest.MonkeyPatch, bad: object,
) -> None:
    harness = Harness([], policy="ask")
    monkeypatch.setattr(harness.controller, "_approval_id_factory", lambda: bad)
    with pytest.raises((TypeError, ValueError), match="approval_id"):
        harness.controller.handle(_call())
    assert harness.pending._pending == {} and harness.executed == []


@pytest.mark.parametrize("stage", ["initial", "approval"])
def test_policy_observer_error_preserves_identity_and_prevents_execution(stage: str) -> None:
    harness = Harness([], policy="ask" if stage == "approval" else "allow")
    failure = OSError("policy trace failed")

    def fail(call: ToolCall, decision: PolicyDecision, after_approval: bool) -> None:
        raise failure

    with pytest.raises(OSError) as caught:
        if stage == "initial":
            harness.controller.handle(_call(), on_decision=fail)
        else:
            request = harness.controller.handle(_call())
            assert isinstance(request, ApprovalRequest)
            harness.controller.resolve_approval(ApprovalResponse(request.approval_id, True), on_decision=fail)
    assert caught.value is failure
    assert "observe tool policy for call c1" in failure.__notes__
    assert harness.executed == []


def test_policy_observer_snapshot_cannot_change_action_or_arguments() -> None:
    harness = Harness([])

    def mutate(call: ToolCall, decision: PolicyDecision, after_approval: bool) -> None:
        cast(dict[str, object], call.arguments)["value"] = -1
        object.__setattr__(decision, "action", PolicyAction.DENY)

    result = harness.controller.handle(_call(), on_decision=mutate)
    assert result == ToolResult("c1", "42", False)
    assert harness.executed == [{"value": 42}]


def test_non_none_policy_observer_result_stops_execution() -> None:
    harness = Harness([])

    def bad(call: ToolCall, decision: PolicyDecision, after_approval: bool) -> str:
        return "unexpected"

    with pytest.raises(TypeError, match="on_decision must return None"):
        harness.controller.handle(_call(), on_decision=cast(PolicyObserver, bad))
    assert harness.executed == []


def test_unknown_tool_has_no_fabricated_policy_event() -> None:
    harness = Harness([])
    observed: list[object] = []

    def observe(*values: object) -> None:
        observed.extend(values)

    result = harness.controller.handle(_call(name="missing"), on_decision=observe)
    assert isinstance(result, ToolResult) and result.is_error
    assert observed == [] and harness.executed == []


@pytest.mark.parametrize(("field", "value", "expected"), [
    ("response", None, TypeError), ("approval_id", None, TypeError),
    ("approval_id", " ", ValueError), ("approved", 1, TypeError),
])
def test_invalid_approval_response_does_not_consume_pending(
    field: str, value: object, expected: type[Exception],
) -> None:
    harness = Harness([], policy="ask")
    request = harness.controller.handle(_call())
    assert isinstance(request, ApprovalRequest)
    response = ApprovalResponse(request.approval_id, True)
    if field != "response":
        object.__setattr__(response, field, value)
    with pytest.raises(expected):
        harness.controller.resolve_approval(cast(ApprovalResponse, value) if field == "response" else response)
    assert harness.pending.resolve(ApprovalResponse(request.approval_id, False)).request == request
    assert harness.executed == []
