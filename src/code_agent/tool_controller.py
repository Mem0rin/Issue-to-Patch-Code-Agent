"""Apply tool preparation, policy and one-shot approval decisions."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import TypeAlias, cast

from ._event_payload import validate_payload
from .kernel_types import ToolCall, ToolResult
from .tool_approval import ApprovalRequest, ApprovalResponse, PendingApprovals
from .tool_policy import PolicyAction, PolicyDecision, ToolPolicy
from .tool_runtime import PreparedToolCall, ToolRuntime

ApprovalIdFactory: TypeAlias = Callable[[], str]
ToolOutcome: TypeAlias = ToolResult | ApprovalRequest
PolicyObserver: TypeAlias = Callable[[ToolCall, PolicyDecision, bool], None]


def _nonblank(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _check_observer(observer: PolicyObserver | None) -> None:
    if observer is not None and not callable(observer):
        raise TypeError("on_decision must be callable or None")


def _check_call(call: ToolCall) -> None:
    if not isinstance(call, ToolCall):
        raise TypeError("call must be a ToolCall")
    _nonblank(call.call_id, "call.call_id")
    _nonblank(call.tool_name, "call.tool_name")
    if not isinstance(call.arguments, Mapping):
        raise TypeError("call.arguments must be a Mapping")
    validate_payload(call.arguments)


class ToolController:
    def __init__(
        self, *, runtime: ToolRuntime, policy: ToolPolicy,
        pending_approvals: PendingApprovals, approval_id_factory: ApprovalIdFactory,
    ) -> None:
        if not isinstance(runtime, ToolRuntime):
            raise TypeError("runtime must be a ToolRuntime")
        if not isinstance(policy, ToolPolicy):
            raise TypeError("policy must be a ToolPolicy")
        if not isinstance(pending_approvals, PendingApprovals):
            raise TypeError("pending_approvals must be PendingApprovals")
        if not callable(approval_id_factory):
            raise TypeError("approval_id_factory must be callable")
        self._runtime = runtime
        self._policy = policy
        self._pending_approvals = pending_approvals
        self._approval_id_factory = approval_id_factory

    def _decide(
        self, prepared: PreparedToolCall, observer: PolicyObserver | None,
        after_approval: bool,
    ) -> PolicyDecision:
        decision = self._policy.decide(prepared.call)
        if not isinstance(decision, PolicyDecision):
            raise TypeError("policy must return a PolicyDecision")
        if not isinstance(decision.action, PolicyAction):
            raise TypeError("policy decision.action must be a PolicyAction")
        _nonblank(decision.reason, "policy decision.reason")
        if observer is not None:
            try:
                result = cast(Callable[..., object], observer)(
                    deepcopy(prepared.call), replace(decision), after_approval,
                )
                if result is not None:
                    raise TypeError("on_decision must return None")
            except Exception as error:
                error.add_note(f"observe tool policy for call {prepared.call.call_id}")
                raise
        return decision

    def handle(
        self, call: ToolCall, *, on_decision: PolicyObserver | None = None,
    ) -> ToolOutcome:
        _check_observer(on_decision)
        _check_call(call)
        preparation = self._runtime.prepare(call)
        if isinstance(preparation, ToolResult):
            return preparation

        decision = self._decide(preparation, on_decision, False)
        if decision.action is PolicyAction.ALLOW:
            return self._runtime.execute(preparation)
        if decision.action is PolicyAction.DENY:
            return ToolResult(call.call_id, decision.reason, True)

        approval_id = self._approval_id_factory()
        _nonblank(approval_id, "approval_id")
        request = ApprovalRequest(approval_id, preparation.call, decision.reason)
        self._pending_approvals.add(request)
        return request

    def resolve_approval(
        self, response: ApprovalResponse, *, on_decision: PolicyObserver | None = None,
    ) -> ToolResult:
        _check_observer(on_decision)
        if not isinstance(response, ApprovalResponse):
            raise TypeError("response must be an ApprovalResponse")
        _nonblank(response.approval_id, "response.approval_id")
        if type(response.approved) is not bool:
            raise TypeError("response.approved must be a bool")

        resolution = self._pending_approvals.resolve(response)
        call = resolution.request.call
        if not resolution.approved:
            return ToolResult(call.call_id, "tool execution rejected by user", True)

        preparation = self._runtime.prepare(call)
        if isinstance(preparation, ToolResult):
            return preparation
        current = self._decide(preparation, on_decision, True)
        if current.action is PolicyAction.DENY:
            return ToolResult(call.call_id, current.reason, True)
        return self._runtime.execute(preparation)
