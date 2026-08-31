from collections.abc import Mapping

import pytest

from code_agent.kernel_types import (
    ToolCall,
    ToolResult,
    ToolSpec,
)
from code_agent.tool_approval import (
    ApprovalRequest,
    ApprovalResponse,
    PendingApprovals,
    UnknownApprovalError,
)
from code_agent.tool_controller import ToolController
from code_agent.tool_policy import ToolPolicy
from code_agent.tool_registry import (
    ToolDefinition,
    ToolRegistry,
)
from code_agent.tool_runtime import ToolRuntime


def _accept_any_arguments(
    arguments: Mapping[str, object],
) -> None:
    _ = arguments


def _make_definition(
    name: str,
    executed: list[str],
) -> ToolDefinition:
    def handler(
        arguments: Mapping[str, object],
    ) -> str:
        _ = arguments
        executed.append(name)
        return f"executed: {name}"

    return ToolDefinition(
        spec=ToolSpec(
            name=name,
            description=f"Execute {name}.",
            input_schema={"type": "object"},
        ),
        validate_arguments=_accept_any_arguments,
        handler=handler,
    )


def _make_call(
    tool_name: str,
) -> ToolCall:
    return ToolCall(
        call_id="call_1",
        tool_name=tool_name,
        arguments={},
    )


def test_allow_executes_handler() -> None:
    executed: list[str] = []
    definition = _make_definition(
        "read_file",
        executed,
    )
    controller = ToolController(
        runtime=ToolRuntime(
            ToolRegistry([definition])
        ),
        policy=ToolPolicy(
            allowed_tools={"read_file"},
            confirmation_required_tools=set(),
        ),
        pending_approvals=PendingApprovals(),
        approval_id_factory=lambda: "approval_1",
    )

    result = controller.handle(
        _make_call("read_file")
    )

    assert result == ToolResult(
        call_id="call_1",
        content="executed: read_file",
        is_error=False,
    )
    assert executed == ["read_file"]


def test_deny_does_not_execute_handler() -> None:
    executed: list[str] = []
    definition = _make_definition(
        "delete_file",
        executed,
    )
    controller = ToolController(
        runtime=ToolRuntime(
            ToolRegistry([definition])
        ),
        policy=ToolPolicy(
            allowed_tools=set(),
            confirmation_required_tools=set(),
        ),
        pending_approvals=PendingApprovals(),
        approval_id_factory=lambda: "approval_1",
    )

    result = controller.handle(
        _make_call("delete_file")
    )

    assert result == ToolResult(
        call_id="call_1",
        content="tool is denied by default policy",
        is_error=True,
    )
    assert executed == []


def test_ask_creates_pending_request_without_execution() -> None:
    executed: list[str] = []
    definition = _make_definition(
        "apply_patch",
        executed,
    )
    pending = PendingApprovals()
    controller = ToolController(
        runtime=ToolRuntime(
            ToolRegistry([definition])
        ),
        policy=ToolPolicy(
            allowed_tools=set(),
            confirmation_required_tools={"apply_patch"},
        ),
        pending_approvals=pending,
        approval_id_factory=lambda: "approval_1",
    )

    outcome = controller.handle(
        _make_call("apply_patch")
    )

    assert isinstance(outcome, ApprovalRequest)
    assert outcome.approval_id == "approval_1"
    assert outcome.call.tool_name == "apply_patch"
    assert executed == []

    resolution = pending.resolve(
        ApprovalResponse(
            approval_id="approval_1",
            approved=True,
        )
    )
    assert resolution.request is outcome


def test_approved_request_executes_exactly_once() -> None:
    executed: list[str] = []
    definition = _make_definition(
        "apply_patch",
        executed,
    )
    pending = PendingApprovals()
    controller = ToolController(
        runtime=ToolRuntime(
            ToolRegistry([definition])
        ),
        policy=ToolPolicy(
            allowed_tools=set(),
            confirmation_required_tools={"apply_patch"},
        ),
        pending_approvals=pending,
        approval_id_factory=lambda: "approval_1",
    )

    outcome = controller.handle(
        _make_call("apply_patch")
    )

    assert isinstance(outcome, ApprovalRequest)
    assert executed == []

    response = ApprovalResponse(
        approval_id=outcome.approval_id,
        approved=True,
    )
    result = controller.resolve_approval(response)

    assert result == ToolResult(
        call_id="call_1",
        content="executed: apply_patch",
        is_error=False,
    )
    assert executed == ["apply_patch"]

    # 同一个批准不能再次执行
    with pytest.raises(
        UnknownApprovalError,
        match="approval is unknown or already consumed",
    ):
        controller.resolve_approval(response)

    assert executed == ["apply_patch"]


def test_rejected_request_never_executes_handler() -> None:
    executed: list[str] = []
    definition = _make_definition(
        "apply_patch",
        executed,
    )
    controller = ToolController(
        runtime=ToolRuntime(
            ToolRegistry([definition])
        ),
        policy=ToolPolicy(
            allowed_tools=set(),
            confirmation_required_tools={"apply_patch"},
        ),
        pending_approvals=PendingApprovals(),
        approval_id_factory=lambda: "approval_1",
    )

    outcome = controller.handle(
        _make_call("apply_patch")
    )
    assert isinstance(outcome, ApprovalRequest)

    result = controller.resolve_approval(
        ApprovalResponse(
            approval_id=outcome.approval_id,
            approved=False,
        )
    )

    assert result == ToolResult(
        call_id="call_1",
        content="tool execution rejected by user",
        is_error=True,
    )
    assert executed == []
