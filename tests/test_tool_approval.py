"""Tests for one-shot tool approvals."""

import pytest

from code_agent.kernel_types import ToolCall
from code_agent.tool_approval import (
    ApprovalRequest,
    ApprovalResolution,
    ApprovalResponse,
    DuplicateApprovalError,
    PendingApprovals,
    UnknownApprovalError,
)


def _make_request(
    *,
    approval_id: str = "approval_1",
    path: str = "README.md",
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        call=ToolCall(
            call_id="call_1",
            tool_name="read_file",
            arguments={"path": path},
        ),
        reason="tool requires user confirmation",
    )


def test_resolve_returns_original_request() -> None:
    pending = PendingApprovals()
    request = _make_request()
    pending.add(request)

    resolution = pending.resolve(
        ApprovalResponse(
            approval_id="approval_1",
            approved=True,
        )
    )

    assert resolution == ApprovalResolution(
        request=request,
        approved=True,
    )


def test_approval_can_only_be_consumed_once() -> None:
    pending = PendingApprovals()
    pending.add(_make_request())

    response = ApprovalResponse(
        approval_id="approval_1",
        approved=True,
    )
    pending.resolve(response)

    with pytest.raises(
        UnknownApprovalError,
        match="approval is unknown or already consumed",
    ):
        pending.resolve(response)


def test_rejected_approval_is_also_consumed() -> None:
    pending = PendingApprovals()
    pending.add(_make_request())

    response = ApprovalResponse(
        approval_id="approval_1",
        approved=False,
    )
    resolution = pending.resolve(response)

    assert resolution.approved is False

    with pytest.raises(UnknownApprovalError):
        pending.resolve(response)


def test_duplicate_id_does_not_replace_original_request() -> None:
    pending = PendingApprovals()
    original = _make_request(path="README.md")
    replacement = _make_request(path="secret.txt")

    pending.add(original)

    with pytest.raises(
        DuplicateApprovalError,
        match="duplicate approval: approval_1",
    ):
        pending.add(replacement)

    resolution = pending.resolve(
        ApprovalResponse(
            approval_id="approval_1",
            approved=True,
        )
    )

    assert resolution.request is original
