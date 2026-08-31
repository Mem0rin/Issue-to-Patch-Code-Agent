# tests/test_tool_policy.py

import pytest

from code_agent.kernel_types import ToolCall
from code_agent.tool_policy import (
    PolicyAction,
    PolicyDecision,
    ToolPolicy,
)


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        (
            "read_file",
            PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="tool is allowed by configured policy",
            ),
        ),
        (
            "apply_patch",
            PolicyDecision(
                action=PolicyAction.ASK,
                reason="tool requires user confirmation",
            ),
        ),
        (
            "delete_everything",
            PolicyDecision(
                action=PolicyAction.DENY,
                reason="tool is denied by default policy",
            ),
        ),
    ],
)
def test_policy_returns_configured_decision(
    tool_name: str,
    expected: PolicyDecision,
) -> None:
    policy = ToolPolicy(
        allowed_tools={"read_file"},
        confirmation_required_tools={"apply_patch"},
    )
    call = ToolCall(
        call_id="call_1",
        tool_name=tool_name,
        arguments={},
    )

    assert policy.decide(call) == expected


def test_policy_rejects_overlapping_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="tools cannot be both allow and ask: read_file",
    ):
        ToolPolicy(
            allowed_tools={"read_file"},
            confirmation_required_tools={"read_file"},
        )


def test_policy_decision_requires_reason() -> None:
    with pytest.raises(
        ValueError,
        match="policy reason must not be blank",
    ):
        PolicyDecision(
            action=PolicyAction.DENY,
            reason=" ",
        )