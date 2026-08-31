# src/code_agent/tool_policy.py

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .kernel_types import ToolCall


class PolicyAction(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"

@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("policy reason must not be blank")


class ToolPolicy:
    def __init__(
        self,
        *,
        allowed_tools: Iterable[str],
        confirmation_required_tools: Iterable[str],
    ) -> None:
        allowed = frozenset(allowed_tools)
        confirmation_required = frozenset(
            confirmation_required_tools
        )

        overlap = allowed & confirmation_required
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"tools cannot be both allow and ask: {names}"
            )

        self._allowed_tools = allowed
        self._confirmation_required_tools = (
            confirmation_required
        )

    def decide(self, call: ToolCall) -> PolicyDecision:
        name = call.tool_name

        if name in self._allowed_tools:
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="tool is allowed by configured policy",
            )

        if name in self._confirmation_required_tools:
            return PolicyDecision(
                action=PolicyAction.ASK,
                reason="tool requires user confirmation",
            )

        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="tool is denied by default policy",
            )