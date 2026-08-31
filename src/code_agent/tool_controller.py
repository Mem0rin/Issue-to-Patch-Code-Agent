'''综合工具运行时、策略和待确认请求，处理工具调用。'''

from collections.abc import Callable
from typing import TypeAlias

from .kernel_types import ToolCall, ToolResult
from .tool_approval import (
    ApprovalRequest,
    ApprovalResponse,
    PendingApprovals,
)
from .tool_policy import (
    PolicyAction,
    ToolPolicy,
)
from .tool_runtime import (
    PreparedToolCall,
    ToolRuntime,
)


ApprovalIdFactory: TypeAlias = Callable[[], str]

ToolOutcome: TypeAlias = (
    ToolResult | ApprovalRequest
)


class ToolController:
    def __init__(
        self,
        *,
        runtime: ToolRuntime,
        policy: ToolPolicy,
        pending_approvals: PendingApprovals,
        approval_id_factory: ApprovalIdFactory,
    ) -> None:
        self._runtime = runtime
        self._policy = policy
        self._pending_approvals = pending_approvals
        self._approval_id_factory = (
            approval_id_factory
        )

    def handle(
        self,
        call: ToolCall,
    ) -> ToolOutcome:
        preparation = self._runtime.prepare(call)

        # prepare 返回 ToolResult，说明工具未知或参数错误
        if isinstance(preparation, ToolResult):
            return preparation

        # 通过上面的判断后，mypy 能确认它是 PreparedToolCall
        prepared: PreparedToolCall = preparation
        decision = self._policy.decide(
            prepared.call
        )

        if decision.action is PolicyAction.ALLOW:
            # TODO 1：执行已准备好的调用
            return self._runtime.execute(prepared)

        if decision.action is PolicyAction.DENY:
            # TODO 2：返回拒绝的错误 ToolResult
            return ToolResult(
                call_id=prepared.call.call_id,
                content=decision.reason,
                is_error=True,
            )

        if decision.action is PolicyAction.ASK:
            request = ApprovalRequest(
                # TODO 3：通过注入的 factory 生成 approval_id
                approval_id=self._approval_id_factory(),
                call=prepared.call,
                reason=decision.reason,
            )

            # TODO 4：保存待确认请求
            self._pending_approvals.add(request)

            # 返回事件，不执行 handler
            return request

        raise AssertionError(
            f"unsupported policy action: {decision.action}"
        )

    def resolve_approval(
        self,
        response: ApprovalResponse,
    ) -> ToolResult:
        # resolve 会取出并删除请求，防止重复消费
        resolution = self._pending_approvals.resolve(
            response
        )
        call = resolution.request.call

        if not resolution.approved:
            return ToolResult(
                call_id=call.call_id,
                content="tool execution rejected by user",
                is_error=True,
            )

        # 等待确认期间，工具配置或运行条件可能变化，
        # 所以批准后重新解析并校验一次
        preparation = self._runtime.prepare(call)

        if isinstance(preparation, ToolResult):
            return preparation

        prepared: PreparedToolCall = preparation

        # 用户批准不能覆盖当前的硬性 DENY
        current_decision = self._policy.decide(
            prepared.call
        )

        if current_decision.action is PolicyAction.DENY:
            return ToolResult(
                call_id=prepared.call.call_id,
                content=current_decision.reason,
                is_error=True,
            )

        # 当前仍为 ASK，说明用户刚才的批准有效；
        # 当前变成 ALLOW，也可以执行
        return self._runtime.execute(prepared)
