"""Control-flow tests for the internal loop after entry validation.

RunRequest/RunResult own their existing input matrices. This suite owns
turn transitions, callback outputs and runtime error propagation.
"""
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from importlib import import_module
from typing import cast

import pytest

from code_agent.agent_loop import ModelStopReason, RunRequest, RunResult, RunStatus
from code_agent.kernel_types import (
    FinalAnswer, HistoryItem, Message, MessageRole, ModelFeedback, ToolCall, ToolResult, ToolSpec,
)
from code_agent.model_adapter import (
    FakeModelAdapter, FakeOutcome, InvalidModelOutputError, ModelCallBudget,
    ModelProviderError, ModelResponse, Retryability, TokenUsage,
)
from code_agent.run_budget import BudgetStopReason, BudgetTracker, RunBudget
from code_agent.tool_approval import ApprovalRequest, ApprovalResponse, PendingApprovals
from code_agent.tool_controller import ToolController
from code_agent.tool_policy import ToolPolicy
from code_agent.tool_registry import ToolDefinition, ToolRegistry
from code_agent.tool_runtime import ToolExecutionError, ToolRuntime

Core = Callable[..., RunResult]


@pytest.fixture
def core() -> Core:
    candidate: object = getattr(import_module("code_agent.agent_loop"), "_run_loop", None)
    assert callable(candidate), "agent_loop._run_loop must be implemented"
    return cast(Core, candidate)


def _response(action: ToolCall | FinalAnswer) -> ModelResponse:
    return ModelResponse(action, TokenUsage(1, 1))


def _call(call_id: str = "c1", name: str = "calc") -> ToolCall:
    return ToolCall(call_id, name, {"value": 42})


def _approve(request: ApprovalRequest) -> ApprovalResponse:
    return ApprovalResponse(request.approval_id, True)


class Harness:
    def __init__(
        self, outcomes: Sequence[FakeOutcome], *,
        budget: RunBudget | None = None,
        policy: str = "allow",
        handler: Callable[[Mapping[str, object]], str] | None = None,
    ) -> None:
        self.now = 0.0
        self.executed: list[Mapping[str, object]] = []
        self.pending = PendingApprovals()
        self.adapter = FakeModelAdapter(outcomes)
        self.spec = ToolSpec("calc", "Calculate.", {"type": "object"})

        def execute(arguments: Mapping[str, object]) -> str:
            self.executed.append(dict(arguments))
            return handler(arguments) if handler is not None else "42"

        definition = ToolDefinition(
            self.spec, lambda arguments: None, execute,
        )
        self.controller = ToolController(
            runtime=ToolRuntime(ToolRegistry([definition])),
            policy=ToolPolicy(
                allowed_tools={"calc"} if policy == "allow" else set(),
                confirmation_required_tools={"calc"} if policy == "ask" else set(),
            ),
            pending_approvals=self.pending,
            approval_id_factory=lambda: "approval-1",
        )
        self.request = RunRequest(
            "core-run", (Message(MessageRole.USER, "Task"),),
            budget or RunBudget(10, 10, 1000, 5), ModelCallBudget(20), 10,
        )
        self.tracker = BudgetTracker(self.request.budget, clock=lambda: self.now)

    def run(
        self, core: Core, *,
        approval_handler: Callable[[ApprovalRequest], ApprovalResponse] = _approve,
        tools: tuple[ToolSpec, ...] | None = None,
    ) -> RunResult:
        return core(
            self.request, adapter=self.adapter, controller=self.controller,
            tools=(self.spec,) if tools is None else tools,
            tracker=self.tracker, approval_handler=approval_handler,
        )


def test_direct_final_at_last_allowed_call_completes_without_tools(core: Core) -> None:
    harness = Harness(
        [_response(FinalAnswer("done"))], budget=RunBudget(1, 1, 10, 5),
    )
    result = harness.run(core, tools=())
    assert result.status is RunStatus.COMPLETED
    assert result.final_answer == FinalAnswer("done")
    assert result.history == (*harness.request.history, Message(MessageRole.ASSISTANT, "done"))
    assert result.usage.model_calls == result.usage.action_steps == 1
    assert result.usage.total_tokens == 2
    assert harness.executed == []
    assert harness.adapter.calls[0].tools == ()


@pytest.mark.parametrize("turns", [1, 3])
def test_tool_turns_preserve_pairing_and_feed_results_to_next_model(
    core: Core, turns: int,
) -> None:
    outcomes: list[FakeOutcome] = [
        _response(_call(f"c{index}")) for index in range(turns)
    ]
    outcomes.append(_response(FinalAnswer("done")))
    harness = Harness(outcomes)
    original = harness.request.history
    result = harness.run(core)

    assert result.status is RunStatus.COMPLETED
    assert len(harness.executed) == turns
    assert result.usage.model_calls == result.usage.action_steps == turns + 1
    for index in range(turns):
        call = _call(f"c{index}")
        tool_result = ToolResult(call.call_id, "42", False)
        assert result.history[1 + index * 2:3 + index * 2] == (call, tool_result)
        assert harness.adapter.calls[index + 1].history[-2:] == (call, tool_result)
    assert harness.request.history == original
    assert len(original) == 1


@pytest.mark.parametrize(("budget", "reason"), [
    (RunBudget(1, 5, 100, 5), BudgetStopReason.MODEL_CALLS_EXHAUSTED),
    (RunBudget(5, 1, 100, 5), BudgetStopReason.ACTION_STEPS_EXHAUSTED),
])
def test_last_allowed_tool_executes_before_next_model_is_stopped(
    core: Core, budget: RunBudget, reason: BudgetStopReason,
) -> None:
    harness = Harness([_response(_call())], budget=budget)
    result = harness.run(core)
    assert result.status is RunStatus.BUDGET_STOPPED
    assert result.reason is reason
    assert len(harness.executed) == len(harness.adapter.calls) == 1
    assert result.history[-1] == ToolResult("c1", "42", False)


def test_reservation_over_balance_stops_before_model_or_tool(core: Core) -> None:
    harness = Harness([], budget=RunBudget(5, 5, 9, 5))
    result = harness.run(core)
    assert result.reason is BudgetStopReason.TOKENS_EXHAUSTED
    assert result.usage.model_calls == 0
    assert result.history == harness.request.history
    assert harness.adapter.calls == []
    assert harness.executed == []


@pytest.mark.parametrize("policy", ["deny", "ask"])
def test_rejection_is_returned_to_model_as_tool_error(core: Core, policy: str) -> None:
    harness = Harness([_response(_call()), _response(FinalAnswer("denied"))], policy=policy)
    result = harness.run(
        core, approval_handler=lambda request: ApprovalResponse(request.approval_id, False),
    )
    feedback = harness.adapter.calls[1].history[-1]
    assert isinstance(feedback, ToolResult)
    assert feedback.call_id == "c1" and feedback.is_error
    assert result.status is RunStatus.COMPLETED
    assert harness.executed == []


def test_approval_executes_once_then_continues(core: Core) -> None:
    harness = Harness([_response(_call()), _response(FinalAnswer("done"))], policy="ask")
    decisions: list[str] = []

    def approve(request: ApprovalRequest) -> ApprovalResponse:
        decisions.append(request.approval_id)
        return _approve(request)

    result = harness.run(core, approval_handler=approve)
    assert result.status is RunStatus.COMPLETED
    assert decisions == ["approval-1"]
    assert len(harness.executed) == 1


def test_unknown_tool_becomes_feedback_without_execution(core: Core) -> None:
    harness = Harness([_response(_call(name="unknown")), _response(FinalAnswer("done"))])
    result = harness.run(core)
    feedback = result.history[2]
    assert isinstance(feedback, ToolResult) and feedback.is_error
    assert feedback.call_id == "c1"
    assert harness.executed == []


def test_expected_tool_error_becomes_feedback(core: Core) -> None:
    def fail(arguments: Mapping[str, object]) -> str:
        raise ToolExecutionError("cannot calculate")

    harness = Harness([_response(_call()), _response(FinalAnswer("done"))], handler=fail)
    result = harness.run(core)
    feedback = result.history[2]
    assert isinstance(feedback, ToolResult) and feedback.is_error
    assert "cannot calculate" in feedback.content
    assert len(harness.executed) == 1


def test_duplicate_call_id_is_counted_but_not_executed_again(core: Core) -> None:
    harness = Harness([_response(_call()), _response(_call())])
    result = harness.run(core)
    assert result.status is RunStatus.MODEL_FAILED
    assert result.reason is ModelStopReason.DUPLICATE_CALL_ID
    assert result.usage.model_calls == result.usage.action_steps == 2
    assert len(harness.executed) == 1
    assert len(result.history) == 3


@pytest.mark.parametrize(("error", "reason"), [
    (ModelProviderError("provider"), ModelStopReason.PROVIDER_NON_RETRYABLE),
    (ModelProviderError("retry", retryability=Retryability.RETRYABLE),
     ModelStopReason.PROVIDER_RETRIES_EXHAUSTED),
    (InvalidModelOutputError("format", usage=TokenUsage(1, 1)),
     ModelStopReason.INVALID_OUTPUT_RETRIES_EXHAUSTED),
])
def test_classified_model_failure_returns_fixed_reason(
    core: Core, error: ModelProviderError | InvalidModelOutputError,
    reason: ModelStopReason,
) -> None:
    harness = Harness([error])
    result = harness.run(core)
    assert result.status is RunStatus.MODEL_FAILED
    assert result.reason is reason
    assert result.usage.model_calls == 1 and result.usage.action_steps == 0
    assert harness.executed == []
    if isinstance(error, ModelProviderError):
        assert result.usage.reserved_tokens == 10
    else:
        assert result.usage.total_tokens == 2


@pytest.mark.parametrize("end", [4.999, 5.0, 5.001])
def test_model_return_time_boundary_prevents_late_tool_start(
    core: Core, monkeypatch: pytest.MonkeyPatch, end: float,
) -> None:
    harness = Harness([_response(_call())], budget=RunBudget(1, 2, 100, 5))
    original = harness.adapter.complete

    def complete(
        history: Sequence[HistoryItem], tools: Sequence[ToolSpec], budget: ModelCallBudget,
    ) -> ModelResponse:
        response = original(history, tools, budget)
        harness.now = end
        return response

    monkeypatch.setattr(harness.adapter, "complete", complete)
    result = harness.run(core)
    assert result.status is RunStatus.BUDGET_STOPPED
    if end < 5:
        assert result.reason is BudgetStopReason.MODEL_CALLS_EXHAUSTED
        assert len(harness.executed) == 1
    else:
        assert result.reason is BudgetStopReason.TIME_EXHAUSTED
        assert harness.executed == []
    assert result.usage.action_steps == 1


def test_expired_approval_does_not_execute_or_consume_pending_request(core: Core) -> None:
    harness = Harness([_response(_call())], policy="ask")

    def approve(request: ApprovalRequest) -> ApprovalResponse:
        harness.now = 5.0
        return _approve(request)

    result = harness.run(core, approval_handler=approve)
    assert result.reason is BudgetStopReason.TIME_EXHAUSTED
    assert harness.executed == []
    pending = harness.pending.resolve(ApprovalResponse("approval-1", False))
    assert pending.request.call == _call()


@pytest.mark.parametrize(("field", "value", "error"), [
    ("response", None, TypeError),
    ("approval_id", None, TypeError),
    ("approval_id", "", ValueError),
    ("approval_id", "wrong", ValueError),
    ("approved", 1, TypeError),
    ("approved", None, TypeError),
])
def test_invalid_approval_response_is_rejected_before_consumption(
    core: Core, field: str, value: object, error: type[Exception],
) -> None:
    harness = Harness([_response(_call())], policy="ask")

    def approve(request: ApprovalRequest) -> ApprovalResponse:
        if field == "response":
            return cast(ApprovalResponse, value)
        response = _approve(request)
        object.__setattr__(response, field, value)
        return response

    with pytest.raises(error, match="approval") as caught:
        harness.run(core, approval_handler=approve)
    assert "run core-run: request approval" in caught.value.__notes__
    assert harness.executed == [] and len(harness.adapter.calls) == 1
    assert harness.pending.resolve(ApprovalResponse("approval-1", False)).request.call == _call()


@pytest.mark.parametrize(("field", "value", "error"), [
    ("result", None, TypeError),
    ("call_id", None, TypeError),
    ("call_id", "", ValueError),
    ("call_id", "wrong", ValueError),
    ("content", None, TypeError),
    ("is_error", 0, TypeError),
])
def test_invalid_tool_result_stops_before_next_model(
    core: Core, monkeypatch: pytest.MonkeyPatch,
    field: str, value: object, error: type[Exception],
) -> None:
    harness = Harness([_response(_call())])
    result = ToolResult("c1", "42", False)
    if field != "result":
        object.__setattr__(result, field, value)

    def handle(call: ToolCall) -> ToolResult:
        return cast(ToolResult, value) if field == "result" else result

    monkeypatch.setattr(harness.controller, "handle", handle)
    with pytest.raises(error, match="tool result"):
        harness.run(core)
    assert len(harness.adapter.calls) == 1 and harness.executed == []


def test_tool_argument_mutation_does_not_rewrite_history_or_model_response(core: Core) -> None:
    call = _call()

    def mutate(arguments: Mapping[str, object]) -> str:
        cast(dict[str, object], arguments)["value"] = -1
        return "changed"

    harness = Harness([_response(call), _response(FinalAnswer("done"))], handler=mutate)
    result = harness.run(core)
    stored = result.history[1]
    assert isinstance(stored, ToolCall)
    assert stored.arguments == {"value": 42}
    assert call.arguments == {"value": 42}


def test_unexpected_tool_failure_retains_identity_and_run_phase(core: Core) -> None:
    failure = RuntimeError("tool broke")

    def fail(arguments: Mapping[str, object]) -> str:
        raise failure

    harness = Harness([_response(_call())], handler=fail)
    with pytest.raises(RuntimeError) as caught:
        harness.run(core)
    assert caught.value is failure
    assert "run core-run: handle tool" in failure.__notes__
    assert len(harness.adapter.calls) == len(harness.executed) == 1


def test_unexpected_model_failure_keeps_reservation_and_context(
    core: Core, monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = RuntimeError("model broke")
    harness = Harness([])

    def fail(
        history: Sequence[HistoryItem], tools: Sequence[ToolSpec], budget: ModelCallBudget,
    ) -> ModelResponse:
        raise failure

    monkeypatch.setattr(harness.adapter, "complete", fail)
    with pytest.raises(RuntimeError) as caught:
        harness.run(core)
    assert caught.value is failure
    assert "run core-run: complete model" in failure.__notes__
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.executed == []

@pytest.mark.parametrize(("field", "value", "error"), [
    ("call_id", None, TypeError), ("call_id", " ", ValueError),
    ("tool_name", None, TypeError), ("tool_name", "", ValueError),
    ("arguments", [], TypeError), ("arguments", {"bad": float("inf")}, ValueError),
])
def test_invalid_model_tool_call_is_rejected_before_controller(
    core: Core, field: str, value: object, error: type[Exception],
) -> None:
    call = _call()
    object.__setattr__(call, field, value)
    harness = Harness([_response(call)])
    with pytest.raises(error):
        harness.run(core)
    assert harness.executed == [] and len(harness.adapter.calls) == 1


@pytest.mark.parametrize(("field", "value", "error"), [
    ("approval_id", None, TypeError), ("approval_id", " ", ValueError),
    ("reason", None, TypeError), ("reason", "", ValueError),
    ("call", _call("different"), ValueError),
])
def test_invalid_approval_request_is_rejected_before_user_callback(
    core: Core, monkeypatch: pytest.MonkeyPatch,
    field: str, value: object, error: type[Exception],
) -> None:
    harness = Harness([_response(_call())])
    approval = ApprovalRequest("approval-1", _call(), "Confirm")
    object.__setattr__(approval, field, value)
    monkeypatch.setattr(harness.controller, "handle", lambda call: approval)
    asked: list[str] = []

    def approve(request: ApprovalRequest) -> ApprovalResponse:
        asked.append(request.approval_id)
        return _approve(request)

    with pytest.raises(error, match="approval"):
        harness.run(core, approval_handler=approve)
    assert asked == []
    assert harness.executed == []


def test_time_expiry_between_response_and_controller_prevents_tool(core: Core) -> None:
    harness = Harness([_response(_call())])
    times = iter([0.0, 0.0, 4.9, 5.0, 5.0])
    harness.tracker = BudgetTracker(harness.request.budget, clock=lambda: next(times))
    result = harness.run(core)
    assert result.reason is BudgetStopReason.TIME_EXHAUSTED
    assert result.history[-1] == _call()
    assert harness.executed == []


def test_unrecognized_model_action_is_an_unexpected_contract_error(core: Core) -> None:
    action = cast(FinalAnswer, Message(MessageRole.ASSISTANT, "unexpected"))
    harness = Harness([_response(action)])
    with pytest.raises(TypeError, match="model response.action") as caught:
        harness.run(core)
    assert "run core-run: complete model" in caught.value.__notes__
    assert harness.tracker.usage.action_steps == 0
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.executed == []


def test_zero_remaining_time_stops_before_model(core: Core) -> None:
    harness = Harness([])
    harness.now = 5.0
    result = harness.run(core)
    assert result.reason is BudgetStopReason.TIME_EXHAUSTED
    assert harness.adapter.calls == []
    assert harness.executed == []


@pytest.mark.parametrize("failures", [1, 2])
def test_retry_feedback_survives_tool_turn_and_final_history(core: Core, failures: int) -> None:
    errors: list[FakeOutcome] = [
        InvalidModelOutputError(f"repair-{index}", usage=TokenUsage(1, 0))
        for index in range(failures)
    ]
    harness = Harness([*errors, _response(_call()), _response(FinalAnswer("done"))])
    harness.request = replace(harness.request, max_invalid_output_retries=failures)
    result = harness.run(core)
    feedbacks = tuple(ModelFeedback(f"repair-{index}") for index in range(failures))
    expected = (
        *harness.request.history, *feedbacks, _call(), ToolResult("c1", "42", False),
    )
    assert harness.adapter.calls[-1].history == expected
    assert result.history == (*expected, Message(MessageRole.ASSISTANT, "done"))
    assert harness.adapter.calls[failures].history == (*harness.request.history, *feedbacks)
    assert len(harness.request.history) == 1
    assert len(harness.executed) == 1
    assert result.usage.action_steps == 2


def test_scheduled_feedback_is_retained_when_budget_prevents_retry(core: Core) -> None:
    harness = Harness(
        [InvalidModelOutputError("repair")], budget=RunBudget(1, 2, 100, 5),
    )
    harness.request = replace(harness.request, max_invalid_output_retries=1)
    result = harness.run(core)
    assert result.reason is BudgetStopReason.MODEL_CALLS_EXHAUSTED
    assert result.history == (*harness.request.history, ModelFeedback("repair"))
    assert len(harness.adapter.calls) == 1 and harness.executed == []
    assert result.usage.reserved_tokens == 10


def test_final_failed_attempt_does_not_append_unscheduled_feedback(core: Core) -> None:
    harness = Harness([
        InvalidModelOutputError("first"), InvalidModelOutputError("last"),
    ])
    harness.request = replace(harness.request, max_invalid_output_retries=1)
    result = harness.run(core)
    assert result.reason is ModelStopReason.INVALID_OUTPUT_RETRIES_EXHAUSTED
    assert result.history == (*harness.request.history, ModelFeedback("first"))
    assert len(harness.adapter.calls) == 2 and harness.executed == []


def test_provider_retry_does_not_add_model_feedback_to_run_history(core: Core) -> None:
    harness = Harness([
        ModelProviderError("temporary", retryability=Retryability.RETRYABLE),
        _response(FinalAnswer("done")),
    ])
    harness.request = replace(harness.request, max_provider_retries=1)
    result = harness.run(core)
    assert result.history == (*harness.request.history, Message(MessageRole.ASSISTANT, "done"))
    assert len(harness.adapter.calls) == 2
