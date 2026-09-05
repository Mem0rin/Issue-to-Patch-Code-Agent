"""Model-output validation must precede budget settlement and tool execution."""
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType
from typing import cast

import pytest

from code_agent.agent_loop import RunRequest, RunResult, RunStatus, _run_loop
from code_agent.kernel_types import (
    FinalAnswer, HistoryItem, Message, MessageRole, ToolCall, ToolSpec,
)
from code_agent.model_adapter import (
    ConsumptionState, InvalidModelOutputError, ModelCallBudget,
    ModelProviderError, ModelResponse, Retryability, TokenUsage,
)
from code_agent.run_budget import BudgetTracker, RunBudget
from code_agent.tool_approval import ApprovalResponse, PendingApprovals
from code_agent.tool_controller import ToolController
from code_agent.tool_policy import ToolPolicy
from code_agent.tool_registry import ToolDefinition, ToolRegistry
from code_agent.tool_runtime import ToolRuntime


class ScriptedAdapter:
    def __init__(self, first: object) -> None:
        self.outcomes = [first, ModelResponse(FinalAnswer("done"), TokenUsage(1, 1))]
        self.calls = 0
        self.schemas: list[dict[str, object]] = []
        self.on_call: Callable[[Sequence[ToolSpec]], None] = lambda tools: None

    def complete(
        self, history: Sequence[HistoryItem], tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        self.schemas.append(deepcopy(dict(tools[0].input_schema)))
        self.calls += 1
        self.on_call(tools)
        outcome = self.outcomes[self.calls - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return cast(ModelResponse, outcome)


class Harness:
    def __init__(self, outcome: object) -> None:
        self.executed: list[Mapping[str, object]] = []
        self.adapter = ScriptedAdapter(outcome)
        self.spec = ToolSpec("echo", "Echo.", {"type": "object"})

        def execute(arguments: Mapping[str, object]) -> str:
            self.executed.append(arguments)
            return "42"

        definition = ToolDefinition(self.spec, lambda arguments: None, execute)
        self.controller = ToolController(
            runtime=ToolRuntime(ToolRegistry([definition])),
            policy=ToolPolicy(allowed_tools={"echo"}, confirmation_required_tools=set()),
            pending_approvals=PendingApprovals(),
            approval_id_factory=lambda: "approval-1",
        )
        self.request = RunRequest(
            "boundary-run", (Message(MessageRole.USER, "Task"),),
            RunBudget(5, 5, 1000, 5), ModelCallBudget(64), 10,
        )
        self.tracker = BudgetTracker(self.request.budget, clock=lambda: 0.0)

    def run(self) -> RunResult:
        return _run_loop(
            self.request, adapter=self.adapter, controller=self.controller,
            tools=(self.spec,), tracker=self.tracker,
            approval_handler=lambda request: ApprovalResponse(request.approval_id, False),
        )


def _response() -> ModelResponse:
    return ModelResponse(ToolCall("c1", "echo", {}), TokenUsage(1, 1))


@pytest.mark.parametrize(("field", "value", "error"), [
    ("input_tokens", True, TypeError),
    ("input_tokens", None, TypeError),
    ("input_tokens", 1.0, TypeError),
    ("input_tokens", "1", TypeError),
    ("input_tokens", -1, ValueError),
    ("output_tokens", True, TypeError),
    ("output_tokens", None, TypeError),
    ("output_tokens", 1.0, TypeError),
    ("output_tokens", "1", TypeError),
    ("output_tokens", -1, ValueError),
])
def test_invalid_usage_is_rejected_before_settlement_or_tool(
    field: str, value: object, error: type[Exception],
) -> None:
    response = _response()
    object.__setattr__(response.usage, field, value)
    harness = Harness(response)

    with pytest.raises(error, match="usage") as caught:
        harness.run()

    assert "validate model response" in caught.value.__notes__
    assert harness.tracker.usage.action_steps == 0
    assert harness.tracker.usage.input_tokens == harness.tracker.usage.output_tokens == 0
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.executed == [] and harness.adapter.calls == 1


@pytest.mark.parametrize("value", [None, {}, 1])
def test_non_usage_object_is_rejected_without_releasing_reservation(value: object) -> None:
    response = _response()
    object.__setattr__(response, "usage", value)
    harness = Harness(response)
    with pytest.raises(TypeError, match="usage"):
        harness.run()
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.tracker.usage.action_steps == 0
    assert harness.executed == []


@pytest.mark.parametrize("value", [None, {}, FinalAnswer("wrong envelope")])
def test_non_response_object_is_rejected_without_settlement(value: object) -> None:
    harness = Harness(value)
    with pytest.raises(TypeError, match="model response"):
        harness.run()
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.tracker.usage.action_steps == 0
    assert harness.executed == []


@pytest.mark.parametrize(("kind", "field", "value", "error"), [
    ("final", "content", None, TypeError),
    ("final", "content", " ", ValueError),
    ("call", "call_id", None, TypeError),
    ("call", "call_id", "", ValueError),
    ("call", "tool_name", None, TypeError),
    ("call", "tool_name", " ", ValueError),
    ("call", "arguments", [], TypeError),
    ("call", "arguments", {"bad": float("inf")}, ValueError),
])
def test_invalid_action_is_rejected_before_action_counter_increases(
    kind: str, field: str, value: object, error: type[Exception],
) -> None:
    action = FinalAnswer("done") if kind == "final" else ToolCall("c1", "echo", {})
    object.__setattr__(action, field, value)
    harness = Harness(ModelResponse(action, TokenUsage(1, 1)))
    with pytest.raises(error):
        harness.run()
    assert harness.tracker.usage.action_steps == 0
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.executed == []


@pytest.mark.parametrize("tokens", [0, 1, 10**30])
def test_valid_usage_accepts_zero_and_nonnegative_integer_counts(tokens: int) -> None:
    harness = Harness(ModelResponse(FinalAnswer("done"), TokenUsage(tokens, tokens)))
    result = harness.run()
    assert result.status is RunStatus.COMPLETED
    assert result.usage.input_tokens == result.usage.output_tokens == tokens
    assert result.usage.action_steps == 1
    assert result.usage.reserved_tokens == 0


def test_response_is_rebuilt_from_validated_action_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ModelResponse(FinalAnswer("done"), TokenUsage(1, 2))
    harness = Harness(original)
    recorded: list[ModelResponse] = []
    record = harness.tracker.record_model_response

    def capture(response: ModelResponse) -> None:
        recorded.append(response)
        record(response)

    monkeypatch.setattr(harness.tracker, "record_model_response", capture)
    result = harness.run()
    assert result.status is RunStatus.COMPLETED
    assert recorded[0] == original
    assert recorded[0] is not original
    assert recorded[0].action is not original.action
    assert recorded[0].usage is not original.usage


def test_mapping_arguments_are_normalized_before_core_deepcopy() -> None:
    nested = MappingProxyType({"count": 1})
    arguments = MappingProxyType({"nested": [nested, None, False, "text"]})
    harness = Harness(ModelResponse(ToolCall("c1", "echo", arguments), TokenUsage(1, 1)))
    result = harness.run()
    assert result.status is RunStatus.COMPLETED
    call = result.history[1]
    assert isinstance(call, ToolCall)
    assert type(call.arguments) is dict
    assert call.arguments == {"nested": [{"count": 1}, None, False, "text"]}
    assert len(harness.executed) == 1
    assert arguments["nested"][0] is nested


@pytest.mark.parametrize(("state", "usage", "reserved", "total"), [
    (ConsumptionState.NO_CONSUMPTION, None, 0, 0),
    (ConsumptionState.UNKNOWN_CONSUMPTION, None, 10, 10),
    (ConsumptionState.ACTUAL_USAGE, TokenUsage(2, 3), 0, 5),
])
def test_valid_provider_failure_preserves_consumption_and_error_identity(
    state: ConsumptionState, usage: TokenUsage | None, reserved: int, total: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = ModelProviderError("provider", consumption_state=state, usage=usage)
    harness = Harness(failure)
    record = harness.tracker.record_provider_error
    recorded: list[ModelProviderError] = []

    def capture(error: ModelProviderError) -> None:
        recorded.append(error)
        record(error)

    monkeypatch.setattr(harness.tracker, "record_provider_error", capture)
    result = harness.run()
    assert result.status is RunStatus.MODEL_FAILED
    assert result.usage.reserved_tokens == reserved
    assert result.usage.total_tokens == total
    assert result.usage.action_steps == 0
    assert recorded == [failure] and recorded[0] is failure


@pytest.mark.parametrize(("field", "value", "error"), [
    ("consumption_state", "no_consumption", TypeError),
    ("consumption_state", None, TypeError),
    ("retryability", "retryable", TypeError),
    ("retryability", None, TypeError),
    ("usage", TokenUsage(1, 1), ValueError),
])
def test_invalid_provider_metadata_preserves_error_chain_and_reservation(
    field: str, value: object, error: type[Exception],
) -> None:
    failure = ModelProviderError("provider", consumption_state=ConsumptionState.NO_CONSUMPTION)
    setattr(failure, field, value)
    harness = Harness(failure)
    with pytest.raises(error, match=field) as caught:
        harness.run()
    assert caught.value.__context__ is failure
    assert "validate provider error" in caught.value.__notes__
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.executed == [] and harness.adapter.calls == 1


def test_actual_usage_failure_requires_usage_before_settlement() -> None:
    failure = ModelProviderError(
        "provider", consumption_state=ConsumptionState.ACTUAL_USAGE, usage=TokenUsage(1, 1),
    )
    failure.usage = None
    harness = Harness(failure)
    with pytest.raises(ValueError, match="usage") as caught:
        harness.run()
    assert caught.value.__context__ is failure
    assert harness.tracker.usage.reserved_tokens == 10


@pytest.mark.parametrize("kind", ["provider", "invalid_output"])
def test_invalid_error_usage_is_rejected_before_budget_recording(kind: str) -> None:
    usage = TokenUsage(1, 1)
    object.__setattr__(usage, "input_tokens", True)
    failure = (
        ModelProviderError("provider", consumption_state=ConsumptionState.ACTUAL_USAGE, usage=usage)
        if kind == "provider" else InvalidModelOutputError("output", usage=usage)
    )
    harness = Harness(failure)
    with pytest.raises(TypeError, match="usage.input_tokens") as caught:
        harness.run()
    assert caught.value.__context__ is failure
    assert harness.tracker.usage.reserved_tokens == 10
    assert harness.tracker.usage.action_steps == 0


@pytest.mark.parametrize(("usage", "total"), [(None, 10), (TokenUsage(0, 0), 0)])
def test_valid_invalid_output_failure_preserves_known_or_unknown_usage(
    usage: TokenUsage | None, total: int,
) -> None:
    harness = Harness(InvalidModelOutputError("output", usage=usage))
    result = harness.run()
    assert result.status is RunStatus.MODEL_FAILED
    assert result.usage.total_tokens == total
    assert result.usage.action_steps == 0


def test_retry_attempt_gets_fresh_tool_schema_snapshot() -> None:
    failure = ModelProviderError(
        "retry", consumption_state=ConsumptionState.NO_CONSUMPTION,
        retryability=Retryability.RETRYABLE,
    )
    harness = Harness(failure)
    harness.request = replace(harness.request, max_provider_retries=1)

    def mutate_first(tools: Sequence[ToolSpec]) -> None:
        if harness.adapter.calls == 1:
            cast(dict[str, object], tools[0].input_schema)["type"] = "corrupted"

    harness.adapter.on_call = mutate_first
    result = harness.run()
    assert result.status is RunStatus.COMPLETED
    assert harness.adapter.schemas == [{"type": "object"}, {"type": "object"}]
    assert harness.spec.input_schema == {"type": "object"}
    assert result.usage.model_calls == 2


@pytest.mark.parametrize("adapter", [None, object(), {"complete": "not a method"}])
def test_invalid_adapter_is_rejected_before_model_reservation(adapter: object) -> None:
    harness = Harness(None)
    harness.adapter = cast(ScriptedAdapter, adapter)
    with pytest.raises(TypeError, match="adapter.complete"):
        harness.run()
    assert harness.tracker.usage.model_calls == 0
    assert harness.tracker.usage.reserved_tokens == 0
    assert harness.executed == []


def test_unexpected_adapter_exception_keeps_identity_and_operation_context() -> None:
    failure = RuntimeError("adapter broke")
    harness = Harness(failure)
    with pytest.raises(RuntimeError) as caught:
        harness.run()
    assert caught.value is failure
    assert "call model adapter" in caught.value.__notes__
    assert "run boundary-run: complete model" in caught.value.__notes__
    assert harness.tracker.usage.reserved_tokens == 10
