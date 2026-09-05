from collections.abc import Callable, Sequence
from copy import deepcopy

from .kernel_types import Action, ToolSpec
from .model_adapter import (
    ConsumptionState, InvalidModelOutputError, ModelAdapter, ModelProviderError,
    ModelResponse, Retryability, TokenUsage,
)
from .model_call import ModelRetryKind, complete_model_with_retry
from .run_budget import BudgetTracker, _finite_clock_value
from .event_store import EventDraft, JsonlEventStore
from .tool_policy import PolicyDecision
from .tool_approval import ApprovalRequest, ApprovalResponse
from .tool_controller import ToolController

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from time import monotonic
from enum import StrEnum
from typing import TypeAlias, cast

from ._event_payload import validate_payload
from .kernel_types import (
    FinalAnswer, HistoryItem, Message, MessageRole,
    ModelFeedback, ToolCall, ToolResult,
)
from .model_adapter import ModelCallBudget
from .run_budget import BudgetStopReason, RunBudget, RunUsage


def _require_non_blank(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_int_at_least(
    value: object,
    field_name: str,
    minimum: int,
) -> None:
    if type(value) is not int:
        raise TypeError(
            f"{field_name} must be an integer (bool is excluded)"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")


def _require_positive_seconds(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")

    try:
        seconds = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite") from error

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field_name} must be finite and > 0")


def _validate_initial_history(history: tuple[Message, ...]) -> None:
    if not isinstance(history, tuple):
        raise TypeError("history must be a tuple")
    if not history:
        raise ValueError("history must contain at least one message")

    for index, message in enumerate(history):
        field_name = f"history[{index}]"
        if not isinstance(message, Message):
            raise TypeError(f"{field_name} must be a Message")
        if not isinstance(message.role, MessageRole):
            raise TypeError(f"{field_name}.role must be a MessageRole")
        if message.role not in (MessageRole.SYSTEM, MessageRole.USER):
            raise ValueError(f"{field_name}.role must be SYSTEM or USER")
        _require_non_blank(message.content, f"{field_name}.content")


def _initial_history_has_user(history: tuple[Message, ...]) -> bool:
    for index, message in enumerate(history):
        if message.role == MessageRole.USER:
            return True
    return False

@dataclass(frozen=True)
class RunRequest:
    run_id: str
    history: tuple[Message, ...]
    budget: RunBudget
    model_budget: ModelCallBudget
    reserved_tokens: int
    max_provider_retries: int = 0
    max_invalid_output_retries: int = 0

    def __post_init__(self) -> None:
        try:
            self._validate()
        except Exception as error:
            error.add_note("validate RunRequest")
            raise

    def _validate(self) -> None:
        _require_non_blank(self.run_id, "run_id")
        _validate_initial_history(self.history)

        if not isinstance(self.budget, RunBudget):
            raise TypeError("budget must be a RunBudget")
        if not isinstance(self.model_budget, ModelCallBudget):
            raise TypeError("model_budget must be a ModelCallBudget")

        for field_name in (
            "max_model_calls",
            "max_action_steps",
            "max_total_tokens",
        ):
            _require_int_at_least(
                getattr(self.budget, field_name),
                f"budget.{field_name}",
                1,
            )
        _require_positive_seconds(
            self.budget.max_elapsed_seconds,
            "budget.max_elapsed_seconds",
        )
        _require_int_at_least(
            self.model_budget.max_output_tokens,
            "model_budget.max_output_tokens",
            1,
        )
        _require_int_at_least(self.reserved_tokens, "reserved_tokens", 1)
        _require_int_at_least(
            self.max_provider_retries,
            "max_provider_retries",
            0,
        )
        _require_int_at_least(
            self.max_invalid_output_retries,
            "max_invalid_output_retries",
            0,
        )

        if not _initial_history_has_user(self.history):
            raise ValueError("history must contain at least one USER message")


class RunStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_STOPPED = "budget_stopped"
    MODEL_FAILED = "model_failed"


class ModelStopReason(StrEnum):
    PROVIDER_NON_RETRYABLE = "provider_non_retryable"
    PROVIDER_RETRIES_EXHAUSTED = "provider_retries_exhausted"
    INVALID_OUTPUT_RETRIES_EXHAUSTED = "invalid_output_retries_exhausted"
    DUPLICATE_CALL_ID = "duplicate_call_id"


RunStopReason: TypeAlias = BudgetStopReason | ModelStopReason


def _require_elapsed_seconds(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("elapsed_seconds must be a number")
    try:
        seconds = float(value)
    except OverflowError as error:
        raise ValueError("elapsed_seconds must be finite") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("elapsed_seconds must be finite and >= 0")


def _validate_run_history(history: tuple[HistoryItem, ...]) -> None:
    if not isinstance(history, tuple):
        raise TypeError("history must be a tuple")
    if not history:
        raise ValueError("history must contain at least one message")

    has_user = False
    for index, item in enumerate(history):
        field = f"history[{index}]"
        if isinstance(item, Message):
            if not isinstance(item.role, MessageRole):
                raise TypeError(f"{field}.role must be a MessageRole")
            _require_non_blank(item.content, f"{field}.content")
            has_user = has_user or item.role is MessageRole.USER
        elif isinstance(item, ModelFeedback):
            _require_non_blank(item.error_message, f"{field}.error_message")
        elif isinstance(item, ToolCall):
            _require_non_blank(item.call_id, f"{field}.call_id")
            _require_non_blank(item.tool_name, f"{field}.tool_name")
            if not isinstance(item.arguments, Mapping):
                raise TypeError(f"{field}.arguments must be a Mapping")
            try:
                validate_payload(item.arguments)
            except (TypeError, ValueError) as error:
                error.add_note(f"validate {field}.arguments")
                raise
        elif isinstance(item, ToolResult):
            _require_non_blank(item.call_id, f"{field}.call_id")
            if not isinstance(item.content, str):
                raise TypeError(f"{field}.content must be a string")
            if type(item.is_error) is not bool:
                raise TypeError(f"{field}.is_error must be a bool")
        else:
            raise TypeError(f"{field} must be a HistoryItem")

    if not has_user:
        raise ValueError("history must contain at least one USER message")


def _termination_is_consistent(
    status: RunStatus,
    final_answer: FinalAnswer | None,
    reason: RunStopReason | None,
) -> bool:
    if status is RunStatus.COMPLETED:
        return final_answer is not None and reason is None
    elif status is RunStatus.BUDGET_STOPPED:
        return final_answer is None and isinstance(reason, BudgetStopReason)
    elif status is RunStatus.MODEL_FAILED:
        return final_answer is None and isinstance(reason, ModelStopReason)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    final_answer: FinalAnswer | None
    reason: RunStopReason | None
    history: tuple[HistoryItem, ...]
    usage: RunUsage
    elapsed_seconds: float

    def __post_init__(self) -> None:
        try:
            self._validate()
        except Exception as error:
            error.add_note("validate RunResult")
            raise

    def _validate(self) -> None:
        _require_non_blank(self.run_id, "run_id")
        if not isinstance(self.status, RunStatus):
            raise TypeError("status must be a RunStatus")
        if self.final_answer is not None:
            if not isinstance(self.final_answer, FinalAnswer):
                raise TypeError("final_answer must be a FinalAnswer or None")
            _require_non_blank(self.final_answer.content, "final_answer.content")
        if self.reason is not None and not isinstance(
            self.reason, (BudgetStopReason, ModelStopReason)
        ):
            raise TypeError("reason must be a RunStopReason or None")

        _validate_run_history(self.history)
        if not isinstance(self.usage, RunUsage):
            raise TypeError("usage must be a RunUsage")
        for field in (
            "model_calls", "action_steps", "input_tokens",
            "output_tokens", "reserved_tokens",
        ):
            _require_int_at_least(getattr(self.usage, field), f"usage.{field}", 0)
        _require_elapsed_seconds(self.elapsed_seconds)

        if not _termination_is_consistent(
            self.status, self.final_answer, self.reason
        ):
            raise ValueError("status, final_answer and reason are inconsistent")

        if self.status is RunStatus.COMPLETED:
            assert self.final_answer is not None
            last = self.history[-1]
            if (
                not isinstance(last, Message)
                or last.role is not MessageRole.ASSISTANT
                or last.content != self.final_answer.content
            ):
                raise ValueError("completed history must end with the final answer")


def _append_tool_result(
    history: tuple[HistoryItem, ...],
    result: ToolResult,
) -> tuple[HistoryItem, ...]:
    history = (*history, result)
    return history


def _check_loop_tool_call(call: ToolCall) -> None:
    _require_non_blank(call.call_id, "model action.call_id")
    _require_non_blank(call.tool_name, "model action.tool_name")
    if not isinstance(call.arguments, Mapping):
        raise TypeError("model action.arguments must be a Mapping")
    validate_payload(call.arguments)


def _check_approval_request(
    approval: ApprovalRequest,
    call: ToolCall,
) -> None:
    _require_non_blank(approval.approval_id, "approval.approval_id")
    _require_non_blank(approval.reason, "approval.reason")
    if not isinstance(approval.call, ToolCall) or approval.call != call:
        raise ValueError("approval.call must match the requested ToolCall")


def _check_approval_response(
    response: ApprovalResponse,
    approval_id: str,
) -> None:
    if not isinstance(response, ApprovalResponse):
        raise TypeError("approval response must be an ApprovalResponse")
    _require_non_blank(response.approval_id, "approval response.approval_id")
    if response.approval_id != approval_id:
        raise ValueError("approval response.approval_id must match the request")
    if type(response.approved) is not bool:
        raise TypeError("approval response.approved must be a bool")


def _check_tool_result(result: ToolResult, call_id: str) -> None:
    if not isinstance(result, ToolResult):
        raise TypeError("tool result must be a ToolResult")
    _require_non_blank(result.call_id, "tool result.call_id")
    if result.call_id != call_id:
        raise ValueError("tool result.call_id must match the ToolCall")
    if not isinstance(result.content, str):
        raise TypeError("tool result.content must be a string")
    if type(result.is_error) is not bool:
        raise TypeError("tool result.is_error must be a bool")


def _validated_token_usage(usage: TokenUsage) -> TokenUsage:
    if not isinstance(usage, TokenUsage):
        raise TypeError("usage must be a TokenUsage")
    _require_int_at_least(usage.input_tokens, "usage.input_tokens", 0)
    _require_int_at_least(usage.output_tokens, "usage.output_tokens", 0)
    return TokenUsage(usage.input_tokens, usage.output_tokens)


def _check_provider_error(error: ModelProviderError) -> None:
    if not isinstance(error.consumption_state, ConsumptionState):
        raise TypeError("provider error.consumption_state must be a ConsumptionState")
    if not isinstance(error.retryability, Retryability):
        raise TypeError("provider error.retryability must be a Retryability")
    if error.consumption_state is ConsumptionState.ACTUAL_USAGE:
        if error.usage is None:
            raise ValueError("provider error.usage is required for ACTUAL_USAGE")
    elif error.usage is not None:
        raise ValueError("provider error.usage must be None unless ACTUAL_USAGE")
    if error.usage is not None:
        _validated_token_usage(error.usage)


def _snapshot_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _snapshot_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_json(item) for item in value]
    return value


def _validated_model_response(response: ModelResponse) -> ModelResponse:
    if not isinstance(response, ModelResponse):
        raise TypeError("model response must be a ModelResponse")
    usage = _validated_token_usage(response.usage)
    action = response.action

    validated_action: Action
    if isinstance(action, FinalAnswer):
        _require_non_blank(action.content, "model response.action.content")
        validated_action = FinalAnswer(action.content)
    elif isinstance(action, ToolCall):
        _check_loop_tool_call(action)
        arguments = cast(Mapping[str, object], _snapshot_json(action.arguments))
        validated_action = ToolCall(action.call_id, action.tool_name, arguments)
    else:
        raise TypeError("model response.action must be a ToolCall or FinalAnswer")

    return ModelResponse(action=validated_action, usage=usage)


class _ValidatedModelAdapter:
    def __init__(self, adapter: ModelAdapter) -> None:
        if not callable(getattr(adapter, "complete", None)):
            raise TypeError("adapter.complete must be callable")
        self._adapter = adapter

    def complete(
        self,
        history: Sequence[HistoryItem],
        tools: Sequence[ToolSpec],
        budget: ModelCallBudget,
    ) -> ModelResponse:
        phase = "call model adapter"
        try:
            try:
                response = self._adapter.complete(
                    history=deepcopy(tuple(history)),
                    tools=deepcopy(tuple(tools)),
                    budget=budget,
                )
            except ModelProviderError as error:
                phase = "validate provider error"
                _check_provider_error(error)
                raise
            except InvalidModelOutputError as error:
                phase = "validate invalid output error"
                if error.usage is not None:
                    _validated_token_usage(error.usage)
                raise

            phase = "validate model response"
            return _validated_model_response(response)
        except Exception as error:
            error.add_note(phase)
            raise


class _RunTimeExpired(RuntimeError):
    pass


class _RunClock:
    def __init__(self, source: Callable[[], float]) -> None:
        self.source = source
        self.first: float | None = None
        self.last: float | None = None

    def __call__(self) -> float:
        value = _finite_clock_value(self.source())
        if self.last is not None and value < self.last:
            raise ValueError("run clock must be nondecreasing")
        if self.first is not None and not math.isfinite(value - self.first):
            raise ValueError("run elapsed time must be finite")
        if self.first is None:
            self.first = value
        self.last = value
        return value

    @property
    def elapsed(self) -> float:
        assert self.first is not None and self.last is not None
        return self.last - self.first


def _action_payload(action: Action) -> dict[str, object]:
    payload = asdict(action)
    payload["type"] = "tool_call" if isinstance(action, ToolCall) else "final_answer"
    return payload


class _RunTrace:
    def __init__(self, store: JsonlEventStore) -> None:
        self.store = store
        self.broken = False
        self.failure: Exception | None = None
        self.phase = "start run"

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        self.phase = f"persist {event_type}"
        try:
            self.store.append(EventDraft(event_type, payload))
        except Exception as error:
            self.broken = True
            self.failure = error
            error.add_note(self.phase)
            raise

    def prepared(self, attempt_index: int, reserved_tokens: int, usage: RunUsage) -> None:
        self.emit("model_call_prepared", {
            "attempt_index": attempt_index, "reserved_tokens": reserved_tokens,
            "usage": asdict(usage),
        })

    def completed(
        self, attempt_index: int, response: ModelResponse, usage: RunUsage, latency: float,
    ) -> None:
        self.emit("model_call_completed", {
            "attempt_index": attempt_index, "action": _action_payload(response.action),
            "actual_usage": asdict(response.usage), "usage": asdict(usage),
            "latency": latency,
        })

    def failed(
        self, attempt_index: int, error: Exception, usage: RunUsage, latency: float,
    ) -> None:
        actual = None
        retryability = None
        state = ConsumptionState.UNKNOWN_CONSUMPTION
        if isinstance(error, ModelProviderError):
            kind = "provider_error"
            state = error.consumption_state
            retryability = error.retryability.value
            actual = error.usage
        elif isinstance(error, InvalidModelOutputError):
            kind = "invalid_model_output"
            actual = error.usage
            if actual is not None:
                state = ConsumptionState.ACTUAL_USAGE
        else:
            kind = "unexpected_error"
        self.emit("model_call_failed", {
            "attempt_index": attempt_index, "error_kind": kind,
            "consumption_state": state.value, "retryability": retryability,
            "actual_usage": None if actual is None else asdict(actual),
            "usage": asdict(usage), "latency": latency,
        })

    def retry(
        self, attempt_index: int, kind: ModelRetryKind,
        feedback: ModelFeedback | None, usage: RunUsage,
    ) -> None:
        self.emit("model_retry_scheduled", {
            "attempt_index": attempt_index, "kind": kind.value,
            "feedback": None if feedback is None else asdict(feedback),
            "usage": asdict(usage),
        })


def _validate_run_inputs(
    request: RunRequest, adapter: ModelAdapter, controller: ToolController,
    tools: tuple[ToolSpec, ...], event_store: JsonlEventStore,
    approval_handler: Callable[[ApprovalRequest], ApprovalResponse],
    clock: Callable[[], float],
) -> tuple[RunRequest, tuple[ToolSpec, ...]]:
    if not isinstance(request, RunRequest):
        raise TypeError("request must be a RunRequest")
    checked_request = replace(request)
    if not callable(getattr(adapter, "complete", None)):
        raise TypeError("adapter.complete must be callable")
    if not isinstance(controller, ToolController):
        raise TypeError("controller must be a ToolController")
    for method in ("handle", "resolve_approval"):
        if not callable(getattr(controller, method, None)):
            raise TypeError(f"controller.{method} must be callable")
    if not isinstance(event_store, JsonlEventStore):
        raise TypeError("event_store must be a JsonlEventStore")
    for method in ("read_all", "append"):
        if not callable(getattr(event_store, method, None)):
            raise TypeError(f"event_store.{method} must be callable")
    if not callable(approval_handler):
        raise TypeError("approval_handler must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    if not isinstance(tools, tuple):
        raise TypeError("tools must be a tuple")
    names: set[str] = set()
    snapshots: list[ToolSpec] = []
    for index, spec in enumerate(tools):
        field = f"tools[{index}]"
        if not isinstance(spec, ToolSpec):
            raise TypeError(f"{field} must be a ToolSpec")
        _require_non_blank(spec.name, f"{field}.name")
        _require_non_blank(spec.description, f"{field}.description")
        if spec.name in names:
            raise ValueError(f"{field}.name must be unique")
        names.add(spec.name)
        if not isinstance(spec.input_schema, Mapping):
            raise TypeError(f"{field}.input_schema must be a Mapping")
        try:
            validate_payload(spec.input_schema)
        except (TypeError, ValueError) as error:
            error.add_note(f"validate {field}.input_schema")
            raise
        snapshots.append(ToolSpec(
            spec.name, spec.description,
            cast(Mapping[str, object], _snapshot_json(spec.input_schema)),
        ))
    if event_store.run_id != checked_request.run_id:
        raise ValueError("event_store.run_id must match request.run_id")
    return deepcopy(checked_request), tuple(snapshots)


def _run_loop(
    request: RunRequest,
    *,
    adapter: ModelAdapter,
    controller: ToolController,
    tools: tuple[ToolSpec, ...],
    tracker: BudgetTracker,
    approval_handler: Callable[[ApprovalRequest], ApprovalResponse],
    trace: _RunTrace | None = None,
) -> RunResult:
    history: tuple[HistoryItem, ...] = ()
    seen_call_ids: set[str] = set()
    phase = "initialize history"

    def finish(
        status: RunStatus,
        *,
        answer: FinalAnswer | None = None,
        reason: RunStopReason | None = None,
    ) -> RunResult:
        return RunResult(
            run_id=request.run_id,
            status=status,
            final_answer=answer,
            reason=reason,
            history=deepcopy(history),
            usage=tracker.usage,
            elapsed_seconds=tracker.elapsed_seconds,
        )

    def record_retry(
        attempt_index: int,
        kind: ModelRetryKind,
        feedback: ModelFeedback | None,
        usage: RunUsage,
    ) -> None:
        nonlocal history
        if trace is not None:
            trace.retry(attempt_index, kind, feedback, usage)
        if feedback is not None:
            history = (*history, feedback)

    def record_policy(
        call: ToolCall, decision: PolicyDecision, after_approval: bool,
    ) -> None:
        assert trace is not None
        trace.emit("tool_policy_decided", {
            "call_id": call.call_id, "action": decision.action.value,
            "reason": decision.reason,
            "phase": "approval" if after_approval else "initial",
        })
        if tracker.time_exhausted:
            raise _RunTimeExpired()

    try:
        history = deepcopy(request.history)
        phase = "validate model adapter"
        adapter = _ValidatedModelAdapter(adapter)
        while True:
            phase = "complete model"
            try:
                response = complete_model_with_retry(
                    adapter, tracker,
                    history=history,
                    tools=tools,
                    budget=request.model_budget,
                    reserved_tokens=request.reserved_tokens,
                    max_provider_retries=request.max_provider_retries,
                    max_invalid_output_retries=request.max_invalid_output_retries,
                    on_retry=record_retry,
                    observer=trace,
                )
            except InvalidModelOutputError:
                return finish(
                    RunStatus.MODEL_FAILED,
                    reason=ModelStopReason.INVALID_OUTPUT_RETRIES_EXHAUSTED,
                )
            except ModelProviderError as error:
                reason = (
                    ModelStopReason.PROVIDER_NON_RETRYABLE
                    if error.retryability is Retryability.NON_RETRYABLE
                    else ModelStopReason.PROVIDER_RETRIES_EXHAUSTED
                )
                return finish(RunStatus.MODEL_FAILED, reason=reason)

            if isinstance(response, BudgetStopReason):
                return finish(RunStatus.BUDGET_STOPPED, reason=response)

            phase = "process model action"
            if tracker.elapsed_seconds >= request.budget.max_elapsed_seconds:
                return finish(
                    RunStatus.BUDGET_STOPPED,
                    reason=BudgetStopReason.TIME_EXHAUSTED,
                )

            action = deepcopy(response.action)
            if isinstance(action, FinalAnswer):
                history = (*history, Message(MessageRole.ASSISTANT, action.content))
                return finish(RunStatus.COMPLETED, answer=action)
            if action.call_id in seen_call_ids:
                return finish(
                    RunStatus.MODEL_FAILED,
                    reason=ModelStopReason.DUPLICATE_CALL_ID,
                )
            seen_call_ids.add(action.call_id)
            history = (*history, action)

            phase = "handle tool"
            if tracker.elapsed_seconds >= request.budget.max_elapsed_seconds:
                return finish(
                    RunStatus.BUDGET_STOPPED,
                    reason=BudgetStopReason.TIME_EXHAUSTED,
                )
            tool_started_at = tracker.elapsed_seconds if trace is not None else 0.0
            if trace is not None:
                trace.emit("tool_call_requested", asdict(action))
            try:
                outcome = (
                    controller.handle(deepcopy(action)) if trace is None
                    else controller.handle(deepcopy(action), on_decision=record_policy)
                )
            except _RunTimeExpired:
                return finish(RunStatus.BUDGET_STOPPED, reason=BudgetStopReason.TIME_EXHAUSTED)

            if isinstance(outcome, ApprovalRequest):
                _check_approval_request(outcome, action)
                phase = "request approval"
                if trace is not None:
                    trace.emit("approval_requested", {
                        "approval_id": outcome.approval_id, "call_id": action.call_id,
                        "reason": outcome.reason,
                    })
                approval = approval_handler(deepcopy(outcome))
                _check_approval_response(approval, outcome.approval_id)
                if trace is not None:
                    trace.emit("approval_resolved", {
                        "approval_id": approval.approval_id, "call_id": action.call_id,
                        "approved": approval.approved,
                    })
                if tracker.elapsed_seconds >= request.budget.max_elapsed_seconds:
                    return finish(
                        RunStatus.BUDGET_STOPPED,
                        reason=BudgetStopReason.TIME_EXHAUSTED,
                    )
                phase = "resolve approval"
                try:
                    outcome = (
                        controller.resolve_approval(approval) if trace is None
                        else controller.resolve_approval(approval, on_decision=record_policy)
                    )
                except _RunTimeExpired:
                    return finish(RunStatus.BUDGET_STOPPED, reason=BudgetStopReason.TIME_EXHAUSTED)

            phase = "record tool result"
            _check_tool_result(outcome, action.call_id)
            if trace is not None:
                trace.emit("tool_result_recorded", {
                    **asdict(outcome), "latency": tracker.elapsed_seconds - tool_started_at,
                })
            history = _append_tool_result(history, deepcopy(outcome))
    except Exception as error:
        error.add_note(f"run {request.run_id}: {phase}")
        if trace is not None and not trace.broken:
            trace.phase = phase
        raise


def run_agent(
    request: RunRequest,
    *,
    adapter: ModelAdapter,
    controller: ToolController,
    tools: tuple[ToolSpec, ...],
    event_store: JsonlEventStore,
    approval_handler: Callable[[ApprovalRequest], ApprovalResponse],
    clock: Callable[[], float] = monotonic,
) -> RunResult:
    """Run one synchronous agent with an exclusive controller and empty trace."""
    try:
        request, tools = _validate_run_inputs(
            request, adapter, controller, tools, event_store, approval_handler, clock,
        )
        previous = event_store.read_all()
        if not isinstance(previous, tuple):
            raise TypeError("event_store.read_all must return a tuple")
        if previous:
            raise ValueError("run_agent requires an empty trace")
        run_clock = _RunClock(clock)
        tracker = BudgetTracker(request.budget, clock=run_clock)
    except Exception as error:
        error.add_note("prepare run_agent")
        raise

    trace = _RunTrace(event_store)
    started = False
    phase = "start run"
    try:
        trace.emit("run_started", {
            "run_id": request.run_id, "history": [asdict(item) for item in request.history],
            "tools": [tool.name for tool in tools], "budget": asdict(request.budget),
            "model_budget": asdict(request.model_budget),
            "reserved_tokens": request.reserved_tokens,
            "max_provider_retries": request.max_provider_retries,
            "max_invalid_output_retries": request.max_invalid_output_retries,
        })
        started = True
        phase = "run loop"
        result = _run_loop(
            request, adapter=adapter, controller=controller, tools=tools,
            tracker=tracker, approval_handler=approval_handler, trace=trace,
        )
        phase = "finish run"
        trace.emit("run_finished", {
            "status": result.status.value,
            "reason": None if result.reason is None else result.reason.value,
            "final_answer": None if result.final_answer is None else result.final_answer.content,
            "usage": asdict(result.usage), "elapsed_seconds": result.elapsed_seconds,
        })
        return result
    except Exception as error:
        error.add_note(f"run {request.run_id}: {phase}")
        if trace.broken:
            failure = trace.failure
            assert failure is not None
            if failure is error:
                raise
            failure.add_note(f"run {request.run_id}: {trace.phase}")
            raise failure from failure.__cause__
        if started:
            try:
                trace.emit("run_aborted", {
                    "error_kind": "unexpected_error", "phase": trace.phase,
                    "usage": asdict(tracker.usage), "elapsed_seconds": run_clock.elapsed,
                })
            except Exception as write_error:
                write_error.add_note(
                    f"run {request.run_id}: abort trace failed after operation failure"
                )
                raise write_error from error
        raise
