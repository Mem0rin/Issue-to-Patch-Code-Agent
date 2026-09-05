"""RunResult contracts, exercised only through its public constructor."""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from enum import StrEnum
from importlib import import_module
from typing import Callable, NamedTuple, Protocol, cast

import pytest

from code_agent.kernel_types import (
    FinalAnswer, HistoryItem, Message, MessageRole,
    ModelFeedback, ToolCall, ToolResult,
)
from code_agent.run_budget import BudgetStopReason, RunUsage


class _Result(Protocol):
    run_id: str
    status: StrEnum
    final_answer: FinalAnswer | None
    reason: StrEnum | None
    history: tuple[HistoryItem, ...]
    usage: RunUsage
    elapsed_seconds: float


class _API(NamedTuple):
    factory: Callable[..., _Result]
    status: type[StrEnum]
    reason: type[StrEnum]


@pytest.fixture
def api() -> _API:
    module = import_module("code_agent.agent_loop")
    factory: object = getattr(module, "RunResult", None)
    status: object = getattr(module, "RunStatus", None)
    reason: object = getattr(module, "ModelStopReason", None)
    assert callable(factory), "agent_loop.RunResult must be implemented"
    assert isinstance(status, type) and issubclass(status, StrEnum)
    assert isinstance(reason, type) and issubclass(reason, StrEnum)
    return _API(cast(Callable[..., _Result], factory), status, reason)


def _arguments(api: _API) -> dict[str, object]:
    return {
        "run_id": "run-result",
        "status": api.status["COMPLETED"],
        "final_answer": FinalAnswer("42"),
        "reason": None,
        "history": (
            Message(MessageRole.USER, "Compute 17 + 25."),
            Message(MessageRole.ASSISTANT, "42"),
        ),
        "usage": RunUsage(2, 2, 17, 4, 0),
        "elapsed_seconds": 0.25,
    }


def _mutated(item: HistoryItem | FinalAnswer, field: str, value: object) -> object:
    object.__setattr__(item, field, value)
    return item


def test_completed_result_preserves_fields_and_is_frozen(api: _API) -> None:
    arguments = _arguments(api)
    result = api.factory(**arguments)

    assert result.run_id == "run-result"
    assert result.status.value == "completed"
    assert result.final_answer == FinalAnswer("42")
    assert result.reason is None
    assert result.history == arguments["history"]
    assert result.usage == RunUsage(2, 2, 17, 4, 0)
    assert result.usage.total_tokens == 21
    assert result.elapsed_seconds == 0.25
    with pytest.raises(FrozenInstanceError):
        setattr(result, "status", api.status["MODEL_FAILED"])


@pytest.mark.parametrize("reason", list(BudgetStopReason))
def test_budget_stop_preserves_reason_and_zero_usage(
    api: _API, reason: BudgetStopReason,
) -> None:
    arguments = _arguments(api)
    arguments.update(
        status=api.status["BUDGET_STOPPED"], final_answer=None, reason=reason,
        history=(Message(MessageRole.USER, "Task"),),
        usage=RunUsage(), elapsed_seconds=0,
    )
    result = api.factory(**arguments)

    assert result.status.value == "budget_stopped"
    assert result.final_answer is None
    assert result.reason is reason
    assert result.usage == RunUsage()
    assert len(result.history) == 1
    assert result.elapsed_seconds == 0


@pytest.mark.parametrize("name", [
    "PROVIDER_NON_RETRYABLE", "PROVIDER_RETRIES_EXHAUSTED",
    "INVALID_OUTPUT_RETRIES_EXHAUSTED", "DUPLICATE_CALL_ID",
])
def test_model_stop_preserves_fixed_reason_and_unknown_consumption(
    api: _API, name: str,
) -> None:
    arguments = _arguments(api)
    arguments.update(
        status=api.status["MODEL_FAILED"], final_answer=None,
        reason=api.reason[name], usage=RunUsage(1, 0, 0, 0, 128),
        history=(Message(MessageRole.USER, "Task"),),
    )
    result = api.factory(**arguments)

    assert result.status.value == "model_failed"
    assert result.reason is api.reason[name]
    assert result.final_answer is None
    assert result.usage.total_tokens == 128


def test_history_accepts_all_item_types_and_empty_tool_output(api: _API) -> None:
    arguments = _arguments(api)
    history = (
        Message(MessageRole.SYSTEM, "Use tools."),
        Message(MessageRole.USER, "Task"),
        ModelFeedback("Return a JSON action."),
        ToolCall("c1", "lookup", {"nested": [None, {"count": 1}]}),
        ToolResult("c1", "", False),
        ToolCall("c2", "lookup", {}),
        ToolResult("c2", "denied", True),
        Message(MessageRole.ASSISTANT, "42"),
    )
    arguments["history"] = history
    result = api.factory(**arguments)
    assert result.history == history


@pytest.mark.parametrize("status", ["COMPLETED", "BUDGET_STOPPED", "MODEL_FAILED"])
@pytest.mark.parametrize("has_answer", [True, False])
@pytest.mark.parametrize("reason_kind", ["none", "budget", "model"])
def test_terminal_combination_matrix(
    api: _API, status: str, has_answer: bool, reason_kind: str,
) -> None:
    arguments = _arguments(api)
    reasons: dict[str, StrEnum | None] = {
        "none": None,
        "budget": BudgetStopReason.TOKENS_EXHAUSTED,
        "model": api.reason["PROVIDER_NON_RETRYABLE"],
    }
    arguments.update(
        status=api.status[status],
        final_answer=FinalAnswer("42") if has_answer else None,
        reason=reasons[reason_kind],
    )
    valid = (
        (status, has_answer, reason_kind)
        in {
            ("COMPLETED", True, "none"),
            ("BUDGET_STOPPED", False, "budget"),
            ("MODEL_FAILED", False, "model"),
        }
    )
    if valid:
        result = api.factory(**arguments)
        assert result.status is api.status[status]
        assert result.reason is reasons[reason_kind]
    else:
        with pytest.raises(ValueError, match="status.*final_answer.*reason"):
            api.factory(**arguments)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("run_id", None, TypeError),
    ("run_id", 1, TypeError),
    ("run_id", "", ValueError),
    ("run_id", "  ", ValueError),
    ("status", None, TypeError),
    ("status", "completed", TypeError),
    ("status", BudgetStopReason.TOKENS_EXHAUSTED, TypeError),
    ("final_answer", "42", TypeError),
    ("final_answer", Message(MessageRole.ASSISTANT, "42"), TypeError),
    ("reason", "tokens_exhausted", TypeError),
    ("reason", 0, TypeError),
    ("history", None, TypeError),
    ("history", [], TypeError),
    ("history", (), ValueError),
    ("history", ("Task",), TypeError),
    ("usage", None, TypeError),
    ("usage", {}, TypeError),
])
def test_invalid_result_fields_are_rejected_with_context(
    api: _API, field: str, value: object, error: type[Exception],
) -> None:
    arguments = _arguments(api)
    arguments[field] = value
    with pytest.raises(error, match=field) as caught:
        api.factory(**arguments)
    assert "validate RunResult" in caught.value.__notes__


@pytest.mark.parametrize("value", [0, 0.0, 0.000001, 1, 1.5])
def test_elapsed_seconds_includes_zero_and_positive_values(
    api: _API, value: float,
) -> None:
    arguments = _arguments(api)
    arguments["elapsed_seconds"] = value
    result = api.factory(**arguments)
    assert result.elapsed_seconds == value


@pytest.mark.parametrize(("value", "error"), [
    (None, TypeError), (True, TypeError), ("1", TypeError),
    (-0.000001, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (-float("inf"), ValueError),
    (10**400, ValueError),
])
def test_invalid_elapsed_seconds_are_rejected(
    api: _API, value: object, error: type[Exception],
) -> None:
    arguments = _arguments(api)
    arguments["elapsed_seconds"] = value
    with pytest.raises(error, match="elapsed_seconds"):
        api.factory(**arguments)


@pytest.mark.parametrize("field", [
    "model_calls", "action_steps", "input_tokens", "output_tokens", "reserved_tokens",
])
@pytest.mark.parametrize(("value", "error"), [
    (True, TypeError), (0.0, TypeError), (None, TypeError),
    ("0", TypeError), (-1, ValueError),
])
def test_invalid_nested_usage_is_rejected(
    api: _API, field: str, value: object, error: type[Exception],
) -> None:
    arguments = _arguments(api)
    usage = RunUsage()
    object.__setattr__(usage, field, value)
    arguments["usage"] = usage
    with pytest.raises(error, match=f"usage.{field}"):
        api.factory(**arguments)


@pytest.mark.parametrize(("kind", "field", "value", "error"), [
    ("message", "role", "user", TypeError),
    ("message", "content", 1, TypeError),
    ("message", "content", " ", ValueError),
    ("feedback", "error_message", None, TypeError),
    ("feedback", "error_message", "", ValueError),
    ("call", "call_id", None, TypeError),
    ("call", "call_id", "", ValueError),
    ("call", "tool_name", None, TypeError),
    ("call", "tool_name", " ", ValueError),
    ("call", "arguments", [], TypeError),
    ("call", "arguments", {1: "bad"}, TypeError),
    ("call", "arguments", {"bad": object()}, TypeError),
    ("call", "arguments", {"bad": float("inf")}, ValueError),
    ("result", "call_id", 1, TypeError),
    ("result", "call_id", " ", ValueError),
    ("result", "content", None, TypeError),
    ("result", "is_error", 0, TypeError),
    ("answer", "content", None, TypeError),
    ("answer", "content", " ", ValueError),
])
def test_malformed_nested_history_or_answer_is_rejected(
    api: _API, kind: str, field: str, value: object, error: type[Exception],
) -> None:
    items: dict[str, HistoryItem | FinalAnswer] = {
        "message": Message(MessageRole.USER, "Task"),
        "feedback": ModelFeedback("Try again"),
        "call": ToolCall("c1", "lookup", {}),
        "result": ToolResult("c1", "", False),
        "answer": FinalAnswer("42"),
    }
    item = _mutated(items[kind], field, value)
    arguments = _arguments(api)
    if kind == "answer":
        arguments["final_answer"] = item
    else:
        arguments["history"] = (Message(MessageRole.USER, "Task"), item)
    with pytest.raises(error) as caught:
        api.factory(**arguments)
    context = str(caught.value) + " ".join(caught.value.__notes__)
    assert "validate RunResult" in context
    assert field in context


def test_history_without_user_is_rejected(api: _API) -> None:
    arguments = _arguments(api)
    arguments["history"] = (Message(MessageRole.ASSISTANT, "42"),)
    with pytest.raises(ValueError, match="USER"):
        api.factory(**arguments)


@pytest.mark.parametrize("last", [
    Message(MessageRole.USER, "42"),
    Message(MessageRole.ASSISTANT, "different"),
    ToolResult("c1", "42", False),
])
def test_completed_history_must_end_with_matching_assistant_answer(
    api: _API, last: HistoryItem,
) -> None:
    arguments = _arguments(api)
    arguments["history"] = (Message(MessageRole.USER, "Task"), last)
    with pytest.raises(ValueError, match="completed history"):
        api.factory(**arguments)


@pytest.mark.parametrize("field", [
    "run_id", "status", "final_answer", "reason",
    "history", "usage", "elapsed_seconds",
])
def test_missing_result_fields_are_rejected(api: _API, field: str) -> None:
    arguments = _arguments(api)
    del arguments[field]
    with pytest.raises(TypeError, match=field):
        api.factory(**arguments)


def test_constructor_has_no_runtime_effects_for_valid_or_invalid_input(
    api: _API, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        calls.append("runtime")
        raise AssertionError("RunResult must not call runtime dependencies")

    monkeypatch.setattr("code_agent.event_store.JsonlEventStore.append", forbidden)
    monkeypatch.setattr("code_agent.model_call.complete_model_with_retry", forbidden)
    monkeypatch.setattr("code_agent.tool_controller.ToolController.handle", forbidden)
    monkeypatch.setattr("code_agent.run_budget.BudgetTracker.__init__", forbidden)
    result = api.factory(**_arguments(api))
    invalid = _arguments(api)
    invalid["elapsed_seconds"] = -1
    with pytest.raises(ValueError, match="elapsed_seconds"):
        api.factory(**invalid)
    assert result.final_answer == FinalAnswer("42")
    assert calls == []


def test_unexpected_history_failure_preserves_identity_and_context(api: _API) -> None:
    failure = RuntimeError("history iteration failed")

    class BrokenHistory(tuple[HistoryItem, ...]):
        def __iter__(self) -> Iterator[HistoryItem]:
            raise failure

    arguments = _arguments(api)
    arguments["history"] = BrokenHistory((Message(MessageRole.USER, "Task"),))
    with pytest.raises(RuntimeError) as caught:
        api.factory(**arguments)
    assert caught.value is failure
    assert "validate RunResult" in caught.value.__notes__
