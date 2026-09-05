"""RunRequest contract tests for the Agent Loop entry point."""

from collections.abc import Iterator
from dataclasses import FrozenInstanceError
from importlib import import_module
from typing import Callable, Protocol, cast

import pytest

from code_agent.kernel_types import Message, MessageRole
from code_agent.model_adapter import ModelCallBudget
from code_agent.run_budget import RunBudget


class _Request(Protocol):
    run_id: str
    history: tuple[Message, ...]
    budget: RunBudget
    model_budget: ModelCallBudget
    reserved_tokens: int
    max_provider_retries: int
    max_invalid_output_retries: int


RequestFactory = Callable[..., _Request]


@pytest.fixture
def request_factory() -> RequestFactory:
    module = import_module("code_agent.agent_loop")
    factory: object = getattr(module, "RunRequest", None)
    assert callable(factory), "code_agent.agent_loop.RunRequest must be callable"
    return cast(RequestFactory, factory)


def _valid_arguments() -> dict[str, object]:
    return {
        "run_id": "run-loop",
        "history": (Message(MessageRole.USER, "Compute 17 + 25."),),
        "budget": RunBudget(4, 3, 1000, 30.0),
        "model_budget": ModelCallBudget(64),
        "reserved_tokens": 128,
    }


def _budget_with(field_name: str, value: object) -> RunBudget:
    budget = RunBudget(4, 3, 1000, 30.0)
    object.__setattr__(budget, field_name, value)
    return budget


def _model_budget_with(value: object) -> ModelCallBudget:
    budget = ModelCallBudget(64)
    object.__setattr__(budget, "max_output_tokens", value)
    return budget


def _message_with(field_name: str, value: object) -> Message:
    message = Message(MessageRole.USER, "Task")
    object.__setattr__(message, field_name, value)
    return message


def test_request_preserves_configuration_and_defaults(
    request_factory: RequestFactory,
) -> None:
    arguments = _valid_arguments()
    history = arguments["history"]

    request = request_factory(**arguments)

    assert request.run_id == "run-loop"
    assert request.history == history
    assert request.budget == RunBudget(4, 3, 1000, 30.0)
    assert request.model_budget == ModelCallBudget(64)
    assert request.reserved_tokens == 128
    assert request.max_provider_retries == 0
    assert request.max_invalid_output_retries == 0
    assert arguments["history"] == history


def test_request_accepts_inclusive_minimum_limits(
    request_factory: RequestFactory,
) -> None:
    arguments = _valid_arguments()
    arguments.update(
        budget=RunBudget(1, 1, 1, 0.001),
        model_budget=ModelCallBudget(1),
        reserved_tokens=1,
    )

    request = request_factory(**arguments)

    assert request.budget.max_model_calls == 1
    assert request.budget.max_action_steps == 1
    assert request.budget.max_total_tokens == 1
    assert request.budget.max_elapsed_seconds == 0.001
    assert request.model_budget.max_output_tokens == 1
    assert request.reserved_tokens == 1


def test_request_accepts_system_and_user_messages_with_retry_settings(
    request_factory: RequestFactory,
) -> None:
    arguments = _valid_arguments()
    history = (
        Message(MessageRole.SYSTEM, "Use the declared tools."),
        Message(MessageRole.USER, "Compute 17 + 25."),
    )
    arguments.update(
        history=history,
        max_provider_retries=1,
        max_invalid_output_retries=2,
    )

    request = request_factory(**arguments)

    assert request.history == history
    assert request.max_provider_retries == 1
    assert request.max_invalid_output_retries == 2


@pytest.mark.parametrize("reservation", [1000, 1001])
def test_reservation_at_or_above_total_limit_is_left_to_budget_tracker(
    request_factory: RequestFactory, reservation: int,
) -> None:
    arguments = _valid_arguments()
    arguments["reserved_tokens"] = reservation

    request = request_factory(**arguments)

    assert request.reserved_tokens == reservation
    assert request.budget.max_total_tokens == 1000


def test_request_accepts_integer_elapsed_seconds(
    request_factory: RequestFactory,
) -> None:
    arguments = _valid_arguments()
    arguments["budget"] = RunBudget(4, 3, 1000, 30)

    request = request_factory(**arguments)

    assert request.budget.max_elapsed_seconds == 30


def test_request_fields_cannot_be_reassigned(
    request_factory: RequestFactory,
) -> None:
    request = request_factory(**_valid_arguments())

    with pytest.raises(FrozenInstanceError):
        request.run_id = "replacement"

    assert request.run_id == "run-loop"


@pytest.mark.parametrize(
    ("value", "error"),
    [(None, TypeError), (1, TypeError), (True, TypeError),
     ("", ValueError), (" \t\n", ValueError)],
)
def test_request_rejects_invalid_run_id(
    request_factory: RequestFactory,
    value: object,
    error: type[Exception],
) -> None:
    arguments = _valid_arguments()
    arguments["run_id"] = value

    with pytest.raises(error, match="run_id"):
        request_factory(**arguments)


@pytest.mark.parametrize(
    ("history", "error", "context"),
    [
        (None, TypeError, "history"),
        ([], TypeError, "history"),
        ("task", TypeError, "history"),
        ((), ValueError, "history"),
        ((None,), TypeError, r"history\[0\]"),
        (("task",), TypeError, r"history\[0\]"),
        ((Message(MessageRole.ASSISTANT, "past answer"),), ValueError, "role"),
        ((_message_with("role", "user"),), TypeError, "role"),
        ((_message_with("content", None),), TypeError, "content"),
        ((_message_with("content", ""),), ValueError, "content"),
        ((_message_with("content", " \t"),), ValueError, "content"),
        ((Message(MessageRole.SYSTEM, "instructions only"),), ValueError, "USER"),
    ],
)
def test_request_rejects_invalid_initial_history(
    request_factory: RequestFactory,
    history: object,
    error: type[Exception],
    context: str,
) -> None:
    arguments = _valid_arguments()
    arguments["history"] = history

    with pytest.raises(error, match=context):
        request_factory(**arguments)

    assert arguments["history"] is history


@pytest.mark.parametrize("field_name", ["budget", "model_budget"])
@pytest.mark.parametrize("value", [None, {}, "budget"])
def test_request_rejects_wrong_budget_object_type(
    request_factory: RequestFactory, field_name: str, value: object,
) -> None:
    arguments = _valid_arguments()
    arguments[field_name] = value

    with pytest.raises(TypeError, match=field_name):
        request_factory(**arguments)


@pytest.mark.parametrize(
    "field_name",
    ["max_model_calls", "max_action_steps", "max_total_tokens"],
)
@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError),
     (None, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_request_revalidates_nested_run_budget_integer_limits(
    request_factory: RequestFactory,
    field_name: str,
    value: object,
    error: type[Exception],
) -> None:
    arguments = _valid_arguments()
    arguments["budget"] = _budget_with(field_name, value)

    with pytest.raises(error, match=field_name):
        request_factory(**arguments)


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), ("1", TypeError), (None, TypeError),
     (0.0, ValueError), (-0.1, ValueError), (float("nan"), ValueError),
     (float("inf"), ValueError), (float("-inf"), ValueError),
     (10 ** 400, ValueError)],
)
def test_request_rejects_non_positive_or_non_finite_elapsed_limit(
    request_factory: RequestFactory, value: object, error: type[Exception],
) -> None:
    arguments = _valid_arguments()
    arguments["budget"] = _budget_with("max_elapsed_seconds", value)

    with pytest.raises(error, match="max_elapsed_seconds"):
        request_factory(**arguments)


@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError),
     (None, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_request_revalidates_model_output_limit(
    request_factory: RequestFactory, value: object, error: type[Exception],
) -> None:
    arguments = _valid_arguments()
    arguments["model_budget"] = _model_budget_with(value)

    with pytest.raises(error, match="max_output_tokens"):
        request_factory(**arguments)


@pytest.mark.parametrize(
    "field_name",
    ["reserved_tokens", "max_provider_retries", "max_invalid_output_retries"],
)
@pytest.mark.parametrize(
    ("value", "error"),
    [(True, TypeError), (1.0, TypeError), ("1", TypeError),
     (None, TypeError), (-1, ValueError)],
)
def test_request_rejects_invalid_reservation_and_retry_values(
    request_factory: RequestFactory,
    field_name: str,
    value: object,
    error: type[Exception],
) -> None:
    arguments = _valid_arguments()
    arguments[field_name] = value

    with pytest.raises(error, match=field_name):
        request_factory(**arguments)


def test_request_rejects_zero_reservation(
    request_factory: RequestFactory,
) -> None:
    arguments = _valid_arguments()
    arguments["reserved_tokens"] = 0

    with pytest.raises(ValueError, match="reserved_tokens"):
        request_factory(**arguments)


@pytest.mark.parametrize("field_name", list(_valid_arguments()))
def test_request_rejects_missing_required_fields(
    request_factory: RequestFactory, field_name: str,
) -> None:
    arguments = _valid_arguments()
    del arguments[field_name]

    with pytest.raises(TypeError, match=field_name):
        request_factory(**arguments)


def test_request_constructor_does_not_invoke_runtime_dependencies(
    request_factory: RequestFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        calls.append("runtime")
        raise AssertionError("RunRequest must not invoke runtime dependencies")

    monkeypatch.setattr("code_agent.event_store.JsonlEventStore.append", forbidden)
    monkeypatch.setattr("code_agent.model_call.complete_model_with_retry", forbidden)
    monkeypatch.setattr("code_agent.tool_controller.ToolController.handle", forbidden)
    monkeypatch.setattr("code_agent.run_budget.BudgetTracker.__init__", forbidden)
    request = request_factory(**_valid_arguments())
    invalid = _valid_arguments()
    invalid["reserved_tokens"] = 0
    with pytest.raises(ValueError, match="reserved_tokens"):
        request_factory(**invalid)

    assert request.reserved_tokens == 128
    assert calls == []


def test_unexpected_history_error_is_not_swallowed(
    request_factory: RequestFactory,
) -> None:
    failure = RuntimeError("injected history iteration failure")

    class BrokenHistory(tuple[Message, ...]):
        def __iter__(self) -> Iterator[Message]:
            raise failure

    arguments = _valid_arguments()
    arguments["history"] = BrokenHistory(
        (Message(MessageRole.USER, "Task"),)
    )

    with pytest.raises(RuntimeError) as caught:
        request_factory(**arguments)

    assert caught.value is failure
    assert any("RunRequest" in note for note in caught.value.__notes__)
