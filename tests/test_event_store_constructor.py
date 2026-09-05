from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from code_agent.event_store import EventClock, JsonlEventStore


class _CallableClock:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return datetime(2026, 9, 3, 19, 0, tzinfo=UTC)


def test_constructor_accepts_path_run_id_and_callable_clock(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "constructor.jsonl"
    clock = _CallableClock()

    store = JsonlEventStore(trace_path, run_id="run-001", clock=clock)

    assert store.read_all() == ()
    assert clock.calls == 0
    assert not trace_path.exists()


def test_constructor_accepts_one_character_run_id(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "one-character-run.jsonl"

    store = JsonlEventStore(trace_path, run_id="x")

    assert store.read_all() == ()
    assert not trace_path.exists()


@pytest.mark.parametrize(
    "invalid_path",
    [None, "", "trace.jsonl", b"trace.jsonl", True, 1, object()],
    ids=["none", "empty-string", "string", "bytes", "bool", "int", "object"],
)
def test_constructor_rejects_non_path_values(
    tmp_path: Path,
    invalid_path: object,
) -> None:
    untouched_path = tmp_path / "untouched.jsonl"

    with pytest.raises(TypeError, match=r"path.*Path"):
        JsonlEventStore(
            cast(Path, invalid_path),
            run_id="run-valid",
        )

    assert not untouched_path.exists()


@pytest.mark.parametrize(
    "invalid_run_id",
    [None, True, 1, 1.0, b"run", [], {}],
    ids=["none", "bool", "int", "float", "bytes", "list", "mapping"],
)
def test_constructor_rejects_non_string_run_id(
    tmp_path: Path,
    invalid_run_id: object,
) -> None:
    trace_path = tmp_path / "invalid-run-id.jsonl"

    with pytest.raises(TypeError, match=r"run_id.*string"):
        JsonlEventStore(
            trace_path,
            run_id=cast(str, invalid_run_id),
        )

    assert not trace_path.exists()


@pytest.mark.parametrize(
    "invalid_run_id",
    ["", " ", "\t\r\n"],
    ids=["empty", "space", "whitespace"],
)
def test_constructor_rejects_blank_run_id(
    tmp_path: Path,
    invalid_run_id: str,
) -> None:
    trace_path = tmp_path / "blank-run-id.jsonl"

    with pytest.raises(ValueError, match=r"run_id.*blank"):
        JsonlEventStore(trace_path, run_id=invalid_run_id)

    assert not trace_path.exists()


@pytest.mark.parametrize(
    "invalid_clock",
    [None, True, 1, 1.0, "clock", object()],
    ids=["none", "bool", "int", "float", "string", "object"],
)
def test_constructor_rejects_non_callable_clock(
    tmp_path: Path,
    invalid_clock: object,
) -> None:
    trace_path = tmp_path / "invalid-clock.jsonl"

    with pytest.raises(TypeError, match=r"clock.*callable"):
        JsonlEventStore(
            trace_path,
            run_id="run-valid",
            clock=cast(EventClock, invalid_clock),
        )

    assert not trace_path.exists()


def test_constructor_does_not_call_valid_function_clock(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "clock-not-called.jsonl"
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 19, 0, tzinfo=UTC)

    typed_clock: Callable[[], datetime] = clock
    JsonlEventStore(
        trace_path,
        run_id="run-valid",
        clock=typed_clock,
    )

    assert clock_calls == []
    assert not trace_path.exists()
