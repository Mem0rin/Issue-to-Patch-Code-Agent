import os
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, TextIO, cast

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


class _FailingTextFile:
    def __init__(
        self,
        *,
        stage: Literal["write", "flush"],
        error: OSError,
    ) -> None:
        self._stage = stage
        self._error = error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def write(self, value: str) -> int:
        if self._stage == "write":
            raise self._error
        return len(value)

    def flush(self) -> None:
        if self._stage == "flush":
            raise self._error

    def fileno(self) -> int:
        raise AssertionError("fsync must not run after an earlier I/O failure")


def _draft(event_type: str = "step_completed") -> EventDraft:
    return EventDraft(event_type=event_type, payload={"result": 42})


def _clock() -> datetime:
    return datetime(2026, 9, 3, 18, 0, tzinfo=UTC)


def test_append_flushes_each_complete_line_before_fsync_and_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "durable-success.jsonl"
    snapshots_at_fsync: list[bytes] = []

    def record_fsync(file_descriptor: int) -> None:
        assert file_descriptor >= 0
        snapshots_at_fsync.append(trace_path.read_bytes())

    monkeypatch.setattr(os, "fsync", record_fsync)
    store = JsonlEventStore(trace_path, run_id="run-durable", clock=_clock)

    first = store.append(_draft("run_started"))
    second = store.append(_draft("run_finished"))

    assert (first.sequence, second.sequence) == (1, 2)
    assert len(snapshots_at_fsync) == 2
    assert all(snapshot.endswith(b"\n") for snapshot in snapshots_at_fsync)
    assert [len(snapshot.splitlines()) for snapshot in snapshots_at_fsync] == [1, 2]
    assert store.read_all() == (first, second)


@pytest.mark.parametrize("stage", ["write", "flush"])
def test_append_propagates_pre_fsync_oserror_with_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: Literal["write", "flush"],
) -> None:
    trace_path = tmp_path / f"{stage}-failure.jsonl"
    failure = OSError(f"simulated {stage} failure")
    failing_file = _FailingTextFile(stage=stage, error=failure)
    fsync_calls: list[int] = []

    def fake_open(*args: object, **kwargs: object) -> TextIO:
        return cast(TextIO, failing_file)

    def record_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)

    monkeypatch.setattr(Path, "open", fake_open)
    monkeypatch.setattr(os, "fsync", record_fsync)
    store = JsonlEventStore(trace_path, run_id="run-durable", clock=_clock)

    with pytest.raises(OSError) as caught:
        store.append(_draft())

    assert caught.value is failure
    assert any(
        "append trace" in note
        and str(trace_path) in note
        and "durability is unknown" in note
        for note in caught.value.__notes__
    )
    assert fsync_calls == []
    assert not trace_path.exists()


def test_append_fsync_failure_reports_unknown_durability_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "fsync-failure.jsonl"
    failure = OSError("simulated fsync failure")
    fsync_calls: list[int] = []

    def fail_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        raise failure

    monkeypatch.setattr(os, "fsync", fail_fsync)
    store = JsonlEventStore(trace_path, run_id="run-durable", clock=_clock)

    with pytest.raises(OSError) as caught:
        store.append(_draft())

    assert caught.value is failure
    assert len(fsync_calls) == 1
    assert any(
        "append trace" in note
        and str(trace_path) in note
        and "durability is unknown" in note
        for note in caught.value.__notes__
    )
    assert trace_path.read_bytes().endswith(b"\n")
    stored_after_failure = store.read_all()
    assert len(stored_after_failure) == 1
    assert stored_after_failure[0].sequence == 1
