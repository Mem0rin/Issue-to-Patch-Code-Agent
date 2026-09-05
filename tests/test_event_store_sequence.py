import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _write_sequence_trace(
    path: Path,
    sequences: tuple[object, ...],
) -> bytes:
    records = [
        {
            "schema_version": 1,
            "run_id": "run-sequence",
            "sequence": sequence,
            "recorded_at": "2026-09-02T12:00:00+00:00",
            "event_type": "step_completed",
            "payload": {"secret": "fixture-secret-not-for-errors"},
        }
        for sequence in sequences
    ]
    content = "".join(json.dumps(record) + "\n" for record in records)
    encoded = content.encode("utf-8")
    path.write_bytes(encoded)
    return encoded


def test_read_rejects_gap_with_line_context_and_preserves_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "gap.jsonl"
    original_bytes = _write_sequence_trace(trace_path, (1, 3))
    store = JsonlEventStore(trace_path, run_id="run-sequence")

    with pytest.raises(ValueError, match=r"read_all.*line 2.*sequence") as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


def test_read_accepts_continuous_sequences_starting_at_one(tmp_path: Path) -> None:
    trace_path = tmp_path / "continuous.jsonl"
    original_bytes = _write_sequence_trace(trace_path, (1, 2, 3))
    store = JsonlEventStore(trace_path, run_id="run-sequence")

    assert tuple(event.sequence for event in store.read_all()) == (1, 2, 3)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("sequences", "line_number"),
    [
        pytest.param((0,), 1, id="zero"),
        pytest.param((-1,), 1, id="negative"),
        pytest.param((2,), 1, id="starts-at-two"),
        pytest.param((1, 1), 2, id="duplicate"),
        pytest.param((1, 2, 1), 3, id="moves-backwards"),
        pytest.param((True,), 1, id="true-is-not-a-sequence"),
        pytest.param((False,), 1, id="false-is-not-a-sequence"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(("1",), 1, id="numeric-string"),
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" ",), 1, id="blank-string"),
        pytest.param((None,), 1, id="null"),
        pytest.param(([],), 1, id="array"),
        pytest.param(({},), 1, id="object"),
    ],
)
def test_read_rejects_invalid_sequences_without_modifying_trace(
    tmp_path: Path,
    sequences: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid.jsonl"
    original_bytes = _write_sequence_trace(trace_path, sequences)
    store = JsonlEventStore(trace_path, run_id="run-sequence")

    with pytest.raises(ValueError, match=rf"read_all.*line {line_number}.*sequence") as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "fixture-secret-not-for-errors" not in str(caught.value.__cause__)
    assert trace_path.read_bytes() == original_bytes


def test_read_rejects_missing_sequence_with_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-sequence.jsonl"
    _write_sequence_trace(trace_path, (1,))
    record: dict[str, object] = json.loads(trace_path.read_text(encoding="utf-8"))
    del record["sequence"]
    original_bytes = (json.dumps(record) + "\n").encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-sequence")

    with pytest.raises(ValueError, match=r"read_all.*line 1.*sequence is required") as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize("sequences", [(1, 3), (True,)], ids=["gap", "bool"])
def test_append_rejects_invalid_trace_before_clock_or_write(
    tmp_path: Path,
    sequences: tuple[object, ...],
) -> None:
    trace_path = tmp_path / "append-invalid.jsonl"
    original_bytes = _write_sequence_trace(trace_path, sequences)
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 2, 12, 0, tzinfo=UTC)

    store = JsonlEventStore(trace_path, run_id="run-sequence", clock=clock)
    try:
        with pytest.raises(ValueError, match=r"read_all.*line .*sequence"):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert clock_calls == []
        assert trace_path.read_bytes() == original_bytes


def test_read_preserves_unexpected_io_error_with_operation_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "unreadable.jsonl"
    original_bytes = _write_sequence_trace(trace_path, (1,))
    store = JsonlEventStore(trace_path, run_id="run-sequence")
    failure = PermissionError("fixture read denied")

    def fail_open(*args: object, **kwargs: object) -> NoReturn:
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_open)
        with pytest.raises(PermissionError, match="fixture read denied") as caught:
            store.read_all()

    assert caught.value is failure
    assert any(
        "read_all" in note and str(trace_path) in note
        for note in getattr(caught.value, "__notes__", ())
    )
    assert trace_path.read_bytes() == original_bytes
