import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _valid_record(*, sequence: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-line-ending",
        "sequence": sequence,
        "recorded_at": "2026-09-05T09:00:00+08:00",
        "event_type": "step_completed",
        "payload": {},
    }


def test_read_rejects_complete_final_record_without_lf_and_preserves_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "missing-final-lf.jsonl"
    original_bytes = json.dumps(_valid_record()).encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-line-ending")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*end with LF",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes

@pytest.mark.parametrize(
    "newline",
    [b"\n", b"\r\n"],
    ids=["lf", "crlf"],
)
def test_read_accepts_lf_and_crlf_line_endings(
    tmp_path: Path,
    newline: bytes,
) -> None:
    trace_path = tmp_path / "valid-line-ending.jsonl"
    original_bytes = json.dumps(_valid_record()).encode("utf-8") + newline
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-line-ending")

    events = store.read_all()

    assert len(events) == 1
    assert events[0].event_type == "step_completed"
    assert trace_path.read_bytes() == original_bytes

def test_append_rejects_missing_lf_before_clock_or_write(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "append-after-missing-lf.jsonl"
    original_bytes = (
        json.dumps(_valid_record()).encode("utf-8")
        + b"\n"
        + json.dumps(_valid_record(sequence=2)).encode("utf-8")
    )
    trace_path.write_bytes(original_bytes)
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 5, 9, 0, tzinfo=UTC)

    store = JsonlEventStore(
        trace_path,
        run_id="run-line-ending",
        clock=clock,
    )
    try:
        with pytest.raises(
            ValueError,
            match=r"read_all.*line 2.*end with LF",
        ):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], original_bytes)


def test_read_rejects_truncated_final_record_as_missing_lf_without_leaking_it(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "truncated-final-record.jsonl"
    original_bytes = b'{"payload":"fixture-secret-not-for-errors"'
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-line-ending")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*end with LF",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert trace_path.read_bytes() == original_bytes