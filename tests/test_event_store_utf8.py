import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _valid_utf8_line(*, newline: bytes = b"\n") -> bytes:
    record = {
        "schema_version": 1,
        "run_id": "run-utf8",
        "sequence": 1,
        "recorded_at": "2026-09-03T17:00:00+00:00",
        "event_type": "步骤完成",
        "payload": {"说明": "已完成 ✅"},
    }
    return json.dumps(record, ensure_ascii=False).encode("utf-8") + newline


def test_read_accepts_utf8_content_and_crlf_line_ending(tmp_path: Path) -> None:
    trace_path = tmp_path / "valid-utf8.jsonl"
    original_bytes = _valid_utf8_line(newline=b"\r\n")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-utf8")

    events = store.read_all()

    assert len(events) == 1
    assert events[0].event_type == "步骤完成"
    assert events[0].payload == {"说明": "已完成 ✅"}
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("trace_bytes", "line_number"),
    [
        pytest.param(b"\xff\n", 1, id="invalid-leading-byte"),
        pytest.param(
            b'{"value":"truncated-\xe4"}\n',
            1,
            id="truncated-multibyte-sequence",
        ),
        pytest.param(
            _valid_utf8_line()
            + b'{"value":"fixture-secret-not-for-errors-\xff"}\n',
            2,
            id="invalid-second-line",
        ),
    ],
)
def test_read_rejects_invalid_utf8_with_physical_line_context(
    tmp_path: Path,
    trace_bytes: bytes,
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-utf8.jsonl"
    trace_path.write_bytes(trace_bytes)
    store = JsonlEventStore(trace_path, run_id="run-utf8")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*UTF-8",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    validation_error = caught.value.__cause__
    assert isinstance(validation_error, ValueError)
    assert isinstance(validation_error.__cause__, UnicodeDecodeError)
    assert trace_path.read_bytes() == trace_bytes


def test_append_rejects_invalid_utf8_before_clock_or_write(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "append-invalid-utf8.jsonl"
    original_bytes = b'{"value":"\xff"}\n'
    trace_path.write_bytes(original_bytes)
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 17, 0, tzinfo=UTC)

    store = JsonlEventStore(trace_path, run_id="run-utf8", clock=clock)
    try:
        with pytest.raises(
            ValueError,
            match=r"read_all.*line 1.*UTF-8",
        ):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], original_bytes)
