import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _write_event_type_trace(
    path: Path,
    event_types: tuple[object, ...],
) -> bytes:
    records = [
        {
            "schema_version": 1,
            "run_id": "run-event-type",
            "sequence": index,
            "recorded_at": "2026-09-03T14:00:00+00:00",
            "event_type": event_type,
            "payload": {"secret": "fixture-secret-not-for-errors"},
        }
        for index, event_type in enumerate(event_types, start=1)
    ]
    encoded = "".join(json.dumps(record) + "\n" for record in records).encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return encoded


def test_read_accepts_non_blank_event_types_without_modifying_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "valid-event-type.jsonl"
    original_bytes = _write_event_type_trace(trace_path, ("run_started", "x"))
    store = JsonlEventStore(trace_path, run_id="run-event-type")

    events = store.read_all()

    assert tuple(event.event_type for event in events) == ("run_started", "x")
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("event_types", "line_number"),
    [
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" \t",), 1, id="blank-string"),
        pytest.param(("run_started", ""), 2, id="blank-second-line"),
        pytest.param((None,), 1, id="null"),
        pytest.param((1,), 1, id="integer"),
        pytest.param((True,), 1, id="bool"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(([],), 1, id="array"),
        pytest.param(({},), 1, id="object"),
    ],
)
def test_read_rejects_invalid_event_types_without_modifying_trace(
    tmp_path: Path,
    event_types: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-event-type.jsonl"
    original_bytes = _write_event_type_trace(trace_path, event_types)
    store = JsonlEventStore(trace_path, run_id="run-event-type")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*event_type",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


def test_read_rejects_missing_event_type_with_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-event-type.jsonl"
    _write_event_type_trace(trace_path, ("run_started",))
    record: dict[str, object] = json.loads(trace_path.read_text(encoding="utf-8"))
    del record["event_type"]
    original_bytes = (json.dumps(record) + "\n").encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-event-type")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*event_type is required",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "stored_event_type",
    [" ", None],
    ids=["blank", "null"],
)
def test_append_rejects_invalid_event_type_before_clock_or_write(
    tmp_path: Path,
    stored_event_type: object,
) -> None:
    trace_path = tmp_path / "append-invalid-event-type.jsonl"
    original_bytes = _write_event_type_trace(trace_path, (stored_event_type,))
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 14, 0, tzinfo=UTC)

    store = JsonlEventStore(trace_path, run_id="run-event-type", clock=clock)
    try:
        with pytest.raises(ValueError, match=r"read_all.*line 1.*event_type"):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], original_bytes)
