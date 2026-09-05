import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _write_payload_trace(
    path: Path,
    payloads: tuple[object, ...],
) -> bytes:
    records = [
        {
            "schema_version": 1,
            "run_id": "run-payload",
            "sequence": index,
            "recorded_at": "2026-09-03T15:00:00+00:00",
            "event_type": "step_completed",
            "payload": payload,
        }
        for index, payload in enumerate(payloads, start=1)
    ]
    encoded = "".join(json.dumps(record) + "\n" for record in records).encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return encoded


def test_read_accepts_object_payloads_including_empty_object(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "valid-payload.jsonl"
    original_bytes = _write_payload_trace(
        trace_path,
        ({}, {"nested": {"value": 42}, "items": [1, 2]}),
    )
    store = JsonlEventStore(trace_path, run_id="run-payload")

    events = store.read_all()

    assert tuple(event.payload for event in events) == (
        {},
        {"nested": {"value": 42}, "items": [1, 2]},
    )
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("payloads", "line_number"),
    [
        pytest.param((None,), 1, id="null"),
        pytest.param((True,), 1, id="bool"),
        pytest.param((0,), 1, id="integer"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" ",), 1, id="blank-string"),
        pytest.param(([],), 1, id="empty-array"),
        pytest.param((["fixture-secret-not-for-errors"],), 1, id="array"),
        pytest.param(({}, None), 2, id="invalid-second-line"),
    ],
)
def test_read_rejects_non_object_payloads_without_modifying_trace(
    tmp_path: Path,
    payloads: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-payload.jsonl"
    original_bytes = _write_payload_trace(trace_path, payloads)
    store = JsonlEventStore(trace_path, run_id="run-payload")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*payload",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


def test_read_rejects_missing_payload_with_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-payload.jsonl"
    _write_payload_trace(trace_path, ({},))
    record: dict[str, object] = json.loads(trace_path.read_text(encoding="utf-8"))
    del record["payload"]
    original_bytes = (json.dumps(record) + "\n").encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-payload")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*payload is required",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "stored_payload",
    [None, []],
    ids=["null", "array"],
)
def test_append_rejects_non_object_payload_before_clock_or_write(
    tmp_path: Path,
    stored_payload: object,
) -> None:
    trace_path = tmp_path / "append-invalid-payload.jsonl"
    original_bytes = _write_payload_trace(trace_path, (stored_payload,))
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 15, 0, tzinfo=UTC)

    store = JsonlEventStore(trace_path, run_id="run-payload", clock=clock)
    try:
        with pytest.raises(ValueError, match=r"read_all.*line 1.*payload"):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], original_bytes)
