import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _write_recorded_at_trace(
    path: Path,
    recorded_at_values: tuple[object, ...],
) -> bytes:
    records = [
        {
            "schema_version": 1,
            "run_id": "run-recorded-at",
            "sequence": index,
            "recorded_at": recorded_at,
            "event_type": "step_completed",
            "payload": {},
        }
        for index, recorded_at in enumerate(recorded_at_values, start=1)
    ]
    encoded = "".join(json.dumps(record) + "\n" for record in records).encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return encoded


def test_read_accepts_iso_datetimes_with_explicit_offsets(tmp_path: Path) -> None:
    trace_path = tmp_path / "valid-recorded-at.jsonl"
    original_bytes = _write_recorded_at_trace(
        trace_path,
        (
            "2026-09-03T15:00:00Z",
            "2026-09-03T23:00:00+08:00",
        ),
    )
    store = JsonlEventStore(trace_path, run_id="run-recorded-at")

    events = store.read_all()

    assert tuple(event.recorded_at for event in events) == (
        datetime(2026, 9, 3, 15, 0, tzinfo=UTC),
        datetime(
            2026,
            9,
            3,
            23,
            0,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )
    assert all(event.recorded_at.utcoffset() is not None for event in events)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("recorded_at_values", "line_number"),
    [
        pytest.param((None,), 1, id="null"),
        pytest.param((True,), 1, id="bool"),
        pytest.param((0,), 1, id="integer"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" ",), 1, id="blank-string"),
        pytest.param(
            ("fixture-secret-not-for-errors",),
            1,
            id="invalid-format",
        ),
        pytest.param(
            ("2026-09-03T15:00:00",),
            1,
            id="missing-offset",
        ),
        pytest.param(
            (
                "2026-09-03T15:00:00+00:00",
                "fixture-secret-not-for-errors",
            ),
            2,
            id="invalid-second-line",
        ),
    ],
)
def test_read_rejects_invalid_recorded_at_without_modifying_trace(
    tmp_path: Path,
    recorded_at_values: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-recorded-at.jsonl"
    original_bytes = _write_recorded_at_trace(trace_path, recorded_at_values)
    store = JsonlEventStore(trace_path, run_id="run-recorded-at")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*recorded_at",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


def test_read_rejects_missing_recorded_at_with_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-recorded-at.jsonl"
    _write_recorded_at_trace(
        trace_path,
        ("2026-09-03T15:00:00+00:00",),
    )
    record: dict[str, object] = json.loads(trace_path.read_text(encoding="utf-8"))
    del record["recorded_at"]
    original_bytes = (json.dumps(record) + "\n").encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-recorded-at")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*recorded_at is required",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


def test_append_rejects_invalid_stored_recorded_at_before_clock_or_write(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "append-invalid-recorded-at.jsonl"
    original_bytes = _write_recorded_at_trace(trace_path, ("not-a-datetime",))
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 15, 0, tzinfo=UTC)

    store = JsonlEventStore(
        trace_path,
        run_id="run-recorded-at",
        clock=clock,
    )
    try:
        with pytest.raises(
            ValueError,
            match=r"read_all.*line 1.*recorded_at",
        ):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], original_bytes)


@pytest.mark.parametrize(
    "clock_value",
    [
        "2026-09-03T15:00:00+00:00",
        datetime(2026, 9, 3, 15, 0),
    ],
    ids=["wrong-type", "missing-offset"],
)
def test_append_rejects_invalid_clock_value_before_write(
    tmp_path: Path,
    clock_value: object,
) -> None:
    trace_path = tmp_path / "invalid-clock-recorded-at.jsonl"
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return cast(datetime, clock_value)

    store = JsonlEventStore(
        trace_path,
        run_id="run-recorded-at",
        clock=clock,
    )

    with pytest.raises(
        ValueError,
        match=r"append.*recorded_at",
    ) as caught:
        store.append(EventDraft(event_type="run_started", payload={}))

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert clock_calls == ["called"]
    assert not trace_path.exists()
