import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _write_run_id_trace(
    path: Path,
    run_ids: tuple[object, ...],
) -> bytes:
    records = [
        {
            "schema_version": 1,
            "run_id": run_id,
            "sequence": index,
            "recorded_at": "2026-09-03T13:00:00+00:00",
            "event_type": "step_completed",
            "payload": {"secret": "fixture-secret-not-for-errors"},
        }
        for index, run_id in enumerate(run_ids, start=1)
    ]
    encoded = "".join(json.dumps(record) + "\n" for record in records).encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return encoded


def test_read_accepts_matching_run_ids_without_modifying_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "matching-run-id.jsonl"
    original_bytes = _write_run_id_trace(trace_path, ("run-expected",) * 2)
    store = JsonlEventStore(trace_path, run_id="run-expected")

    events = store.read_all()

    assert tuple(event.run_id for event in events) == (
        "run-expected",
        "run-expected",
    )
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("run_ids", "line_number"),
    [
        pytest.param(("run-other",), 1, id="different-run"),
        pytest.param(("run-expected", "run-other"), 2, id="different-second-line"),
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" ",), 1, id="blank-string"),
        pytest.param((None,), 1, id="null"),
        pytest.param((1,), 1, id="integer"),
        pytest.param((True,), 1, id="bool"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(([],), 1, id="array"),
        pytest.param(({},), 1, id="object"),
    ],
)
def test_read_rejects_invalid_run_ids_without_modifying_trace(
    tmp_path: Path,
    run_ids: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-run-id.jsonl"
    original_bytes = _write_run_id_trace(trace_path, run_ids)
    store = JsonlEventStore(trace_path, run_id="run-expected")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*run_id",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "fixture-secret-not-for-errors" not in str(caught.value.__cause__)
    assert trace_path.read_bytes() == original_bytes


def test_read_rejects_missing_run_id_with_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-run-id.jsonl"
    _write_run_id_trace(trace_path, ("run-expected",))
    record: dict[str, object] = json.loads(trace_path.read_text(encoding="utf-8"))
    del record["run_id"]
    original_bytes = (json.dumps(record) + "\n").encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-expected")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*run_id is required",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "stored_run_id",
    ["run-other", " "],
    ids=["different-run", "blank"],
)
def test_append_rejects_invalid_run_id_before_clock_or_write(
    tmp_path: Path,
    stored_run_id: str,
) -> None:
    trace_path = tmp_path / "append-invalid-run-id.jsonl"
    original_bytes = _write_run_id_trace(trace_path, (stored_run_id,))
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 13, 0, tzinfo=UTC)

    store = JsonlEventStore(trace_path, run_id="run-expected", clock=clock)
    try:
        with pytest.raises(ValueError, match=r"read_all.*line 1.*run_id"):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], original_bytes)
