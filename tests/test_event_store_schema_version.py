import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _write_version_trace(
    path: Path,
    versions: tuple[object, ...],
) -> bytes:
    records = [
        {
            "schema_version": version,
            "run_id": "run-version",
            "sequence": index,
            "recorded_at": "2026-09-03T12:00:00+00:00",
            "event_type": "step_completed",
            "payload": {"secret": "fixture-secret-not-for-errors"},
        }
        for index, version in enumerate(versions, start=1)
    ]
    encoded = "".join(json.dumps(record) + "\n" for record in records).encode(
        "utf-8"
    )
    path.write_bytes(encoded)
    return encoded


def test_read_accepts_current_schema_version_without_modifying_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "supported-version.jsonl"
    original_bytes = _write_version_trace(trace_path, (1, 1))
    store = JsonlEventStore(trace_path, run_id="run-version")

    events = store.read_all()

    assert tuple(event.schema_version for event in events) == (1, 1)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("versions", "line_number"),
    [
        pytest.param((0,), 1, id="zero"),
        pytest.param((-1,), 1, id="negative"),
        pytest.param((2,), 1, id="next-version"),
        pytest.param((1, 2), 2, id="unsupported-second-line"),
        pytest.param((True,), 1, id="true-is-not-a-version"),
        pytest.param((False,), 1, id="false-is-not-a-version"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(("1",), 1, id="numeric-string"),
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" ",), 1, id="blank-string"),
        pytest.param((None,), 1, id="null"),
        pytest.param(([],), 1, id="array"),
        pytest.param(({},), 1, id="object"),
    ],
)
def test_read_rejects_invalid_schema_versions_without_modifying_trace(
    tmp_path: Path,
    versions: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-version.jsonl"
    original_bytes = _write_version_trace(trace_path, versions)
    store = JsonlEventStore(trace_path, run_id="run-version")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*schema_version",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert "fixture-secret-not-for-errors" not in str(caught.value.__cause__)
    assert trace_path.read_bytes() == original_bytes


def test_read_rejects_missing_schema_version_with_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing-version.jsonl"
    _write_version_trace(trace_path, (1,))
    record: dict[str, object] = json.loads(trace_path.read_text(encoding="utf-8"))
    del record["schema_version"]
    original_bytes = (json.dumps(record) + "\n").encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-version")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*schema_version is required",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "versions",
    [(2,), (True,)],
    ids=["unsupported-version", "bool"],
)
def test_append_rejects_invalid_version_before_clock_or_write(
    tmp_path: Path,
    versions: tuple[object, ...],
) -> None:
    trace_path = tmp_path / "append-invalid-version.jsonl"
    original_bytes = _write_version_trace(trace_path, versions)
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 12, 0, tzinfo=UTC)

    store = JsonlEventStore(trace_path, run_id="run-version", clock=clock)
    try:
        with pytest.raises(ValueError, match=r"read_all.*line 1.*schema_version"):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert clock_calls == []
        assert trace_path.read_bytes() == original_bytes
