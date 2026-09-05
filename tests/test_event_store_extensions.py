"""Compatibility of outer extension fields at the public EventStore boundary."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _record(extension: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-extensions",
        "sequence": 1,
        "recorded_at": "2026-09-05T00:00:00+00:00",
        "event_type": "step_completed",
        "payload": {"answer": 42},
        "extension": extension,
    }


@pytest.mark.parametrize(
    "extension",
    [None, {}, [], {"sequence": 99, "payload": {"answer": 0}, "items": [True, "x", 1.25]}],
    ids=["null", "empty-object", "empty-array", "nested-known-names"],
)
def test_read_accepts_extensions_but_returns_only_known_event_fields(
    tmp_path: Path, extension: object,
) -> None:
    path = tmp_path / "extensions.jsonl"
    original = (json.dumps(_record(extension)) + "\n").encode("utf-8")
    path.write_bytes(original)

    events = JsonlEventStore(path, run_id="run-extensions").read_all()

    assert len(events) == 1
    assert asdict(events[0]) == {
        "schema_version": 1,
        "run_id": "run-extensions",
        "sequence": 1,
        "recorded_at": datetime(2026, 9, 5, tzinfo=UTC),
        "event_type": "step_completed",
        "payload": {"answer": 42},
    }
    assert path.read_bytes() == original


def test_append_preserves_existing_extension_bytes_and_continues_sequence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "append-extensions.jsonl"
    original = (json.dumps(_record({"source": "fixture"}), indent=None) + "\r\n").encode("utf-8")
    path.write_bytes(original)
    store = JsonlEventStore(
        path, run_id="run-extensions",
        clock=lambda: datetime(2026, 9, 5, tzinfo=UTC),
    )

    stored = store.append(EventDraft("run_finished", {"status": "completed"}))

    assert stored.sequence == 2
    assert stored.event_type == "run_finished"
    assert stored.payload == {"status": "completed"}
    assert path.read_bytes().startswith(original)
    reopened = JsonlEventStore(path, run_id="run-extensions").read_all()
    assert [event.sequence for event in reopened] == [1, 2]
    assert reopened[0].payload == {"answer": 42}
    assert reopened[1] == stored


def test_append_rejects_overflow_in_unused_extension_before_clock_or_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-extension.jsonl"
    original = (
        json.dumps(_record({"metric": "OVERFLOW"})).replace('"OVERFLOW"', "1e999")
        + "\n"
    ).encode("utf-8")
    path.write_bytes(original)
    clock = Mock(return_value=datetime(2026, 9, 5, tzinfo=UTC))
    store = JsonlEventStore(path, run_id="run-extensions", clock=clock)

    with pytest.raises(ValueError, match=r"read_all.*line 1.*valid JSON") as caught:
        store.append(EventDraft("run_finished", {}))

    assert str(path) in str(caught.value)
    cause = caught.value.__cause__
    assert isinstance(cause, ValueError)
    assert isinstance(cause.__cause__, ValueError)
    assert str(cause.__cause__) == "JSON floating-point values must be finite"
    clock.assert_not_called()
    assert path.read_bytes() == original


def test_extension_payload_does_not_replace_missing_required_outer_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-payload.jsonl"
    record = _record({"payload": {"answer": 42}})
    del record["payload"]
    original = (json.dumps(record) + "\n").encode("utf-8")
    path.write_bytes(original)

    with pytest.raises(ValueError, match=r"read_all.*line 1.*payload is required") as caught:
        JsonlEventStore(path, run_id="run-extensions").read_all()

    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert path.read_bytes() == original
