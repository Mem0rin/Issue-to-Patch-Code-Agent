from datetime import UTC, datetime
from pathlib import Path

from code_agent.event_store import (
    EventDraft,
    JsonlEventStore,
    StoredEvent,
)


def test_append_persists_one_versioned_event(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run-001.jsonl"
    recorded_at = datetime(
        2026,
        9,
        1,
        12,
        30,
        tzinfo=UTC,
    )
    store = JsonlEventStore(
        trace_path,
        run_id="run-001",
        clock=lambda: recorded_at,
    )

    stored = store.append(
        EventDraft(
            event_type="run_started",
            payload={"model": "deepseek-v4-pro"},
        )
    )

    expected = StoredEvent(
        schema_version=1,
        run_id="run-001",
        sequence=1,
        recorded_at=recorded_at,
        event_type="run_started",
        payload={"model": "deepseek-v4-pro"},
    )
    assert stored == expected

    reopened = JsonlEventStore(
        trace_path,
        run_id="run-001",
        clock=lambda: recorded_at,
    )
    assert reopened.read_all() == (expected,)
