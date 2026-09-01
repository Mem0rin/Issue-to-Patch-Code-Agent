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


def test_append_redacts_nested_sensitive_fields_before_storage(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run-002.jsonl"
    recorded_at = datetime(
        2026,
        9,
        1,
        13,
        0,
        tzinfo=UTC,
    )
    store = JsonlEventStore(
        trace_path,
        run_id="run-002",
        clock=lambda: recorded_at,
    )

    stored = store.append(
        EventDraft(
            event_type="model_call_completed",
            payload={
                "provider": {
                    "api_key": "sk-must-not-reach-disk",
                },
                "headers": {
                    "Authorization": "Bearer must-not-reach-disk",
                },
                "attempts": [
                    {"access_token": "must-not-reach-disk"},
                ],
                "usage": {
                    "input_tokens": 316,
                    "output_tokens": 58,
                },
            },
        )
    )

    expected_payload = {
        "provider": {"api_key": "[REDACTED]"},
        "headers": {"Authorization": "[REDACTED]"},
        "attempts": [{"access_token": "[REDACTED]"}],
        "usage": {
            "input_tokens": 316,
            "output_tokens": 58,
        },
    }
    assert stored.payload == expected_payload

    reopened = JsonlEventStore(
        trace_path,
        run_id="run-002",
        clock=lambda: recorded_at,
    )
    assert reopened.read_all()[0].payload == expected_payload
