"""Persist one Run Event Trace as append-only JSON Lines."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


SCHEMA_VERSION = 1

EventClock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _require_non_blank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


@dataclass(frozen=True)
class EventDraft:
    """A Run Event before storage metadata is assigned."""

    event_type: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_non_blank(self.event_type, "event_type")


@dataclass(frozen=True)
class StoredEvent:
    """One persisted, ordered Run Event."""

    schema_version: int
    run_id: str
    sequence: int
    recorded_at: datetime
    event_type: str
    payload: Mapping[str, object]


class JsonlEventStore:
    """Append and read one Run's versioned JSONL trace."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        clock: EventClock = _utc_now,
    ) -> None:
        _require_non_blank(run_id, "run_id")
        self._path = path
        self._run_id = run_id
        self._clock = clock

    def append(self, draft: EventDraft) -> StoredEvent:
        existing_events = self.read_all()
        stored = StoredEvent(
            schema_version=SCHEMA_VERSION,
            run_id=self._run_id,
            sequence=len(existing_events) + 1,
            recorded_at=self._clock(),
            event_type=draft.event_type,
            payload=dict(draft.payload),
        )
        encoded = json.dumps(
            _encode_event(stored),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._path.open(
            "a",
            encoding="utf-8",
            newline="\n",
        ) as trace_file:
            trace_file.write(encoded)
            trace_file.write("\n")
        return stored

    def read_all(self) -> tuple[StoredEvent, ...]:
        if not self._path.exists():
            return ()

        events: list[StoredEvent] = []
        with self._path.open("r", encoding="utf-8") as trace_file:
            for line in trace_file:
                decoded: object = json.loads(line)
                if not isinstance(decoded, dict):
                    raise ValueError("stored event must be a JSON object")
                events.append(
                    _decode_event(cast(dict[str, object], decoded))
                )
        return tuple(events)


def _encode_event(event: StoredEvent) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "recorded_at": event.recorded_at.isoformat(),
        "event_type": event.event_type,
        "payload": event.payload,
    }


def _decode_event(encoded: Mapping[str, object]) -> StoredEvent:
    schema_version = encoded["schema_version"]
    run_id = encoded["run_id"]
    sequence = encoded["sequence"]
    recorded_at = encoded["recorded_at"]
    event_type = encoded["event_type"]
    payload = encoded["payload"]

    if not isinstance(schema_version, int):
        raise ValueError("schema_version must be an integer")
    if not isinstance(run_id, str):
        raise ValueError("run_id must be a string")
    if not isinstance(sequence, int):
        raise ValueError("sequence must be an integer")
    if not isinstance(recorded_at, str):
        raise ValueError("recorded_at must be a string")
    if not isinstance(event_type, str):
        raise ValueError("event_type must be a string")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    return StoredEvent(
        schema_version=schema_version,
        run_id=run_id,
        sequence=sequence,
        recorded_at=datetime.fromisoformat(recorded_at),
        event_type=event_type,
        payload=cast(dict[str, object], payload),
    )
