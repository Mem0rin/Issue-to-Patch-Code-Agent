"""Persist one Run Event Trace as append-only JSON Lines."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ._event_payload import redact_payload, validate_payload
from ._event_schema import (
    SCHEMA_VERSION as SCHEMA_VERSION,
    StoredEvent as StoredEvent,
    _InvalidRecordedAt,
    _InvalidStoredEvent,
    decode_event_line,
    encode_event_line,
    validate_clock_recorded_at,
)


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
        if not isinstance(self.event_type, str):
            raise TypeError("event_type must be a string")
        _require_non_blank(self.event_type, "event_type")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        validate_payload(self.payload)


class JsonlEventStore:
    """Append and read one Run's versioned JSONL trace."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        clock: EventClock = _utc_now,
    ) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be a Path")

        if not isinstance(run_id, str):
            raise TypeError("run_id must be a string")
        _require_non_blank(run_id, "run_id")

        if not callable(clock):
            raise TypeError("clock must be callable")

        self._path = path
        self._run_id = run_id
        self._clock = clock

    @property
    def run_id(self) -> str:
        return self._run_id

    def append(self, draft: EventDraft) -> StoredEvent:
        if not isinstance(draft, EventDraft):
            raise TypeError("draft must be an EventDraft")
        redacted_payload = redact_payload(draft.payload)

        existing_events = self.read_all()
        try:
            recorded_at = validate_clock_recorded_at(self._clock())
        except _InvalidRecordedAt as exc:
            raise ValueError(
                f"append trace '{self._path}': {exc}"
            ) from exc
        stored = StoredEvent(
            schema_version=SCHEMA_VERSION,
            run_id=self._run_id,
            sequence=len(existing_events) + 1,
            recorded_at=recorded_at,
            event_type=draft.event_type,
            payload=redacted_payload,
        )
        event_line = encode_event_line(stored)
        try:
            with self._path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as trace_file:
                trace_file.write(event_line)
                trace_file.flush()
                os.fsync(trace_file.fileno())
        except OSError as exc:
            exc.add_note(
                f"append trace '{self._path}' failed; "
                "event durability is unknown"
            )
            raise
        return stored

    def read_all(self) -> tuple[StoredEvent, ...]:
        try:
            if not self._path.exists():
                return ()

            events: list[StoredEvent] = []
            with self._path.open("rb") as trace_file:
                for line_number, raw_line in enumerate(trace_file, start=1):
                    try:
                        event = decode_event_line(
                            raw_line,
                            expected_run_id=self._run_id,
                            expected_sequence=len(events) + 1,
                        )
                    except _InvalidStoredEvent as exc:
                        raise ValueError(
                            f"read_all trace '{self._path}' "
                            f"line {line_number}: {exc}"
                        ) from exc
                    events.append(event)
            return tuple(events)
        except OSError as exc:
            exc.add_note(f"read_all trace '{self._path}' failed")
            raise
