import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import cast


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredEvent:
    """One persisted, ordered Run Event."""

    schema_version: int
    run_id: str
    sequence: int
    recorded_at: datetime
    event_type: str
    payload: Mapping[str, object]


class _InvalidStoredEvent(ValueError):
    pass


class _InvalidSchemaVersion(_InvalidStoredEvent):
    """An expected schema version error without raw trace data."""


class _InvalidRunId(_InvalidStoredEvent):
    """An expected run id error without raw trace data."""


class _InvalidEventType(_InvalidStoredEvent):
    """An expected event type error without raw trace data."""


class _InvalidPayload(_InvalidStoredEvent):
    """An expected payload error without raw trace data."""


class _InvalidRecordedAt(_InvalidStoredEvent):
    pass


class _InvalidSequence(_InvalidStoredEvent):
    """An expected sequence validation error without raw trace data."""


def validate_clock_recorded_at(recorded_at: object) -> datetime:
    if not isinstance(recorded_at, datetime):
        raise _InvalidRecordedAt(
            "recorded_at returned by clock must be a datetime"
        )

    if not _recorded_at_has_utc_offset(recorded_at):
        raise _InvalidRecordedAt(
            "recorded_at returned by clock must include a UTC offset"
        )

    return recorded_at


def encode_event_line(event: StoredEvent) -> str:
    encoded = json.dumps(
        _encode_event(event),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{encoded}\n"


def _require_lf_terminated(raw_line: bytes) -> None:
    if not raw_line.endswith(b"\n"):
        raise _InvalidStoredEvent(
            "stored event line must end with LF"
        )


def decode_event_line(
    raw_line: bytes,
    *,
    expected_run_id: str,
    expected_sequence: int,
) -> StoredEvent:
    _require_lf_terminated(raw_line)
    encoded = _decode_stored_event(raw_line)
    schema_version = _validate_stored_schema_version(encoded)
    run_id = _validate_stored_run_id(
        encoded,
        expected_run_id=expected_run_id,
    )
    event_type = _validate_stored_event_type(encoded)
    payload = _validate_stored_payload(encoded)
    recorded_at = _validate_stored_recorded_at(encoded)
    sequence = _validate_stored_sequence(
        encoded,
        expected_sequence=expected_sequence,
    )
    return _decode_event(
        schema_version=schema_version,
        run_id=run_id,
        sequence=sequence,
        recorded_at=recorded_at,
        event_type=event_type,
        payload=payload,
    )


def _reject_non_standard_json_constant(
    _constant: str,
) -> object:
    raise ValueError(
        "non-standard JSON numeric constants are not allowed"
    )


def _key_has_appeared(
    decoded: dict[str, object],
    key: str,
) -> bool:
    return key in decoded


def _object_from_unique_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    decoded: dict[str, object] = {}

    for key, value in pairs:
        if _key_has_appeared(decoded, key):
            raise ValueError(
                "duplicate JSON object keys are not allowed"
            )
        decoded[key] = value

    return decoded


def _decode_finite_json_float(encoded: str) -> float:
    decoded = float(encoded)
    if not isfinite(decoded):
        raise ValueError(
            "JSON floating-point values must be finite"
        )
    return decoded


def _decode_stored_event(raw_line: bytes) -> dict[str, object]:
    try:
        line = raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _InvalidStoredEvent(
            "stored event must be UTF-8"
        ) from exc

    try:
        decoded: object = json.loads(
            line,
            parse_constant=_reject_non_standard_json_constant,
            parse_float=_decode_finite_json_float,
            object_pairs_hook=_object_from_unique_pairs,
        )
    except ValueError as exc:
        raise _InvalidStoredEvent(
            "stored event must be valid JSON"
        ) from exc

    if not isinstance(decoded, dict):
        raise _InvalidStoredEvent(
            "stored event must be a JSON object"
        )

    return cast(dict[str, object], decoded)


def _validate_stored_schema_version(
    encoded: Mapping[str, object],
) -> int:
    if "schema_version" not in encoded:
        raise _InvalidSchemaVersion("schema_version is required")

    schema_version = encoded["schema_version"]
    if type(schema_version) is not int:
        raise _InvalidSchemaVersion(
            "schema_version must be an integer (bool is excluded)"
        )

    if not _schema_version_is_supported(schema_version):
        raise _InvalidSchemaVersion(
            f"schema_version is unsupported; expected {SCHEMA_VERSION}"
        )

    return schema_version


def _schema_version_is_supported(schema_version: int) -> bool:
    """schema_version is an int; decide whether this reader supports it."""
    return schema_version == SCHEMA_VERSION


def _validate_stored_run_id(
    encoded: Mapping[str, object],
    *,
    expected_run_id: str,
) -> str:
    if "run_id" not in encoded:
        raise _InvalidRunId("run_id is required")

    run_id = encoded["run_id"]
    if not isinstance(run_id, str):
        raise _InvalidRunId("run_id must be a string")

    if not run_id.strip():
        raise _InvalidRunId("run_id must not be blank")

    if not _run_id_matches(run_id, expected_run_id):
        raise _InvalidRunId("run_id must match the configured run_id")

    return run_id


def _run_id_matches(run_id: str, expected_run_id: str) -> bool:
    return run_id == expected_run_id


def _validate_stored_event_type(
    encoded: Mapping[str, object],
) -> str:
    if "event_type" not in encoded:
        raise _InvalidEventType("event_type is required")

    event_type = encoded["event_type"]
    if not isinstance(event_type, str):
        raise _InvalidEventType("event_type must be a string")

    if not _event_type_is_non_blank(event_type):
        raise _InvalidEventType("event_type must not be blank")

    return event_type


def _event_type_is_non_blank(event_type: str) -> bool:
    return bool(event_type.strip())


def _validate_stored_payload(
    encoded: Mapping[str, object],
) -> dict[str, object]:
    if "payload" not in encoded:
        raise _InvalidPayload("payload is required")

    payload = encoded["payload"]
    if not _payload_is_object(payload):
        raise _InvalidPayload("payload must be an object")

    return cast(dict[str, object], payload)


def _payload_is_object(payload: object) -> bool:
    return isinstance(payload, dict)


def _validate_stored_recorded_at(
    encoded: Mapping[str, object],
) -> datetime:
    if "recorded_at" not in encoded:
        raise _InvalidRecordedAt("recorded_at is required")

    encoded_recorded_at = encoded["recorded_at"]
    if not isinstance(encoded_recorded_at, str):
        raise _InvalidRecordedAt("recorded_at must be a string")
    if not encoded_recorded_at.strip():
        raise _InvalidRecordedAt("recorded_at must not be blank")

    try:
        recorded_at = datetime.fromisoformat(encoded_recorded_at)
    except ValueError as exc:
        raise _InvalidRecordedAt(
            "recorded_at must be an ISO 8601 datetime"
        ) from exc

    if not _recorded_at_has_utc_offset(recorded_at):
        raise _InvalidRecordedAt(
            "recorded_at must include a UTC offset"
        )

    return recorded_at


def _recorded_at_has_utc_offset(recorded_at: datetime) -> bool:
    return recorded_at.utcoffset() is not None


def _validate_stored_sequence(
    encoded: Mapping[str, object],
    *,
    expected_sequence: int,
) -> int:
    if "sequence" not in encoded:
        raise _InvalidSequence("sequence is required")
    sequence = encoded["sequence"]
    if type(sequence) is not int:
        raise _InvalidSequence("sequence must be an integer (bool is excluded)")
    if not _sequence_is_expected(sequence, expected_sequence):
        raise _InvalidSequence(
            f"sequence must equal expected sequence {expected_sequence}"
        )
    return sequence


def _sequence_is_expected(sequence: int, expected_sequence: int) -> bool:
    """Both inputs are ints; expected_sequence >= 1 comes from accepted events."""
    return sequence == expected_sequence


def _encode_event(event: StoredEvent) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "recorded_at": event.recorded_at.isoformat(),
        "event_type": event.event_type,
        "payload": event.payload,
    }


def _decode_event(
    *,
    schema_version: int,
    run_id: str,
    sequence: int,
    recorded_at: datetime,
    event_type: str,
    payload: dict[str, object],
) -> StoredEvent:
    return StoredEvent(
        schema_version=schema_version,
        run_id=run_id,
        sequence=sequence,
        recorded_at=recorded_at,
        event_type=event_type,
        payload=payload,
    )
