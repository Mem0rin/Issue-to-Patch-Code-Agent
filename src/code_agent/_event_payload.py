import math
from collections.abc import Mapping


_REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_FIELD_NAMES = frozenset({
    "access_token",
    "api_key",
    "authorization",
    "password",
    "refresh_token",
    "secret",
    "token",
})


def validate_payload(payload: Mapping[str, object]) -> None:
    _validate_payload_value(
        payload,
        active_container_ids=set(),
    )


def _validate_payload_value(
    value: object,
    *,
    active_container_ids: set[int],
) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload floats must be finite")
        return

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError("payload must not contain cycles")

        active_container_ids.add(container_id)
        try:
            for field_name, nested_value in value.items():
                if not isinstance(field_name, str):
                    raise TypeError(
                        "payload object keys must be strings"
                    )
                _validate_payload_value(
                    nested_value,
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return

    if isinstance(value, list):
        container_id = id(value)
        if container_id in active_container_ids:
            raise ValueError("payload must not contain cycles")

        active_container_ids.add(container_id)
        try:
            for item in value:
                _validate_payload_value(
                    item,
                    active_container_ids=active_container_ids,
                )
        finally:
            active_container_ids.remove(container_id)
        return

    raise TypeError("payload values must be JSON-compatible")


def redact_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    validate_payload(payload)
    return {
        field_name: _redact_value(
            value,
            field_name=field_name,
        )
        for field_name, value in payload.items()
    }


def _redact_value(
    value: object,
    *,
    field_name: str | None = None,
) -> object:
    if (
        field_name is not None
        and field_name.casefold() in _SENSITIVE_FIELD_NAMES
    ):
        return _REDACTED_VALUE

    if isinstance(value, Mapping):
        return {
            name: _redact_value(
                nested_value,
                field_name=name,
            )
            for name, nested_value in value.items()
        }

    if isinstance(value, list):
        return [
            _redact_value(item)
            for item in value
        ]

    return value
