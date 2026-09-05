import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _valid_record(*, sequence: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-json-line",
        "sequence": sequence,
        "recorded_at": "2026-09-03T16:00:00+00:00",
        "event_type": "step_completed",
        "payload": {},
    }


def _encode_json_lines(values: tuple[object, ...]) -> bytes:
    return "".join(json.dumps(value) + "\n" for value in values).encode(
        "utf-8"
    )


def _encode_record_with_non_standard_constant(constant: str) -> bytes:
    record = _valid_record(sequence=2)
    record["payload"] = {
        "metric": "__NON_STANDARD_CONSTANT__",
        "secret": "fixture-secret-not-for-errors",
    }
    encoded = json.dumps(record).replace(
        '"__NON_STANDARD_CONSTANT__"',
        constant,
    )
    return f"{encoded}\n".encode("utf-8")


def test_read_accepts_json_object_with_surrounding_whitespace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "whitespace.jsonl"
    record = json.dumps(_valid_record())
    original_bytes = f" \t{record}  \n".encode("utf-8")
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-json-line")

    events = store.read_all()

    assert len(events) == 1
    assert events[0].event_type == "step_completed"
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("trace_bytes", "line_number"),
    [
        pytest.param(b"\n", 1, id="empty-line"),
        pytest.param(b" \t\n", 1, id="blank-line"),
        pytest.param(b'{"schema_version": 1\n', 1, id="malformed-json"),
        pytest.param(
            _encode_json_lines((_valid_record(),))
            + b'{"value":"fixture-secret-not-for-errors"\n',
            2,
            id="malformed-second-line",
        ),
    ],
)
def test_read_rejects_invalid_json_with_context_without_modifying_trace(
    tmp_path: Path,
    trace_bytes: bytes,
    line_number: int,
) -> None:
    trace_path = tmp_path / "invalid-json.jsonl"
    trace_path.write_bytes(trace_bytes)
    store = JsonlEventStore(trace_path, run_id="run-json-line")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*valid JSON",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    validation_error = caught.value.__cause__
    assert isinstance(validation_error, ValueError)
    assert isinstance(validation_error.__cause__, json.JSONDecodeError)
    assert trace_path.read_bytes() == trace_bytes


@pytest.mark.parametrize(
    "constant",
    ["NaN", "Infinity", "-Infinity"],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_read_rejects_non_standard_numeric_constants(
    tmp_path: Path,
    constant: str,
) -> None:
    trace_path = tmp_path / "non-standard-number.jsonl"
    original_bytes = (
        _encode_json_lines((_valid_record(),))
        + _encode_record_with_non_standard_constant(constant)
    )
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-json-line")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 2.*valid JSON",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    validation_error = caught.value.__cause__
    assert isinstance(validation_error, ValueError)
    assert isinstance(validation_error.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("values", "line_number"),
    [
        pytest.param((None,), 1, id="null"),
        pytest.param((False,), 1, id="bool"),
        pytest.param((0,), 1, id="integer"),
        pytest.param((1.0,), 1, id="float"),
        pytest.param(("",), 1, id="empty-string"),
        pytest.param((" ",), 1, id="blank-string"),
        pytest.param(([],), 1, id="empty-array"),
        pytest.param(
            (["fixture-secret-not-for-errors"],),
            1,
            id="array",
        ),
        pytest.param(
            (_valid_record(), ["fixture-secret-not-for-errors"]),
            2,
            id="non-object-second-line",
        ),
    ],
)
def test_read_rejects_non_object_json_without_modifying_trace(
    tmp_path: Path,
    values: tuple[object, ...],
    line_number: int,
) -> None:
    trace_path = tmp_path / "non-object.jsonl"
    original_bytes = _encode_json_lines(values)
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-json-line")

    with pytest.raises(
        ValueError,
        match=rf"read_all.*line {line_number}.*JSON object",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    assert isinstance(caught.value.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "trace_bytes",
    [b"{\n", b"[]\n"],
    ids=["invalid-json", "non-object"],
)
def test_append_rejects_invalid_json_line_before_clock_or_write(
    tmp_path: Path,
    trace_bytes: bytes,
) -> None:
    trace_path = tmp_path / "append-invalid-json-line.jsonl"
    trace_path.write_bytes(trace_bytes)
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 16, 0, tzinfo=UTC)

    store = JsonlEventStore(
        trace_path,
        run_id="run-json-line",
        clock=clock,
    )
    try:
        with pytest.raises(ValueError, match=r"read_all.*line 1"):
            store.append(EventDraft(event_type="run_finished", payload={}))
    finally:
        assert (clock_calls, trace_path.read_bytes()) == ([], trace_bytes)
