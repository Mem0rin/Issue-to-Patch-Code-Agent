from datetime import UTC, datetime
from pathlib import Path

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def test_read_rejects_positive_exponent_overflow_without_modifying_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "positive-exponent-overflow.jsonl"
    original_bytes = (
        b'{"schema_version":1,"run_id":"run-float-overflow",'
        b'"sequence":1,"recorded_at":"2026-09-05T10:00:00+08:00",'
        b'"event_type":"metric_recorded","payload":'
        b'{"metric":1e999,"secret":"fixture-secret-not-for-errors"}}\n'
    )
    trace_path.write_bytes(original_bytes)
    store = JsonlEventStore(trace_path, run_id="run-float-overflow")

    with pytest.raises(
        ValueError,
        match=r"read_all.*line 1.*valid JSON",
    ) as caught:
        store.read_all()

    assert str(trace_path) in str(caught.value)
    assert "fixture-secret-not-for-errors" not in str(caught.value)
    validation_error = caught.value.__cause__
    assert isinstance(validation_error, ValueError)
    assert isinstance(validation_error.__cause__, ValueError)
    assert trace_path.read_bytes() == original_bytes

@pytest.mark.parametrize(
    ("number", "expected"),
    [
        ("0.0", 0.0),
        ("-0.0", -0.0),
        ("1.25", 1.25),
        ("6.25e2", 625.0),
        ("-2.5e-2", -0.025),
        ("1.7976931348623157e308", 1.7976931348623157e308),
        ("-1.7976931348623157e308", -1.7976931348623157e308),
        ("5e-324", 5e-324),
        ("1e-999", 0.0),
        ("123456789012345678901234567890", 123456789012345678901234567890),
    ],
)
def test_read_preserves_finite_numbers_and_integer_type(
    tmp_path: Path, number: str, expected: float | int,
) -> None:
    path = tmp_path / "finite-number.jsonl"
    original = (
        '{"schema_version":1,"run_id":"run-float-overflow",'
        '"sequence":1,"recorded_at":"2026-09-05T10:00:00+08:00",'
        '"event_type":"metric_recorded","payload":{"metric":'
        + number + '}}\n'
    ).encode("utf-8")
    path.write_bytes(original)

    events = JsonlEventStore(path, run_id="run-float-overflow").read_all()

    assert len(events) == 1
    metric = events[0].payload["metric"]
    assert metric == expected
    assert type(metric) is type(expected)
    if number == "-0.0":
        assert str(metric) == "-0.0"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "payload",
    [
        '{"metric":-1e999}',
        '{"metric":1.7976931348623159e308}',
        '{"metric":-1.7976931348623159e308}',
        '{"nested":{"metric":1e999}}',
        '{"items":[0,-1e999]}',
    ],
    ids=["negative", "above-max", "below-min", "nested-object", "nested-array"],
)
def test_read_rejects_nested_or_boundary_overflow_with_second_line_context(
    tmp_path: Path, payload: str,
) -> None:
    path = tmp_path / "second-line-overflow.jsonl"
    first = (
        b'{"schema_version":1,"run_id":"run-float-overflow",'
        b'"sequence":1,"recorded_at":"2026-09-05T10:00:00+08:00",'
        b'"event_type":"metric_recorded","payload":{}}\n'
    )
    second = (
        '{"schema_version":1,"run_id":"run-float-overflow",'
        '"sequence":2,"recorded_at":"2026-09-05T10:00:00+08:00",'
        '"event_type":"metric_recorded","payload":' + payload + '}\n'
    ).encode("utf-8")
    original = first + second
    path.write_bytes(original)

    with pytest.raises(ValueError, match=r"read_all.*line 2.*valid JSON") as caught:
        JsonlEventStore(path, run_id="run-float-overflow").read_all()

    assert str(path) in str(caught.value)
    cause = caught.value.__cause__
    assert isinstance(cause, ValueError)
    assert isinstance(cause.__cause__, ValueError)
    assert str(cause.__cause__) == "JSON floating-point values must be finite"
    assert path.read_bytes() == original


def test_append_rejects_overflow_before_clock_and_preserves_trace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "append-overflow.jsonl"
    original = (
        b'{"schema_version":1,"run_id":"run-float-overflow",'
        b'"sequence":1,"recorded_at":"2026-09-05T10:00:00+08:00",'
        b'"event_type":"metric_recorded","payload":{"metric":1e999}}\n'
    )
    path.write_bytes(original)
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 5, tzinfo=UTC)

    store = JsonlEventStore(path, run_id="run-float-overflow", clock=clock)
    try:
        with pytest.raises(ValueError, match=r"read_all.*line 1.*valid JSON") as caught:
            store.append(EventDraft("run_finished", {}))
        assert str(path) in str(caught.value)
    finally:
        assert clock_calls == []
        assert path.read_bytes() == original
