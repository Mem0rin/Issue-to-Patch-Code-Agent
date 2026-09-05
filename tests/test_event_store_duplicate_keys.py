"""Reject duplicate JSON keys through the public EventStore interface."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def _line(payload_json: str = "{}", *, sequence: int = 1) -> bytes:
    prefix = (
        '{"schema_version":1,"run_id":"run-duplicates",'
        f'"sequence":{sequence},'
        '"recorded_at":"2026-09-03T00:00:00+00:00",'
        '"event_type":"step_completed","payload":'
    )
    return (prefix + payload_json + "}\n").encode("utf-8")


@pytest.mark.parametrize(
    "payload_json",
    [
        "{}",
        '{"x":1}',
        '{"left":{"x":1},"right":{"x":2}}',
        '{"items":[{"x":1},{"x":2}]}',
        '{"x":1,"X":2," x ":3}',
        '{"":1}',
        '{"é":1,"e\\u0301":2}',
    ],
    ids=["empty", "single", "siblings", "array-siblings", "exact-keys", "empty-key", "no-unicode-normalization"],
)
def test_read_preserves_unique_keys_and_object_scopes(
    tmp_path: Path, payload_json: str,
) -> None:
    path = tmp_path / "unique.jsonl"
    original = _line(payload_json)
    path.write_bytes(original)

    events = JsonlEventStore(path, run_id="run-duplicates").read_all()

    assert len(events) == 1
    assert events[0].payload == json.loads(payload_json)
    assert path.read_bytes() == original


_DUPLICATE_LINES = [
    _line().replace(b'"sequence":1', b'"sequence":1,"sequence":1'),
    _line().replace(b'"sequence":1', b'"sequence":9,"sequence":1'),
    _line().replace(b'"schema_version":1', b'"schema_version":99,"schema_version":1'),
    _line().replace(b'"run_id":"run-duplicates"', b'"run_id":"wrong","run_id":"run-duplicates"'),
    _line('{"x":1,"x":1}'),
    _line('{"x":1,"x":2}'),
    _line('{"nested":{"x":1,"x":2}}'),
    _line('{"items":[{"x":1,"x":2}]}'),
    _line('{"x":1,"\\u0078":2}'),
    _line('{"":1,"":2}'),
    _line('{"fixture-private-key":"fixture-private-value","fixture-private-key":2}'),
    _line().replace(b'"payload":{}', b'"payload":{},"payload":{}'),
    _line().replace(b'"payload":{}', b'"payload":{},"extension":{"x":1,"x":2}'),
]

_DUPLICATE_IDS = [
    "same-sequence", "overwritten-sequence", "overwritten-version",
    "overwritten-run-id", "same-value", "different-value", "nested",
    "inside-array", "escaped-key", "empty-key", "safe-error-text",
    "duplicate-payload", "unknown-extension",
]


@pytest.mark.parametrize("bad_line", _DUPLICATE_LINES, ids=_DUPLICATE_IDS)
@pytest.mark.parametrize("line_number", [1, 2])
def test_read_rejects_duplicates_with_safe_context_and_cause(
    tmp_path: Path, bad_line: bytes, line_number: int,
) -> None:
    path = tmp_path / "duplicates.jsonl"
    original = bad_line
    if line_number == 2:
        original = _line() + bad_line.replace(b'"sequence":1', b'"sequence":2')
    path.write_bytes(original)

    with pytest.raises(ValueError, match=rf"read_all.*line {line_number}.*valid JSON") as caught:
        JsonlEventStore(path, run_id="run-duplicates").read_all()

    assert str(path) in str(caught.value)
    validation_error = caught.value.__cause__
    assert isinstance(validation_error, ValueError)
    duplicate_error = validation_error.__cause__
    assert isinstance(duplicate_error, ValueError)
    assert "duplicate" in str(duplicate_error).lower()
    for error in (caught.value, validation_error, duplicate_error):
        assert "fixture-private-key" not in str(error)
        assert "fixture-private-value" not in str(error)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "bad_line", [_DUPLICATE_LINES[0], _DUPLICATE_LINES[7]],
    ids=["outer", "nested"],
)
def test_append_rejects_duplicate_trace_before_clock_or_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_line: bytes,
) -> None:
    path = tmp_path / "append-duplicates.jsonl"
    path.write_bytes(bad_line)
    clock = Mock(return_value=datetime(2026, 9, 3, tzinfo=UTC))
    original_open = Path.open
    open_spy = Mock(side_effect=lambda *args, **kwargs: original_open(*args, **kwargs))
    store = JsonlEventStore(path, run_id="run-duplicates", clock=clock)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", lambda *args, **kwargs: open_spy(*args, **kwargs))
        try:
            with pytest.raises(ValueError, match=r"read_all.*line 1.*valid JSON"):
                store.append(EventDraft(event_type="run_finished", payload={}))
        finally:
            modes = [cast(str, call.args[1]) for call in open_spy.call_args_list]
            assert (
                clock.call_count,
                all(mode == "rb" for mode in modes),
                path.read_bytes(),
            ) == (0, True, bad_line)
    assert path.read_bytes() == bad_line
