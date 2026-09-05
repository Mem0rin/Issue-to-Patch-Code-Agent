from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def test_event_draft_accepts_representative_values() -> None:
    payload: Mapping[str, object] = {"attempt": 1, "retry": False}

    draft = EventDraft(event_type="model_call_started", payload=payload)

    assert draft.event_type == "model_call_started"
    assert draft.payload == payload


def test_event_draft_accepts_one_character_type_and_empty_mapping() -> None:
    payload: Mapping[str, object] = MappingProxyType({})

    draft = EventDraft(event_type="x", payload=payload)

    assert draft.event_type == "x"
    assert draft.payload is payload


@pytest.mark.parametrize(
    "event_type",
    [None, True, 1, 1.0, b"run_started", [], {}],
    ids=["none", "bool", "int", "float", "bytes", "list", "mapping"],
)
def test_event_draft_rejects_non_string_event_type(
    event_type: object,
) -> None:
    with pytest.raises(TypeError, match=r"event_type.*string"):
        EventDraft(
            event_type=cast(str, event_type),
            payload={},
        )


@pytest.mark.parametrize(
    "event_type",
    ["", " ", "\t\r\n"],
    ids=["empty", "space", "whitespace"],
)
def test_event_draft_rejects_blank_event_type(event_type: str) -> None:
    with pytest.raises(ValueError, match=r"event_type.*blank"):
        EventDraft(event_type=event_type, payload={})


@pytest.mark.parametrize(
    "payload",
    [None, True, 1, 1.0, "payload", b"payload", [], ()],
    ids=[
        "none",
        "bool",
        "int",
        "float",
        "string",
        "bytes",
        "list",
        "tuple",
    ],
)
def test_event_draft_rejects_non_mapping_payload(payload: object) -> None:
    with pytest.raises(TypeError, match=r"payload.*mapping"):
        EventDraft(
            event_type="run_started",
            payload=cast(Mapping[str, object], payload),
        )


@pytest.mark.parametrize(
    "invalid_draft",
    [None, "draft", {}, object()],
    ids=["none", "string", "mapping", "object"],
)
def test_append_rejects_non_draft_before_read_clock_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_draft: object,
) -> None:
    trace_path = tmp_path / "invalid-draft.jsonl"
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 17, 0, tzinfo=UTC)

    def unexpected_read() -> tuple[object, ...]:
        raise AssertionError("read_all must not run for an invalid draft")

    store = JsonlEventStore(trace_path, run_id="run-input", clock=clock)
    monkeypatch.setattr(store, "read_all", unexpected_read)

    with pytest.raises(TypeError, match=r"draft.*EventDraft"):
        store.append(cast(EventDraft, invalid_draft))

    assert clock_calls == []
    assert not trace_path.exists()
