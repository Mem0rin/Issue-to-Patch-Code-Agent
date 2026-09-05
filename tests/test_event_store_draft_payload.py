import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from code_agent.event_store import EventDraft, JsonlEventStore


def test_nested_json_payload_is_accepted_and_persisted(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "strict-json.jsonl"
    nested: Mapping[str, object] = MappingProxyType(
        {
            "none": None,
            "bool": False,
            "integer": 0,
            "float": 1.25,
            "text": "",
            "list": [None, True, -1, 0.0, ""],
            "object": MappingProxyType({}),
        }
    )
    store = JsonlEventStore(
        trace_path,
        run_id="run-json-payload",
        clock=lambda: datetime(2026, 9, 3, 18, 0, tzinfo=UTC),
    )

    stored = store.append(
        EventDraft(
            event_type="payload_checked",
            payload=MappingProxyType({"nested": nested}),
        )
    )

    expected_payload: dict[str, object] = {
        "nested": {
            "none": None,
            "bool": False,
            "integer": 0,
            "float": 1.25,
            "text": "",
            "list": [None, True, -1, 0.0, ""],
            "object": {},
        }
    }
    assert stored.payload == expected_payload
    assert store.read_all()[0].payload == expected_payload


def test_event_draft_accepts_shared_container_without_cycle() -> None:
    shared: list[object] = [{"result": 42}]

    draft = EventDraft(
        event_type="tool_result_reused",
        payload={"first": shared, "second": shared},
    )

    assert draft.payload == {"first": shared, "second": shared}


@pytest.mark.parametrize(
    "payload",
    [
        cast(Mapping[str, object], {1: "value"}),
        cast(Mapping[str, object], {"nested": {False: "value"}}),
    ],
    ids=["top-level", "nested"],
)
def test_event_draft_rejects_non_string_payload_keys(
    payload: Mapping[str, object],
) -> None:
    with pytest.raises(TypeError, match=r"payload.*keys.*strings"):
        EventDraft(event_type="run_started", payload=payload)


@pytest.mark.parametrize(
    "invalid_value",
    [
        b"bytes",
        ("tuple",),
        {"set"},
        datetime(2026, 9, 3, 18, 0, tzinfo=UTC),
        object(),
    ],
    ids=["bytes", "tuple", "set", "datetime", "object"],
)
def test_event_draft_rejects_non_json_payload_values(
    invalid_value: object,
) -> None:
    with pytest.raises(TypeError, match=r"payload.*JSON-compatible"):
        EventDraft(
            event_type="run_started",
            payload={"invalid": invalid_value},
        )


@pytest.mark.parametrize(
    "invalid_float",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_event_draft_rejects_non_finite_payload_floats(
    invalid_float: float,
) -> None:
    with pytest.raises(ValueError, match=r"payload.*finite"):
        EventDraft(
            event_type="run_started",
            payload={"invalid": invalid_float},
        )


def test_event_draft_rejects_cyclic_list_payload() -> None:
    cyclic_list: list[object] = []
    cyclic_list.append(cyclic_list)

    with pytest.raises(ValueError, match=r"payload.*cycles"):
        EventDraft(
            event_type="run_started",
            payload={"cycle": cyclic_list},
        )


def test_event_draft_rejects_cyclic_mapping_payload() -> None:
    cyclic_mapping: dict[str, object] = {}
    cyclic_mapping["self"] = cyclic_mapping

    with pytest.raises(ValueError, match=r"payload.*cycles"):
        EventDraft(
            event_type="run_started",
            payload=cyclic_mapping,
        )


def test_append_revalidates_mutated_payload_before_core_logic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "mutated-payload.jsonl"
    items: list[object] = []
    draft = EventDraft(
        event_type="run_started",
        payload={"items": items},
    )
    items.append(object())
    clock_calls: list[str] = []

    def clock() -> datetime:
        clock_calls.append("called")
        return datetime(2026, 9, 3, 18, 0, tzinfo=UTC)

    def unexpected_read() -> tuple[object, ...]:
        raise AssertionError("read_all must not run for an invalid payload")

    store = JsonlEventStore(
        trace_path,
        run_id="run-mutated-payload",
        clock=clock,
    )
    monkeypatch.setattr(store, "read_all", unexpected_read)

    with pytest.raises(TypeError, match=r"payload.*JSON-compatible"):
        store.append(draft)

    assert clock_calls == []
    assert not trace_path.exists()
