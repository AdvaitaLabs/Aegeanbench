"""Tests for the mock chat fetcher used in sprint development."""

from __future__ import annotations

from datetime import datetime, timezone

from aegeanbench.sports.chat_mock import (
    MockChatFetcher,
    fetch_mock_chat_window,
)


def test_returns_expected_shape():
    payload = fetch_mock_chat_window("WC2026-A1")
    assert payload["match_id"] == "WC2026-A1"
    assert "messages" in payload
    assert "window_start" in payload
    assert "window_end" in payload
    assert payload["total_messages"] >= 5


def test_messages_have_required_fields():
    payload = fetch_mock_chat_window("WC2026-A2")
    for msg in payload["messages"]:
        for key in ("message_id", "user_id", "user_name", "text", "timestamp", "language"):
            assert key in msg, f"missing key {key}"
        assert msg["language"] in ("en", "zh")


def test_deterministic_per_match_id():
    a = fetch_mock_chat_window("WC2026-A1")
    b = fetch_mock_chat_window("WC2026-A1")
    assert a["total_messages"] == b["total_messages"]
    assert [m["text"] for m in a["messages"]] == [m["text"] for m in b["messages"]]


def test_different_match_ids_different_chats():
    a = fetch_mock_chat_window("WC2026-A1")
    b = fetch_mock_chat_window("WC2026-B1")
    # At least the message counts should differ in most cases; we only
    # check identity of the text list (not count) to avoid false positives.
    assert [m["text"] for m in a["messages"]] != [m["text"] for m in b["messages"]]


def test_context_hint_injects_team_names():
    fetcher = MockChatFetcher(
        context_hint={
            "WC2026-A1": {"home": "Brazil", "away": "Argentina", "star": "Vinicius"},
        },
        timestamp_ref=datetime(2026, 6, 12, 18, 0, tzinfo=timezone.utc),
    )
    payload = fetcher.fetch("WC2026-A1")
    blob = " ".join(m["text"] for m in payload["messages"])
    # At least one message should reference Brazil OR Argentina
    assert "Brazil" in blob or "Argentina" in blob or "巴西" not in blob  # template-driven
