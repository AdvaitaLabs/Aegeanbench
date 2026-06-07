"""
Tests for per-room chat support across ChatFetcher, ChatAgent, ChatQAHandler,
MockChatFetcher, and the FastAPI Q&A endpoint.

The product story: many user-created chat rooms can be open in parallel
for the same match. We must thread `room_id` through every layer so the
chat service can route messages correctly and our LLM agents can soft-
reference the room without keeping any per-room persistent state.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# ----------------------------- chat_mock with room_id -----------------------------


class TestMockChatFetcherRooms:
    def test_room_id_echoed_in_payload(self):
        from aegeanbench.sports.chat_mock import MockChatFetcher
        fetcher = MockChatFetcher()
        payload = fetcher.fetch("WC2026-A1", room_id="room_xyz")
        assert payload["room_id"] == "room_xyz"

    def test_different_rooms_produce_different_chats(self):
        from aegeanbench.sports.chat_mock import MockChatFetcher
        fetcher = MockChatFetcher()
        room_a = fetcher.fetch("WC2026-A1", room_id="room_a")
        room_b = fetcher.fetch("WC2026-A1", room_id="room_b")
        # Both rooms talk about same match but the message slate should differ
        msgs_a = [m["text"] for m in room_a["messages"]]
        msgs_b = [m["text"] for m in room_b["messages"]]
        assert msgs_a != msgs_b

    def test_same_room_deterministic(self):
        """Text content stable across calls (timestamps move with wall clock)."""
        from aegeanbench.sports.chat_mock import MockChatFetcher
        a = MockChatFetcher().fetch("WC2026-A1", room_id="room_x")
        b = MockChatFetcher().fetch("WC2026-A1", room_id="room_x")
        assert [m["text"] for m in a["messages"]] == [m["text"] for m in b["messages"]]
        assert [m["user_id"] for m in a["messages"]] == [m["user_id"] for m in b["messages"]]

    def test_no_room_falls_back_to_match_level(self):
        from aegeanbench.sports.chat_mock import fetch_mock_chat_window
        payload = fetch_mock_chat_window("WC2026-A1")
        assert payload["room_id"] is None
        assert payload["total_messages"] > 0


# ----------------------------- ChatAgent with rooms -----------------------------


class _MockLLM:
    def __init__(self):
        self.last_prompt = None
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    async def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps({
            "p_home_win": 0.45, "p_draw": 0.3, "p_away_win": 0.25,
            "confidence": 0.6, "rationale": "ok",
        })


class TestChatAgentRoomThreading:
    def test_room_id_extracted_from_task_and_forwarded(self):
        """If the task prompt contains 'Room ID: room_x' the agent should
        forward room_x to its chat fetcher."""
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports.chat_agent import ChatAgent
            captured: Dict[str, Any] = {}

            def fake_fetch(match_id: str, room_id=None):
                captured["match_id"] = match_id
                captured["room_id"] = room_id
                return {
                    "match_id": match_id,
                    "room_id": room_id,
                    "total_messages": 1,
                    "messages": [{"user_name": "u1", "text": "go BRA", "language": "en"}],
                }

            agent = ChatAgent(llm_client=_MockLLM(), chat_fetch_fn=fake_fetch)
            task = "Match: BRA vs ARG\nMatch ID: WC2026-A1\nRoom ID: room_xyz"
            sol = asyncio.run(agent.generate_solution(task))
            assert captured["match_id"] == "WC2026-A1"
            assert captured["room_id"] == "room_xyz"
            # Solution still produced
            data = json.loads(sol.answer)
            assert "p_home_win" in data
        finally:
            sys.path.pop(0)

    def test_no_room_id_in_task_passes_none(self):
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports.chat_agent import ChatAgent
            captured: Dict[str, Any] = {}

            def fake_fetch(match_id: str, room_id=None):
                captured["room_id"] = room_id
                return {"match_id": match_id, "total_messages": 0, "messages": []}

            agent = ChatAgent(llm_client=_MockLLM(), chat_fetch_fn=fake_fetch)
            asyncio.run(agent.generate_solution("Match ID: WC2026-A1"))
            assert captured["room_id"] is None
        finally:
            sys.path.pop(0)

    def test_legacy_one_arg_fetcher_still_works(self):
        """Older mock fetchers that only accept match_id should still work
        thanks to the TypeError fallback in _fetch_chat."""
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports.chat_agent import ChatAgent

            def legacy_fetch(match_id):
                return {"match_id": match_id, "total_messages": 0, "messages": []}

            agent = ChatAgent(llm_client=_MockLLM(), chat_fetch_fn=legacy_fetch)
            sol = asyncio.run(agent.generate_solution("Match ID: WC2026-A1"))
            assert sol is not None
        finally:
            sys.path.pop(0)


# ----------------------------- ChatQAHandler with rooms -----------------------------


class _MockQAClient:
    def __init__(self):
        self.last_prompt = None
        self.last_usage = {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250}

    async def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps({
            "answer": "Brazil has a slight edge.",
            "confidence": 0.7, "rationale": "xG",
        })


class TestChatQAHandlerRooms:
    def test_room_id_appears_in_prompt(self):
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports import ChatQAHandler, StatsAgent
            from aegean.core.agent import AgentRegistry

            registry = AgentRegistry()
            llm = _MockQAClient()
            registry.register_agent(StatsAgent(llm_client=llm, model_name="mock"))
            handler = ChatQAHandler(agent_registry=registry)

            resp = asyncio.run(handler.answer(
                agent_id="stats_specialist",
                question="will Brazil win?",
                match_context="Match: BRA vs ARG\nMatch ID: WC2026-A1",
                room_id="room_demo_1",
                user_name="dongqi",
            ))
            assert resp.room_id == "room_demo_1"
            assert "room_demo_1" in (llm.last_prompt or "")
        finally:
            sys.path.pop(0)

    def test_recent_messages_labelled_as_this_room(self):
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports import ChatQAHandler, StatsAgent
            from aegean.core.agent import AgentRegistry

            registry = AgentRegistry()
            llm = _MockQAClient()
            registry.register_agent(StatsAgent(llm_client=llm, model_name="mock"))
            handler = ChatQAHandler(agent_registry=registry)

            asyncio.run(handler.answer(
                agent_id="stats_specialist",
                question="thoughts?",
                room_id="room_x",
                recent_messages=[
                    {"user_name": "alice", "text": "BRA wins"},
                    {"user_name": "bob", "text": "no, draw"},
                ],
            ))
            assert "RECENT CONVERSATION IN THIS ROOM" in llm.last_prompt
            assert "alice" in llm.last_prompt and "bob" in llm.last_prompt
        finally:
            sys.path.pop(0)

    def test_response_to_dict_includes_room_id(self):
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports import ChatQAResponse
            r = ChatQAResponse(agent_id="x", question="q", answer="a", room_id="room_y")
            d = r.to_dict()
            assert d["room_id"] == "room_y"
        finally:
            sys.path.pop(0)


# ----------------------------- FastAPI endpoint with room_id -----------------------------


class TestQAEndpointRoomBody:
    def test_endpoint_accepts_room_id_in_body(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        class FakeQAResponse:
            def to_dict(self):
                return {"agent_id": "stats_specialist", "answer": "ok",
                        "confidence": 0.7, "room_id": "room_42"}

        class FakeHandler:
            async def answer(self, **kwargs):
                self.last_kwargs = kwargs
                return FakeQAResponse()

        handler = FakeHandler()
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp), qa_handler=handler)
            client = TestClient(app)
            resp = client.post(
                "/api/v1/agents/stats_specialist/answer",
                json={
                    "question": "hi",
                    "room_id": "room_42",
                    "user_name": "tester",
                    "recent_messages": [{"user_name": "alice", "text": "BRA"}],
                },
            )
        assert resp.status_code == 200
        assert handler.last_kwargs["room_id"] == "room_42"
        assert resp.json()["room_id"] == "room_42"
