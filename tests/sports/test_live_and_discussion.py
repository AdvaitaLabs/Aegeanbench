"""Tests for live scores, scheduler, discussion trace, and bundle endpoints."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ----------------------------- soccersapi_live -----------------------------


class TestSoccersAPILive:
    def test_mock_live_matches(self):
        from aegeanbench.sports.sources.soccersapi_live import SoccersAPILiveClient
        client = SoccersAPILiveClient(mock=True)
        matches = client.fetch_live_matches()
        assert len(matches) >= 1
        m = matches[0]
        assert m.match_id
        assert m.status

    def test_mock_events(self):
        from aegeanbench.sports.sources.soccersapi_live import (
            LiveEventKind, SoccersAPILiveClient,
        )
        client = SoccersAPILiveClient(mock=True)
        events = client.fetch_match_events("WC2026-A1")
        assert len(events) >= 2
        assert events[0].kind == LiveEventKind.KICK_OFF
        assert any(e.kind == LiveEventKind.GOAL for e in events)

    def test_event_high_priority_flag(self):
        from aegeanbench.sports.sources.soccersapi_live import LiveEvent, LiveEventKind
        goal = LiveEvent(match_id="X", event_id="1", kind=LiveEventKind.GOAL, minute=10)
        yellow = LiveEvent(match_id="X", event_id="2", kind=LiveEventKind.YELLOW_CARD, minute=10)
        assert goal.is_high_priority is True
        assert yellow.is_high_priority is False

    def test_live_state_finished_flag(self):
        from aegeanbench.sports.sources.soccersapi_live import LiveMatchState
        live = LiveMatchState(match_id="X", status="in_play", minute=45)
        ft = LiveMatchState(match_id="X", status="ft", minute=90)
        assert live.is_live is True
        assert ft.is_finished is True

    def test_event_dedupe(self):
        from aegeanbench.sports.sources.soccersapi_live import (
            LiveEvent, LiveEventKind, SoccersAPILiveClient,
        )
        events = [
            LiveEvent(match_id="X", event_id="e1", kind=LiveEventKind.KICK_OFF, minute=0),
            LiveEvent(match_id="X", event_id="e2", kind=LiveEventKind.GOAL, minute=10),
            LiveEvent(match_id="X", event_id="e3", kind=LiveEventKind.GOAL, minute=23),
        ]
        deduped = SoccersAPILiveClient._dedupe_events(events, last_seen="e1")
        assert [e.event_id for e in deduped] == ["e2", "e3"]


# ----------------------------- MatchEventScheduler -----------------------------


class TestScheduler:
    def test_goal_triggers_immediately(self):
        from aegeanbench.sports.live.scheduler import MatchEventScheduler
        from aegeanbench.sports.sources.soccersapi_live import LiveEvent, LiveEventKind

        s = MatchEventScheduler()
        ev = LiveEvent(match_id="M1", event_id="e1", kind=LiveEventKind.GOAL, minute=23)
        trigger = s.on_live_event(ev)
        assert trigger is not None
        assert trigger.priority == 1

    def test_throttle_blocks_second_event(self):
        from aegeanbench.sports.live.scheduler import MatchEventScheduler
        from aegeanbench.sports.sources.soccersapi_live import LiveEvent, LiveEventKind

        s = MatchEventScheduler(min_reconsensus_seconds=300)
        s.on_live_event(LiveEvent(match_id="M1", event_id="e1", kind=LiveEventKind.GOAL, minute=10))
        # Second goal 30s later should be throttled
        second = LiveEvent(match_id="M1", event_id="e2", kind=LiveEventKind.GOAL, minute=11)
        result = s.on_live_event(second)
        assert result is None

    def test_yellow_card_does_not_trigger(self):
        from aegeanbench.sports.live.scheduler import MatchEventScheduler
        from aegeanbench.sports.sources.soccersapi_live import LiveEvent, LiveEventKind

        s = MatchEventScheduler()
        ev = LiveEvent(match_id="M1", event_id="e1", kind=LiveEventKind.YELLOW_CARD, minute=23)
        assert s.on_live_event(ev) is None

    def test_chat_heat_requires_threshold(self):
        from aegeanbench.sports.live.scheduler import MatchEventScheduler

        s = MatchEventScheduler(chat_heat_threshold=30)
        assert s.on_chat_heat("M1", 5) is None     # below threshold
        assert s.on_chat_heat("M1", 50) is not None  # above threshold

    def test_pre_match_checkpoint_fires_once(self):
        from aegeanbench.sports.live.scheduler import MatchEventScheduler

        s = MatchEventScheduler()
        kickoff = datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc)

        # Now is exactly T-2h
        t_minus_2h = kickoff - timedelta(hours=2)
        trigger1 = s.on_clock_tick("M1", kickoff, now=t_minus_2h)
        assert trigger1 is not None

        # Same checkpoint should not fire again
        trigger2 = s.on_clock_tick("M1", kickoff, now=t_minus_2h + timedelta(seconds=20))
        assert trigger2 is None

        # T-30m checkpoint should still fire
        t_minus_30m = kickoff - timedelta(minutes=30)
        # Use a fresh scheduler so throttle doesn't block; or wait long enough
        s2 = MatchEventScheduler(min_reconsensus_seconds=60)
        s2.on_clock_tick("M1", kickoff, now=t_minus_2h)
        # Advance simulated time past the throttle window
        trigger3 = s2.on_clock_tick("M1", kickoff, now=t_minus_30m)
        assert trigger3 is not None

    def test_manual_trigger_bypasses_throttle(self):
        from aegeanbench.sports.live.scheduler import MatchEventScheduler
        from aegeanbench.sports.sources.soccersapi_live import LiveEvent, LiveEventKind

        s = MatchEventScheduler(min_reconsensus_seconds=600)
        s.on_live_event(LiveEvent(match_id="M1", event_id="e1", kind=LiveEventKind.GOAL, minute=10))
        # Even within the throttle window, manual fires
        manual = s.manual_trigger("M1", detail="ops override")
        assert manual.priority == 1


# ----------------------------- DiscussionTrace -----------------------------


class TestDiscussionTrace:
    def test_parse_real_aegean_response_with_discussion_rounds(self):
        """
        Verify we correctly parse the actual GroupConsensusResult shape
        returned by aegean-consensus (discussion_rounds, not rounds_history).
        """
        import json
        from aegeanbench.sports.discussion import parse_aegean_response_to_trace

        # This mirrors the JSON aegean-consensus emits at
        # POST /api/v1/groups/{id}/consensus
        consensus_response = {
            "consensus_id": "c1",
            "success": True,
            "rounds_used": 2,
            "consensus_reached": True,
            "weighted_votes": {"home_win": 1.75},
            "consensus_path": ["stats_specialist", "market_specialist"],
            "final_solution": {
                "agent_id": "stats_specialist",
                "answer": json.dumps({"p_home_win": 0.55, "p_draw": 0.28, "p_away_win": 0.17}),
                "confidence": 0.72,
                "reasoning": "Home edge from xG",
            },
            "discussion_rounds": [
                {
                    "round_number": 1,
                    "agent_responses": {
                        "stats_specialist": {
                            "agent_id": "stats_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2,
                            }),
                            "confidence": 0.7,
                            "reasoning": "xG favours home",
                        },
                        "market_specialist": {
                            "agent_id": "market_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.4, "p_draw": 0.3, "p_away_win": 0.3,
                            }),
                            "confidence": 0.6,
                            "reasoning": "Market sees a closer game",
                        },
                    },
                    "candidate_answer": None,
                    "candidate_confidence": 0.0,
                    "stability_counter": 1,
                    "consensus_status": "ongoing",
                },
                {
                    "round_number": 2,
                    "agent_responses": {
                        "stats_specialist": {
                            "agent_id": "stats_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.55, "p_draw": 0.28, "p_away_win": 0.17,
                            }),
                            "confidence": 0.75,
                            "reasoning": "Holding",
                        },
                        "market_specialist": {
                            "agent_id": "market_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.5, "p_draw": 0.28, "p_away_win": 0.22,
                            }),
                            "confidence": 0.7,
                            "reasoning": "Converged toward stats",
                        },
                    },
                    "candidate_confidence": 0.65,
                    "stability_counter": 2,
                    "consensus_status": "reached",
                },
            ],
        }
        trace = parse_aegean_response_to_trace("M1", "aegean", consensus_response)
        assert trace.rounds_used == 2
        assert len(trace.rounds) == 2
        # Both agents picked home_win each round so no position change
        round2 = trace.rounds[1]
        assert all(a.current_argmax == "home_win" for a in round2.agents)
        # Second round should be marked as quorum reached
        assert round2.quorum_reached is True
        # raw metadata carries the consensus_path through
        assert trace.raw_metadata["consensus_path"] == [
            "stats_specialist", "market_specialist",
        ]

    def test_parse_response_with_rounds_history(self):
        import json
        from aegeanbench.sports.discussion import parse_aegean_response_to_trace

        consensus_response = {
            "rounds_used": 2,
            "weighted_votes": {"home_win": 1.8},
            "rounds_history": [
                {
                    "round_number": 1,
                    "quorum_reached": False,
                    "agent_solutions": [
                        {
                            "agent_id": "stats_specialist",
                            "role": "stats_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2,
                            }),
                            "confidence": 0.7,
                            "reasoning": "Recent xG favours home",
                        },
                        {
                            "agent_id": "market_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.4, "p_draw": 0.3, "p_away_win": 0.3,
                            }),
                            "confidence": 0.6,
                            "reasoning": "Market sees a closer game",
                        },
                    ],
                },
                {
                    "round_number": 2,
                    "quorum_reached": True,
                    "candidate_outcome": "home_win",
                    "candidate_confidence": 0.65,
                    "agent_solutions": [
                        {
                            "agent_id": "stats_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.55, "p_draw": 0.28, "p_away_win": 0.17,
                            }),
                            "confidence": 0.75,
                            "reasoning": "Holding position",
                        },
                        {
                            "agent_id": "market_specialist",
                            "answer": json.dumps({
                                "p_home_win": 0.5, "p_draw": 0.28, "p_away_win": 0.22,
                            }),
                            "confidence": 0.7,
                            "reasoning": "Converged toward stats lens",
                        },
                    ],
                },
            ],
            "final_solution": {"reasoning": "Home favoured by ~55%"},
            "consensus_reached": True,
        }
        trace = parse_aegean_response_to_trace("M1", "aegean", consensus_response)
        assert trace.rounds_used == 2
        assert len(trace.rounds) == 2
        # Both agents picked home_win in round 1 -> no position change in round 2
        round2 = trace.rounds[1]
        assert all(a.current_argmax == "home_win" for a in round2.agents)
        assert all(a.previous_argmax == "home_win" for a in round2.agents)
        assert all(not a.changed_position for a in round2.agents)

    def test_parse_handles_position_change(self):
        import json
        from aegeanbench.sports.discussion import parse_aegean_response_to_trace

        # Agent A switches from away_win to home_win between rounds
        resp = {
            "rounds_history": [
                {
                    "round_number": 1,
                    "agent_solutions": [
                        {"agent_id": "a1", "answer": json.dumps({
                            "p_home_win": 0.2, "p_draw": 0.3, "p_away_win": 0.5,
                        })},
                    ],
                },
                {
                    "round_number": 2,
                    "agent_solutions": [
                        {"agent_id": "a1", "answer": json.dumps({
                            "p_home_win": 0.6, "p_draw": 0.3, "p_away_win": 0.1,
                        })},
                    ],
                },
            ],
        }
        trace = parse_aegean_response_to_trace("M1", "aegean", resp)
        round2_a1 = trace.rounds[1].agents[0]
        assert round2_a1.previous_argmax == "away_win"
        assert round2_a1.current_argmax == "home_win"
        assert round2_a1.changed_position is True


# ----------------------------- bundle endpoints -----------------------------


class TestBundleEndpoints:
    def test_dashboard_endpoint_bundles_everything(self, tmp_path):
        from aegeanbench.sports.models import Prediction
        from aegeanbench.sports.orchestrator import save_run
        from aegeanbench.sports.orchestrator.persistence import load_run
        from aegeanbench.sports.reporter import build_dashboard_endpoint

        runs_dir = tmp_path / "runs"
        save_run(
            predictions_by_runner={
                "elo": [Prediction("M1", "elo", 0.5, 0.3, 0.2)],
                "aegean": [Prediction("M1", "aegean", 0.45, 0.30, 0.25)],
            },
            portfolio=None,
            runs_dir=runs_dir,
            label="t1",
        )
        save_run(
            predictions_by_runner={
                "elo": [Prediction("M2", "elo", 0.4, 0.3, 0.3)],
            },
            portfolio=None,
            runs_dir=runs_dir,
            label="t2",
        )
        runs = []
        for entry in runs_dir.iterdir():
            if entry.is_dir():
                runs.append(load_run(entry))

        dashboard = build_dashboard_endpoint(runs)
        assert dashboard["leaderboard"]["rows"] == []   # no evaluations
        assert dashboard["summary_stats"]["total_runs"] == 2
        assert dashboard["summary_stats"]["total_predictions"] == 3
        assert len(dashboard["recent_runs"]) == 2

    def test_match_state_endpoint_returns_none_for_unknown(self):
        from aegeanbench.sports.reporter import build_match_state_endpoint
        assert build_match_state_endpoint([], "MISSING") is None

    def test_match_state_endpoint_includes_predictions(self, tmp_path):
        from aegeanbench.sports.models import Prediction
        from aegeanbench.sports.orchestrator import save_run
        from aegeanbench.sports.orchestrator.persistence import load_run
        from aegeanbench.sports.reporter import build_match_state_endpoint

        runs_dir = tmp_path / "runs"
        save_run(
            predictions_by_runner={
                "elo": [Prediction("WC2026-A1", "elo", 0.5, 0.3, 0.2)],
                "aegean": [Prediction(
                    "WC2026-A1", "aegean", 0.45, 0.30, 0.25,
                    metadata={"discussion": {"rounds_used": 2, "rounds": []}},
                )],
            },
            portfolio=None,
            runs_dir=runs_dir,
        )
        runs = [load_run(d) for d in runs_dir.iterdir() if d.is_dir()]
        state = build_match_state_endpoint(runs, "WC2026-A1")
        assert state is not None
        assert state["match_id"] == "WC2026-A1"
        assert len(state["predictions"]) == 2
        # discussion trace was attached via aegean metadata
        assert state["discussion"] is not None
        assert state["discussion"]["rounds_used"] == 2
