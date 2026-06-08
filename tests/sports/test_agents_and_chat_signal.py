"""Tests for the new agents endpoint, dashboard match buckets, chat signal."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ----------------------------- /api/v1/agents -----------------------------


class TestAgentsEndpoint:
    def test_all_agents_listed(self):
        from aegeanbench.sports.reporter.agent_registry import (
            build_agents_endpoint,
        )
        out = build_agents_endpoint()
        # Expanded to include iching_specialist
        assert out["total_agents"] == 8
        ids = {a["id"] for a in out["agents"]}
        expected = {
            "stats_specialist", "player_specialist", "strategy_specialist",
            "market_specialist", "news_specialist", "chat_specialist",
            "occult_specialist", "iching_specialist",
        }
        assert ids == expected

    def test_max_weight_share_below_safety(self):
        from aegeanbench.sports.reporter.agent_registry import build_agents_endpoint
        out = build_agents_endpoint()
        # Paper Refinement Validity requires max share < 50%
        assert out["max_weight_share"] < out["safety_threshold"]

    def test_each_agent_has_chinese_fields(self):
        from aegeanbench.sports.reporter.agent_registry import SPORTS_AGENTS_PUBLIC
        for a in SPORTS_AGENTS_PUBLIC:
            assert a["name_zh"], f"{a['id']} missing name_zh"
            assert a["description_zh"], f"{a['id']} missing description_zh"


# ----------------------------- dashboard match buckets -----------------------------


class TestDashboardMatchBuckets:
    def _mk_run_with_kickoff(self, match_id: str, kickoff: datetime, run_id: str = "r1"):
        """Build a run bundle whose aegean prediction carries a kickoff_at."""
        return {
            "manifest": {
                "run_id": run_id,
                "created_at": "2026-06-12T17:00:00Z",
                "competition": "FIFA World Cup 2026",
            },
            "predictions": {
                "aegean": [
                    {
                        "match_id": match_id,
                        "p_home_win": 0.41,
                        "p_draw": 0.30,
                        "p_away_win": 0.29,
                        "confidence": 0.89,
                        "rationale": "test",
                        "metadata": {"kickoff_at": kickoff.isoformat()},
                    }
                ],
            },
        }

    def test_live_upcoming_recent_classification(self):
        from aegeanbench.sports.reporter.endpoints import _classify_matches_by_time

        now = datetime(2026, 6, 12, 19, 0, tzinfo=timezone.utc)
        # Live: kicked off 90 min ago
        live_run = self._mk_run_with_kickoff(
            "WC2026-LIVE", now - timedelta(minutes=90), run_id="rl"
        )
        # Upcoming: kicks off in 6 hours
        upcoming_run = self._mk_run_with_kickoff(
            "WC2026-UP", now + timedelta(hours=6), run_id="ru"
        )
        # Recent: finished 10 hours ago (kickoff 12 hours ago)
        recent_run = self._mk_run_with_kickoff(
            "WC2026-OLD", now - timedelta(hours=12), run_id="ro"
        )
        # Outside window: kickoff 48 hours ago
        ancient_run = self._mk_run_with_kickoff(
            "WC2026-ANCIENT", now - timedelta(hours=48), run_id="ra"
        )

        buckets = _classify_matches_by_time(
            [live_run, upcoming_run, recent_run, ancient_run],
            now_iso=now.isoformat(),
        )

        live_ids = {m["match_id"] for m in buckets["live"]}
        upcoming_ids = {m["match_id"] for m in buckets["upcoming"]}
        recent_ids = {m["match_id"] for m in buckets["recent"]}

        assert "WC2026-LIVE" in live_ids
        assert "WC2026-UP" in upcoming_ids
        assert "WC2026-OLD" in recent_ids
        # Ancient match should not appear in any bucket
        assert "WC2026-ANCIENT" not in (live_ids | upcoming_ids | recent_ids)

    def test_dashboard_includes_buckets(self, tmp_path):
        from aegeanbench.sports.models import Prediction
        from aegeanbench.sports.orchestrator import save_run
        from aegeanbench.sports.orchestrator.persistence import load_run
        from aegeanbench.sports.reporter import build_dashboard_endpoint

        # Build a real persisted run and verify dashboard exposes new fields
        runs_dir = tmp_path / "runs"
        save_run(
            predictions_by_runner={
                "aegean": [Prediction("WC2026-A1", "aegean", 0.5, 0.3, 0.2)],
            },
            portfolio=None,
            runs_dir=runs_dir,
        )
        runs = [load_run(d) for d in runs_dir.iterdir() if d.is_dir()]
        dashboard = build_dashboard_endpoint(runs)
        assert "live_matches" in dashboard
        assert "upcoming_matches" in dashboard
        assert "recent_results" in dashboard


# ----------------------------- chat signal -----------------------------


class TestChatSignalEndpoint:
    def test_chat_signal_accepts_high_volume(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/chat/signal",
                json={
                    "match_id": "WC2026-A1",
                    "room_id": "room_xyz",
                    "message_count": 50,
                    "window_seconds": 300,
                    "strong_sentiment": True,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["accepted"] is True
            assert data["match_id"] == "WC2026-A1"
            assert data["room_id"] == "room_xyz"
            assert data["trigger"]["reason"] == "chat_heat"

    def test_chat_signal_low_volume_rejected(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/chat/signal",
                json={
                    "match_id": "WC2026-A1",
                    "message_count": 5,   # below default threshold of 30
                    "strong_sentiment": False,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["accepted"] is False
            assert data["trigger"] is None


# ----------------------------- /api/v1/agents over HTTP -----------------------------


class TestAgentsHTTP:
    def test_agents_endpoint_via_http(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.get("/api/v1/agents")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_agents"] == 8
            assert "stats_specialist" in [a["id"] for a in data["agents"]]
            assert "iching_specialist" in [a["id"] for a in data["agents"]]
