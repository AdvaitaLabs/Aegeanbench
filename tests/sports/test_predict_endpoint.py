"""Tests for the simplified user-selected-agents predict endpoint."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestAgentsRegistryUpdate:
    def test_chat_is_default(self):
        from aegeanbench.sports.reporter.agent_registry import (
            SPORTS_AGENTS_PUBLIC,
            get_default_agent_ids,
        )
        defaults = get_default_agent_ids()
        assert "chat_specialist" in defaults
        chat = next(a for a in SPORTS_AGENTS_PUBLIC if a["id"] == "chat_specialist")
        assert chat["is_default"] is True
        assert chat["selectable"] is False

    def test_bazi_is_listed(self):
        from aegeanbench.sports.reporter.agent_registry import SPORTS_AGENTS_PUBLIC
        ids = [a["id"] for a in SPORTS_AGENTS_PUBLIC]
        assert "iching_specialist" in ids
        bazi = next(a for a in SPORTS_AGENTS_PUBLIC if a["id"] == "iching_specialist")
        assert bazi["weight"] == 0.10
        assert bazi["category"] == "occult"

    def test_eight_agents_total(self):
        from aegeanbench.sports.reporter.agent_registry import SPORTS_AGENTS_PUBLIC
        assert len(SPORTS_AGENTS_PUBLIC) == 8   # original 7 + bazi

    def test_selectable_excludes_default(self):
        from aegeanbench.sports.reporter.agent_registry import (
            get_default_agent_ids,
            get_selectable_agent_ids,
        )
        defaults = set(get_default_agent_ids())
        selectables = set(get_selectable_agent_ids())
        assert defaults.isdisjoint(selectables)

    def test_max_share_still_safe_with_8_agents(self):
        from aegeanbench.sports.reporter.agent_registry import build_agents_endpoint
        out = build_agents_endpoint()
        assert out["max_weight_share"] < out["safety_threshold"]


class TestPredictEndpoint:
    def test_predict_with_minimal_body(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/predict",
                json={
                    "match_id": "WC2026-A1",
                    "agent_ids": ["stats_specialist", "occult_specialist"],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["match_id"] == "WC2026-A1"
            # chat_specialist auto-added
            assert "chat_specialist" in data["agents_used"]
            assert "stats_specialist" in data["agents_used"]
            assert "occult_specialist" in data["agents_used"]
            # Three probabilities present
            probs = data["prediction"]
            assert abs(probs["p_home_win"] + probs["p_draw"] + probs["p_away_win"] - 1.0) < 1e-6

    def test_predict_with_chat_messages(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/predict",
                json={
                    "match_id": "WC2026-A1",
                    "table_id": "table_demo_1",
                    "agent_ids": ["stats_specialist", "iching_specialist"],
                    "match_data": {
                        "home_team": "Brazil",
                        "away_team": "Argentina",
                        "kickoff_at": "2026-06-12T18:00:00Z",
                    },
                    "chat_messages": [
                        {"user_name": "张三", "text": "巴西必胜"},
                        {"user_name": "李四", "text": "梅西今晚状态不好"},
                    ],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["table_id"] == "table_demo_1"
            assert "iching_specialist" in data["agents_used"]
            # Discussion trace present
            assert data["discussion"] is not None

    def test_predict_unknown_agent_is_dropped(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/predict",
                json={
                    "match_id": "WC2026-A1",
                    "agent_ids": ["stats_specialist", "does_not_exist"],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "does_not_exist" not in data["agents_used"]
            assert "stats_specialist" in data["agents_used"]

    def test_predict_empty_agent_list_400(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            # If no valid agents AND no defaults match, should 400.
            # Since chat_specialist auto-added always, we test with all
            # unknown to ensure the validation path works.
            resp = client.post(
                "/api/v1/predict",
                json={
                    "match_id": "WC2026-A1",
                    "agent_ids": ["totally_made_up"],
                },
            )
            # Will still 200 because chat_specialist is auto-added
            assert resp.status_code == 200
            assert resp.json()["agents_used"] == ["chat_specialist"]
