"""
Tests for weather wiring into MatchContext + the divination endpoint.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


class TestWeatherInContext:
    def test_gateway_populates_weather_when_venue_matches(self):
        from aegeanbench.sports import SportsDataGateway

        gw = SportsDataGateway(mock=True)
        # Pull a mock fixture and force a known city venue
        fixtures = gw.list_fixtures()
        m = fixtures[0]
        m.venue = "MetLife Stadium, New York"   # contains "New York"
        ctx = gw.build_context(m)
        assert ctx.weather is not None
        assert ctx.weather["city"] == "New York"
        for k in ("temperature_c", "humidity_pct", "wind_kph", "conditions"):
            assert k in ctx.weather

    def test_gateway_weather_none_when_unknown_venue(self):
        from aegeanbench.sports import SportsDataGateway

        gw = SportsDataGateway(mock=True)
        m = gw.list_fixtures()[0]
        m.venue = "Some Unknown Field, Nowhere"
        ctx = gw.build_context(m)
        assert ctx.weather is None

    def test_prompt_includes_weather_section(self):
        from aegeanbench.sports import SportsDataGateway
        from aegeanbench.sports.prompts import build_user_prompt

        gw = SportsDataGateway(mock=True)
        m = gw.list_fixtures()[0]
        m.venue = "MetLife Stadium, New York"
        ctx = gw.build_context(m)
        text = build_user_prompt(ctx)
        assert "Weather at Kickoff" in text
        assert "New York" in text

    def test_prompt_omits_weather_when_none(self):
        from aegeanbench.sports import SportsDataGateway
        from aegeanbench.sports.prompts import build_user_prompt

        gw = SportsDataGateway(mock=True)
        m = gw.list_fixtures()[0]
        m.venue = "Unknown Place"
        ctx = gw.build_context(m)
        text = build_user_prompt(ctx)
        assert "Weather at Kickoff" not in text


# ----------------------------- divination -----------------------------


class TestDivinationCore:
    def test_tarot_draws_three_cards(self):
        from aegeanbench.sports.divination import perform_divination
        r = perform_divination(
            div_type="tarot",
            match_id="WC2026-A1",
            card_indices=[0, 10, 20],
        )
        assert r.type == "tarot"
        assert len(r.drawn_cards) == 3
        positions = [c["position"] for c in r.drawn_cards]
        assert positions == ["past", "present", "future"]
        # Outcome lean probabilities sum to 1
        lean = r.outcome_lean
        assert abs(lean["p_home_win"] + lean["p_draw"] + lean["p_away_win"] - 1.0) < 1e-6

    def test_tarot_with_no_indices_falls_back(self):
        from aegeanbench.sports.divination import perform_divination
        r = perform_divination(div_type="tarot", match_id="WC2026-A1")
        assert len(r.drawn_cards) == 3
        # Should still return non-empty reading
        assert r.reading

    def test_tarot_dedupes_repeated_indices(self):
        from aegeanbench.sports.divination import perform_divination
        r = perform_divination(
            div_type="tarot", match_id="WC2026-A1",
            card_indices=[5, 5, 5],
        )
        # Padded with deterministic fallbacks, should yield 3 distinct cards
        names = {c["name"] for c in r.drawn_cards}
        assert len(names) == 3

    def test_iching_picks_hexagram(self):
        from aegeanbench.sports.divination import perform_divination
        r = perform_divination(
            div_type="iching",
            match_id="WC2026-A1",
            hexagram_index=13,    # 14th hexagram (id "14")
        )
        assert r.type == "iching"
        assert r.hexagram is not None
        assert r.hexagram["id"] == "14"
        assert r.hexagram["name"] == "大有"

    def test_iching_out_of_range_wraps(self):
        from aegeanbench.sports.divination import perform_divination
        r = perform_divination(
            div_type="iching", match_id="WC2026-A1", hexagram_index=999,
        )
        # 999 % 64 = 39 -> hexagram id "40"
        assert r.hexagram["id"] == "40"

    def test_unsupported_type_raises(self):
        from aegeanbench.sports.divination import perform_divination
        with pytest.raises(ValueError):
            perform_divination(div_type="palm_reading", match_id="X")

    def test_llm_call_path(self):
        """When llm_call is provided we use its output instead of template."""
        from aegeanbench.sports.divination import perform_divination

        def fake_llm(system: str, user: str) -> str:
            return (
                '{"reading": "the LLM speaks", '
                '"p_home_win": 0.6, "p_draw": 0.25, "p_away_win": 0.15}'
            )

        r = perform_divination(
            div_type="tarot",
            match_id="X",
            card_indices=[1, 2, 3],
            llm_call=fake_llm,
        )
        assert r.reading == "the LLM speaks"
        assert r.outcome_lean["p_home_win"] == pytest.approx(0.6)


# ----------------------------- /api/v1/divination -----------------------------


class TestDivinationEndpoint:
    def test_tarot_via_http(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/divination",
                json={
                    "type": "tarot",
                    "match_id": "WC2026-A1",
                    "home_team": "Brazil",
                    "away_team": "Argentina",
                    "table_id": "table_demo",
                    "card_indices": [0, 22, 50],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "tarot"
            assert data["match_id"] == "WC2026-A1"
            assert data["table_id"] == "table_demo"
            assert len(data["drawn_cards"]) == 3
            assert data["outcome_lean"]["p_home_win"] >= 0

    def test_iching_via_http(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/divination",
                json={
                    "type": "iching",
                    "match_id": "WC2026-A1",
                    "hexagram_index": 0,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["type"] == "iching"
            assert data["hexagram"]["id"] == "01"
            assert data["hexagram"]["name"] == "乾"

    def test_unsupported_type_returns_400(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(runs_dir=Path(tmp))
            client = TestClient(app)
            resp = client.post(
                "/api/v1/divination",
                json={"type": "palm_reading", "match_id": "X"},
            )
            assert resp.status_code == 400


# ----------------------------- soccersapi optional user -----------------------------


class TestSoccersAPIOptionalUser:
    def test_token_only_works(self, monkeypatch):
        monkeypatch.delenv("AEGEANBENCH_SOCCERSAPI_USER", raising=False)
        monkeypatch.setenv("AEGEANBENCH_SOCCERSAPI_KEY", "tok-123")
        from aegeanbench.sports.sources.soccersapi_live import SoccersAPILiveClient
        client = SoccersAPILiveClient()
        assert client.mock is False    # mock disabled when token present
        assert client.user == ""
        assert client.token == "tok-123"
