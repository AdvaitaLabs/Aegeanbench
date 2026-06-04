"""
Tests for the Day 6 extension: real data sources + V2 WebSocket / Q&A.

All tests run offline. Network paths are exercised by monkey-patching
`requests` so we don't depend on flaky external services in CI.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any, Dict

import pytest

from aegeanbench.sports.models import Team
from aegeanbench.sports.sources import (
    ClubEloAdapter,
    FIFA_TO_CLUBELO,
    enrich_teams_with_elo,
)
from aegeanbench.sports.sources.clubelo import CLUBELO_API_BASE
from aegeanbench.sports.sources.football_data import FootballDataAdapter
from aegeanbench.sports.sources.kaggle_loader import (
    KaggleHistoryLoader,
    download_kaggle_dataset,
)


# ----------------------------- clubelo -----------------------------


SAMPLE_CLUBELO_CSV = """\
Rank,Club,Country,Level,Elo,From,To
1,Argentina,ARG,1,2114.0,2026-06-01,2026-06-15
2,France,FRA,1,2058.0,2026-06-01,2026-06-15
3,England,ENG,1,2026.0,2026-06-01,2026-06-15
4,Brazil,BRA,1,1981.0,2026-06-01,2026-06-15
"""


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestClubEloAdapter:
    def test_mock_mode_returns_fallback_table(self):
        a = ClubEloAdapter(mock=True)
        table = a.fetch_elo_table()
        assert "BRA" in table
        assert "ARG" in table
        # Ensure we got copies not the live table reference
        assert table is not a.FALLBACK_ELO

    def test_csv_parser_extracts_known_teams(self):
        parsed = ClubEloAdapter._parse_csv(SAMPLE_CLUBELO_CSV)
        assert parsed["ARG"] == pytest.approx(2114.0)
        assert parsed["BRA"] == pytest.approx(1981.0)

    def test_csv_parser_ignores_unknown_clubs(self):
        text = "Rank,Club,Country,Level,Elo,From,To\n1,FC Unknown,XYZ,1,1500.0,2026-01-01,2026-01-31\n"
        parsed = ClubEloAdapter._parse_csv(text)
        assert parsed == {}

    def test_live_fetch_uses_requests(self, monkeypatch, tmp_path):
        """Verify the adapter calls requests.get with the expected URL."""
        from aegeanbench.sports.cache import FileCache
        calls = {}

        def fake_get(url, timeout):
            calls["url"] = url
            calls["timeout"] = timeout
            return _FakeResponse(SAMPLE_CLUBELO_CSV)

        # Replace requests.get globally for the duration of this test
        import requests
        monkeypatch.setattr(requests, "get", fake_get)

        a = ClubEloAdapter(mock=False, cache=FileCache(cache_dir=tmp_path))
        table = a.fetch_elo_table(as_of=date(2026, 6, 4))
        assert calls["url"] == f"{CLUBELO_API_BASE}/2026-06-04"
        assert table["ARG"] == 2114.0

    def test_live_fetch_failure_falls_back(self, monkeypatch, tmp_path):
        from aegeanbench.sports.cache import FileCache

        def boom(url, timeout):
            raise RuntimeError("network down")

        import requests
        monkeypatch.setattr(requests, "get", boom)

        a = ClubEloAdapter(mock=False, cache=FileCache(cache_dir=tmp_path))
        table = a.fetch_elo_table()
        # Should return the fallback table
        assert table["BRA"] == ClubEloAdapter.FALLBACK_ELO["BRA"]

    def test_enrich_teams_mutates_elo_rating(self):
        teams = [Team("BRA", "Brazil"), Team("ARG", "Argentina"), Team("XYZ", "Unknown")]
        enrich_teams_with_elo(teams, adapter=ClubEloAdapter(mock=True))
        bra = next(t for t in teams if t.fifa_code == "BRA")
        arg = next(t for t in teams if t.fifa_code == "ARG")
        xyz = next(t for t in teams if t.fifa_code == "XYZ")
        assert bra.elo_rating is not None
        assert arg.elo_rating is not None
        assert xyz.elo_rating is None  # not in fallback table


# ----------------------------- football-data.org -----------------------------


SAMPLE_FD_FIXTURES = {
    "matches": [
        {
            "id": 12345,
            "utcDate": "2026-06-12T18:00:00Z",
            "stage": "GROUP_STAGE",
            "homeTeam": {"tla": "BRA", "name": "Brazil"},
            "awayTeam": {"tla": "ARG", "name": "Argentina"},
            "score": {"fullTime": {"home": None, "away": None}},
            "venue": "MetLife Stadium",
        },
        {
            "id": 12346,
            "utcDate": "2026-06-13T18:00:00Z",
            "stage": "GROUP_STAGE",
            "homeTeam": {"tla": "FRA", "name": "France"},
            "awayTeam": {"tla": "GER", "name": "Germany"},
            "score": {"fullTime": {"home": 2, "away": 1}},
            "venue": "AT&T Stadium",
        },
    ]
}


class TestFootballDataReal:
    def test_competition_code_mapping(self):
        a = FootballDataAdapter(api_key="dummy", mock_by_default=False)
        assert a._competition_code("FIFA World Cup 2026") == "WC"
        assert a._competition_code("WC") == "WC"

    def test_parse_fixtures_basic(self):
        out = FootballDataAdapter._parse_fixtures(
            SAMPLE_FD_FIXTURES["matches"], "FIFA World Cup 2026"
        )
        assert len(out) == 2
        # First match: no result
        assert out[0].home_team.fifa_code == "BRA"
        assert out[0].away_team.fifa_code == "ARG"
        assert out[0].result is None
        # Second match: 2-1 home win
        assert out[1].result is not None
        assert out[1].result.home_goals == 2
        assert out[1].result.away_goals == 1
        assert out[1].venue == "AT&T Stadium"

    def test_real_fetch_fixtures_via_monkeypatch(self, monkeypatch):
        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return SAMPLE_FD_FIXTURES

        def fake_get(url, headers=None, params=None, timeout=None):
            assert "WC/matches" in url
            assert headers == {"X-Auth-Token": "dummy-key"}
            return FakeResp()

        import requests
        monkeypatch.setattr(requests, "get", fake_get)

        a = FootballDataAdapter(api_key="dummy-key", mock_by_default=False)
        from aegeanbench.sports.sources.base import FetchPolicy
        out = a.fetch_fixtures("FIFA World Cup 2026", policy=FetchPolicy(mock=False))
        assert len(out) == 2
        assert out[0].home_team.fifa_code == "BRA"

    def test_real_fetch_falls_back_on_network_error(self, monkeypatch):
        def boom(url, headers=None, params=None, timeout=None):
            raise RuntimeError("dns")
        import requests
        monkeypatch.setattr(requests, "get", boom)
        a = FootballDataAdapter(api_key="dummy", mock_by_default=False)
        from aegeanbench.sports.sources.base import FetchPolicy
        out = a.fetch_fixtures("FIFA World Cup 2026", policy=FetchPolicy(mock=False))
        # Falls back to mock fixtures (8 matches)
        assert len(out) >= 1


# ----------------------------- Kaggle loader -----------------------------


KAGGLE_SAMPLE_CSV = """\
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2024-01-15,Brazil,Argentina,2,1,Friendly,Rio,Brazil,FALSE
2024-02-20,France,Germany,1,1,Friendly,Paris,France,FALSE
"""


class TestKaggleLoader:
    def test_loads_real_csv_when_present(self, tmp_path):
        path = tmp_path / "results.csv"
        path.write_text(KAGGLE_SAMPLE_CSV)
        loader = KaggleHistoryLoader(csv_path=path, mock_by_default=False)
        matches = loader.load(num_matches=10)
        # 2 real rows + may pad with synthetic if num_matches > rows; we asked
        # for 10 so we expect the file's 2 then stop (CSV only had 2)
        assert len(matches) == 2
        assert matches[0].home_team.name == "Brazil"
        assert matches[0].result.home_goals == 2

    def test_falls_back_to_synthetic_when_file_missing(self, tmp_path):
        path = tmp_path / "does_not_exist.csv"
        loader = KaggleHistoryLoader(csv_path=path, mock_by_default=False)
        matches = loader.load(num_matches=20)
        assert len(matches) == 20  # synthetic fill

    def test_download_helper_returns_path(self, tmp_path):
        dest = tmp_path / "kaggle_results.csv"
        path = download_kaggle_dataset(dest=dest)
        assert path == dest


# ----------------------------- LiveHub -----------------------------


class TestLiveHub:
    def test_publish_to_no_subscribers_returns_zero(self):
        from aegeanbench.sports.reporter.realtime import LiveHub
        hub = LiveHub()
        assert asyncio.run(hub.publish("nobody", {"x": 1})) == 0

    def test_subscribe_publish_receive(self):
        from aegeanbench.sports.reporter.realtime import LiveHub
        hub = LiveHub()

        async def scenario():
            q = await hub.subscribe("predictions")
            n = await hub.publish("predictions", {"hello": "world"})
            received = await asyncio.wait_for(q.get(), timeout=1.0)
            await hub.unsubscribe("predictions", q)
            return n, received

        n, received = asyncio.run(scenario())
        assert n == 1
        assert received == {"hello": "world"}

    def test_unsubscribe_cleans_channel(self):
        from aegeanbench.sports.reporter.realtime import LiveHub
        hub = LiveHub()

        async def scenario():
            q = await hub.subscribe("chan")
            await hub.unsubscribe("chan", q)
            return hub.subscriber_count()

        assert asyncio.run(scenario()) == 0


# ----------------------------- FastAPI V2 routes -----------------------------


class TestServerV2:
    def test_v2_routes_present(self, tmp_path):
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed")
        from aegeanbench.sports.reporter.server import create_app
        app = create_app(runs_dir=tmp_path)
        routes = {r.path for r in app.router.routes}
        # V1
        assert "/api/v1/runners" in routes
        # V2
        assert "/api/v1/agents/{agent_id}/answer" in routes
        assert "/ws/predictions" in routes
        assert "/ws/matches/{match_id}" in routes
        assert "/api/v1/_internal/publish" in routes

    def test_qa_endpoint_returns_503_when_no_handler(self, tmp_path):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app
        app = create_app(runs_dir=tmp_path, qa_handler=None)
        client = TestClient(app)
        resp = client.post("/api/v1/agents/stats_specialist/answer", json={"question": "?"})
        assert resp.status_code == 503

    def test_qa_endpoint_routes_to_handler(self, tmp_path):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi[test] not installed")
        from aegeanbench.sports.reporter.server import create_app

        class FakeQAResponse:
            def to_dict(self):
                return {
                    "agent_id": "stats_specialist",
                    "answer": "Brazil edges it.",
                    "confidence": 0.7,
                }

        class FakeHandler:
            async def answer(self, **kwargs):
                self.last_kwargs = kwargs
                return FakeQAResponse()

        handler = FakeHandler()
        app = create_app(runs_dir=tmp_path, qa_handler=handler)
        client = TestClient(app)
        resp = client.post(
            "/api/v1/agents/stats_specialist/answer",
            json={"question": "who wins?", "user_name": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Brazil edges it."
        assert handler.last_kwargs["question"] == "who wins?"
        assert handler.last_kwargs["user_name"] == "test"
