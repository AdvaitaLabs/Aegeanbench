"""Tests for the post-sprint Day 7 additions: weather + injuries + occult upgrade + pre-launch script."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path

import pytest


# ----------------------------- OpenWeather -----------------------------


class TestOpenWeather:
    def test_mock_summary_returns_complete_shape(self):
        from aegeanbench.sports.sources.openweather import OpenWeatherAdapter
        a = OpenWeatherAdapter(mock=True)
        s = a.fetch_for_match("Mexico City", datetime(2026, 6, 12, 18, 0))
        assert s is not None
        for k in ("city", "temperature_c", "humidity_pct", "wind_kph",
                  "precipitation_mm_h", "conditions", "kickoff_at"):
            assert k in s
        assert s["_mock"] is True

    def test_unknown_city_still_returns_mock(self):
        from aegeanbench.sports.sources.openweather import OpenWeatherAdapter
        a = OpenWeatherAdapter(mock=False, api_key=None)   # no key -> mock
        s = a.fetch_for_match("Atlantis", datetime(2026, 6, 12))
        assert s is not None
        assert s["city"] == "Atlantis"

    def test_live_path_monkeypatched(self, monkeypatch, tmp_path):
        import requests
        from aegeanbench.sports.cache import FileCache
        from aegeanbench.sports.sources.openweather import OpenWeatherAdapter

        class Resp:
            def raise_for_status(self): pass
            def json(self):
                return {
                    "main": {"temp": 25.5, "humidity": 70},
                    "wind": {"speed": 4.0},
                    "rain": {"1h": 0.2},
                    "weather": [{"main": "Rain"}],
                }

        def fake_get(url, params=None, timeout=None):
            assert "appid" in params
            return Resp()

        monkeypatch.setattr(requests, "get", fake_get)
        a = OpenWeatherAdapter(api_key="dummy", mock=False, cache=FileCache(cache_dir=tmp_path))
        s = a.fetch_for_match("Mexico City", datetime(2026, 6, 12, 18, 0))
        assert s["temperature_c"] == pytest.approx(25.5)
        assert s["wind_kph"] == pytest.approx(4.0 * 3.6, abs=0.01)
        assert s["conditions"] == "Rain"


# ----------------------------- Wikipedia injuries -----------------------------


SAMPLE_WIKI_SQUADS_HTML = """
<html><body>
<h3><span class="mw-headline" id="Brazil">Brazil</span></h3>
<table>
<tr><td><a href="/wiki/Casemiro">Casemiro</a></td><td>injured (ankle), replaced by Andre</td></tr>
<tr><td><a href="/wiki/Vinicius">Vinicius Jr</a></td><td>squad member</td></tr>
</table>
<h3><span class="mw-headline" id="Germany">Germany</span></h3>
<table>
<tr><td><a href="/wiki/Kimmich">Kimmich</a></td><td>suspended (yellow card accumulation)</td></tr>
</table>
</body></html>
"""


class TestWikipediaInjuries:
    def test_mock_returns_known_teams(self):
        from aegeanbench.sports.sources.wikipedia_injuries import WikipediaInjuriesAdapter
        a = WikipediaInjuriesAdapter(mock_by_default=True)
        data = a.fetch_all()
        assert "Brazil" in data
        assert any(p["player"] == "Casemiro" for p in data["Brazil"])

    def test_parser_extracts_injuries_and_suspensions(self):
        from aegeanbench.sports.sources.wikipedia_injuries import WikipediaInjuriesAdapter
        parsed = WikipediaInjuriesAdapter._parse_squads_html(SAMPLE_WIKI_SQUADS_HTML)
        assert "Brazil" in parsed
        brazil = parsed["Brazil"]
        assert any(p["status"] == "injured" and "Casemiro" in p["player"] for p in brazil)
        assert "Germany" in parsed
        ger = parsed["Germany"]
        assert any(p["status"] == "suspended" and "Kimmich" in p["player"] for p in ger)

    def test_fetch_for_team(self):
        from aegeanbench.sports.sources.wikipedia_injuries import WikipediaInjuriesAdapter
        a = WikipediaInjuriesAdapter(mock_by_default=True)
        bra = a.fetch_for_team("Brazil")
        assert bra and bra[0]["status"] == "injured"

    def test_live_failure_falls_back(self, monkeypatch, tmp_path):
        import requests
        from aegeanbench.sports.cache import FileCache
        from aegeanbench.sports.sources.wikipedia_injuries import WikipediaInjuriesAdapter

        def boom(url, headers=None, timeout=None):
            raise RuntimeError("dns")

        monkeypatch.setattr(requests, "get", boom)
        a = WikipediaInjuriesAdapter(
            mock_by_default=False,
            cache=FileCache(cache_dir=tmp_path),
        )
        data = a.fetch_all()
        # Should fall back to mock
        assert "Brazil" in data


# ----------------------------- OccultAgent extension -----------------------------


class TestOccultData:
    """Tests for the new occult_data helpers in aegean-consensus."""

    def test_tarot_deck_has_78_cards(self):
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports.occult_data import TAROT_DECK
            assert len(TAROT_DECK) == 78
        finally:
            sys.path.pop(0)

    def test_draw_tarot_deterministic_with_seed(self):
        import random
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports.occult_data import draw_tarot_spread
            rng1 = random.Random("seed")
            rng2 = random.Random("seed")
            spread1 = draw_tarot_spread(rng=rng1)
            spread2 = draw_tarot_spread(rng=rng2)
            assert spread1 == spread2
            assert len(spread1) == 3
            assert {c["position"] for c in spread1} == {"past", "present", "future"}
        finally:
            sys.path.pop(0)

    def test_build_mystic_context(self):
        import sys
        sys.path.insert(0, "/Users/liudongqi/aegean-consensus/src")
        try:
            from aegean.agents.sports.occult_data import (
                build_mystic_context,
                format_mystic_context_for_prompt,
            )
            ctx = build_mystic_context("BRA", "ARG", match_date=date(2026, 6, 12))
            assert "tarot_spread" in ctx
            assert ctx["home_zodiac"]["sign"]   # not empty
            assert ctx["numerology"]["number"] in range(1, 10)
            text = format_mystic_context_for_prompt(ctx)
            assert "tarot" in text.lower()
            assert "MYSTIC" in text
        finally:
            sys.path.pop(0)


# ----------------------------- pre-launch script -----------------------------


class TestPreLaunchBenchmark:
    def test_script_runs_to_completion(self, capsys):
        # Import the local script
        import importlib.util
        path = Path(__file__).resolve().parents[2] / "scripts" / "pre_launch_benchmark.py"
        spec = importlib.util.spec_from_file_location("pre_launch_benchmark", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rc = mod.main([])
        # rc may be 0 or 1 depending on calibration; both are valid outcomes
        assert rc in (0, 1)
        captured = capsys.readouterr()
        assert "PRE-LAUNCH CALIBRATION REPORT" in captured.out
