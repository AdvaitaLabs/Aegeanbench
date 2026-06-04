"""
Tests for the Day 7 additions: FBref live scraper + demo entry point.
All network paths exercised via monkeypatched `requests`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aegeanbench.sports.cache import FileCache
from aegeanbench.sports.sources.fbref import (
    FBREF_BASE,
    FIFA_TO_FBREF,
    FBrefAdapter,
)


# ----------------------------- HTML parsing -----------------------------


SAMPLE_FBREF_HTML = """
<html><body>
<!--
<table class="stats_table">
  <tr>
    <td data-stat="xg_per90">2.08</td>
    <td data-stat="xg_against_per90">0.89</td>
    <td data-stat="possession">61.4</td>
    <td data-stat="ppda">8.7</td>
  </tr>
</table>
-->
</body></html>
"""


SAMPLE_FBREF_HTML_MISSING_PPDA = """
<html><body>
<!--
<table>
  <tr>
    <td data-stat="xg_per90">1.55</td>
    <td data-stat="xg_against_per90">1.10</td>
    <td data-stat="possession">52.3</td>
  </tr>
</table>
-->
</body></html>
"""


class TestFBrefParser:
    def test_extracts_xg_metrics(self):
        profile = FBrefAdapter._parse_xg_from_html(SAMPLE_FBREF_HTML)
        assert profile["xg_for"] == pytest.approx(2.08)
        assert profile["xg_against"] == pytest.approx(0.89)
        assert profile["possession"] == pytest.approx(0.614)   # 61.4% normalised
        assert profile["ppda"] == pytest.approx(8.7)

    def test_handles_missing_ppda_with_default(self):
        profile = FBrefAdapter._parse_xg_from_html(SAMPLE_FBREF_HTML_MISSING_PPDA)
        assert profile["xg_for"] == pytest.approx(1.55)
        assert profile["ppda"] == 10.0   # league-average fallback

    def test_returns_defaults_when_html_is_blank(self):
        profile = FBrefAdapter._parse_xg_from_html("")
        assert profile["xg_for"] == 1.2
        assert profile["xg_against"] == 1.3
        assert profile["possession"] == 0.50

    def test_possession_normalised_when_already_fraction(self):
        # If a fbref change ever returns 0.50 directly, we should not divide twice
        html = '<td data-stat="possession">0.50</td>'
        profile = FBrefAdapter._parse_xg_from_html(html)
        assert profile["possession"] == 0.50


# ----------------------------- mock mode -----------------------------


class TestFBrefMockMode:
    def test_default_is_mock_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("FBREF_LIVE", raising=False)
        adapter = FBrefAdapter()
        assert adapter.mock_by_default is True

    def test_env_var_enables_live(self, monkeypatch):
        monkeypatch.setenv("FBREF_LIVE", "1")
        adapter = FBrefAdapter()
        assert adapter.mock_by_default is False

    def test_mock_mode_returns_table(self):
        adapter = FBrefAdapter(mock_by_default=True)
        profile = adapter.fetch_xg_profile("BRA")
        assert profile["xg_for"] > 0
        assert profile["xg_against"] > 0


# ----------------------------- live mode (monkey-patched) -----------------------------


class _FakeResp:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestFBrefLiveScrape:
    def test_live_path_hits_expected_url(self, monkeypatch, tmp_path):
        calls = {}

        def fake_get(url, headers=None, timeout=None):
            calls["url"] = url
            calls["headers"] = headers
            return _FakeResp(SAMPLE_FBREF_HTML)

        import requests
        monkeypatch.setattr(requests, "get", fake_get)
        # Skip the 5-second polite delay during tests
        monkeypatch.setattr("aegeanbench.sports.sources.fbref.RATE_LIMIT_SECONDS", 0.0)

        adapter = FBrefAdapter(
            mock_by_default=False,
            cache=FileCache(cache_dir=tmp_path),
        )
        profile = adapter.fetch_xg_profile("BRA")

        expected_id, expected_name = FIFA_TO_FBREF["BRA"]
        assert calls["url"] == f"{FBREF_BASE}/{expected_id}/{expected_name}-Men-Stats"
        assert "User-Agent" in calls["headers"]
        assert profile["xg_for"] == pytest.approx(2.08)

    def test_live_caches_subsequent_calls(self, monkeypatch, tmp_path):
        call_count = {"n": 0}

        def fake_get(url, headers=None, timeout=None):
            call_count["n"] += 1
            return _FakeResp(SAMPLE_FBREF_HTML)

        import requests
        monkeypatch.setattr(requests, "get", fake_get)
        monkeypatch.setattr("aegeanbench.sports.sources.fbref.RATE_LIMIT_SECONDS", 0.0)

        adapter = FBrefAdapter(
            mock_by_default=False,
            cache=FileCache(cache_dir=tmp_path),
        )
        adapter.fetch_xg_profile("BRA")
        adapter.fetch_xg_profile("BRA")  # Should be cached
        assert call_count["n"] == 1

    def test_live_failure_falls_back_to_mock(self, monkeypatch, tmp_path):
        def boom(url, headers=None, timeout=None):
            raise RuntimeError("dns")

        import requests
        monkeypatch.setattr(requests, "get", boom)
        monkeypatch.setattr("aegeanbench.sports.sources.fbref.RATE_LIMIT_SECONDS", 0.0)

        adapter = FBrefAdapter(
            mock_by_default=False,
            cache=FileCache(cache_dir=tmp_path),
        )
        profile = adapter.fetch_xg_profile("BRA")
        # Falls back to mock BRA entry
        assert profile["xg_for"] > 0

    def test_unknown_team_raises_then_falls_back(self, monkeypatch, tmp_path):
        # No network access at all
        import requests
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResp(""))
        monkeypatch.setattr("aegeanbench.sports.sources.fbref.RATE_LIMIT_SECONDS", 0.0)

        adapter = FBrefAdapter(
            mock_by_default=False,
            cache=FileCache(cache_dir=tmp_path),
        )
        profile = adapter.fetch_xg_profile("XYZ")
        # Falls back to default profile since XYZ isn't in either FIFA_TO_FBREF
        # nor the mock table
        assert profile["xg_for"] == 1.20


# ----------------------------- demo entry point -----------------------------


class TestDemoEntryPoint:
    def test_demo_runs_to_completion(self, tmp_path, monkeypatch, capsys):
        # Redirect runs_dir to a fresh tmp so we don't pollute ~/.aegeanbench
        monkeypatch.setattr(
            "aegeanbench.sports.orchestrator.persistence.DEFAULT_RUNS_DIR",
            tmp_path / "runs",
        )

        from aegeanbench.sports.demo import main
        output_dir = tmp_path / "api_output"
        rc = main(["--output", str(output_dir)])

        assert rc == 0
        # All expected JSONs exist
        assert (output_dir / "runners.json").exists()
        assert (output_dir / "tournaments.json").exists()
        assert (output_dir / "leaderboard.json").exists()
        # Demo printed all six sections
        captured = capsys.readouterr()
        assert "STEP 1" in captured.out
        assert "STEP 6" in captured.out
        assert "DONE" in captured.out
