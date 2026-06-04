"""
End-to-end smoke tests for the sports data pipeline.

Validates that with mock data alone we can:
  1. List World Cup teams
  2. List fixtures
  3. Build a complete MatchContext for any fixture
  4. Construct a Prediction with valid probabilities
  5. Score a Prediction against ground truth (MatchResult)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import pytest

from aegeanbench.sports import (
    Match,
    MatchOutcome,
    MatchResult,
    Prediction,
    SportsDataGateway,
)
from aegeanbench.sports.cache import FileCache


class TestModels:
    """Sanity checks on the dataclass layer."""

    def test_match_result_derives_outcome(self):
        r = MatchResult(home_goals=2, away_goals=1)
        assert r.outcome == MatchOutcome.HOME_WIN

        r = MatchResult(home_goals=1, away_goals=2)
        assert r.outcome == MatchOutcome.AWAY_WIN

        r = MatchResult(home_goals=1, away_goals=1)
        assert r.outcome == MatchOutcome.DRAW

    def test_prediction_validates_probability_sum(self):
        # Valid
        p = Prediction(
            match_id="X", runner_id="test",
            p_home_win=0.5, p_draw=0.3, p_away_win=0.2,
        )
        assert p.predicted_outcome == MatchOutcome.HOME_WIN
        assert p.predicted_probability == 0.5

        # Invalid (sum != 1)
        with pytest.raises(ValueError):
            Prediction(
                match_id="X", runner_id="test",
                p_home_win=0.5, p_draw=0.5, p_away_win=0.5,
            )

    def test_odds_fair_probs_sum_to_one(self):
        from aegeanbench.sports import Odds
        o = Odds("test", datetime.now(), home_win=2.0, draw=3.0, away_win=4.0)
        fp = o.fair_probs
        assert abs(sum(fp.values()) - 1.0) < 1e-9
        # Margin should be positive
        assert o.margin > 0


class TestCache:
    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(cache_dir=Path(tmp))
            cache.set({"hello": "world", "n": 42}, "test", "key1")
            got = cache.get("test", "key1")
            assert got == {"hello": "world", "n": 42}

    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(cache_dir=Path(tmp))
            assert cache.get("nonexistent") is None

    def test_get_or_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = FileCache(cache_dir=Path(tmp))
            call_count = {"n": 0}

            def compute():
                call_count["n"] += 1
                return "computed"

            v1 = cache.get_or_compute(compute, "k1")
            v2 = cache.get_or_compute(compute, "k1")
            assert v1 == v2 == "computed"
            assert call_count["n"] == 1, "compute_fn should be called only once"


class TestGatewayEndToEnd:
    """The flagship test: mock data flows through the whole pipeline."""

    @pytest.fixture
    def gateway(self):
        return SportsDataGateway(mock=True)

    def test_list_teams(self, gateway):
        teams = gateway.list_teams()
        assert len(teams) >= 8
        fifa_codes = {t.fifa_code for t in teams}
        assert {"BRA", "ARG", "FRA", "GER"}.issubset(fifa_codes)

    def test_list_fixtures(self, gateway):
        fixtures = gateway.list_fixtures()
        assert len(fixtures) >= 1
        m = fixtures[0]
        assert m.competition == "FIFA World Cup 2026"
        assert m.home_team.fifa_code != m.away_team.fifa_code

    def test_build_full_context(self, gateway):
        """Pick the opening match and verify all enrichment kicks in."""
        fixtures = gateway.list_fixtures()
        m = fixtures[0]
        ctx = gateway.build_context(m)

        # Match itself
        assert ctx.match.match_id == m.match_id
        # Odds
        assert len(ctx.match.odds) >= 2
        for o in ctx.match.odds:
            assert o.home_win > 1.0
            # fair probs sum to 1
            assert abs(sum(o.fair_probs.values()) - 1.0) < 1e-9
        # Lineups
        assert len(ctx.match.home_lineup) == 11
        assert len(ctx.match.away_lineup) == 11
        # xG profiles
        assert "xg_for" in ctx.home_xg_profile
        assert "xg_against" in ctx.away_xg_profile
        # History
        assert len(ctx.home_history) > 0
        assert len(ctx.away_history) > 0
        # h2h
        assert len(ctx.h2h) > 0
        # Summary works
        assert ctx.match.home_team.fifa_code in ctx.summary()

    def test_consensus_odds_aggregate(self, gateway):
        m = gateway.list_fixtures()[0]
        ctx = gateway.build_context(m)
        co = ctx.match.consensus_odds
        assert co is not None
        # Should be a valid probability distribution
        assert abs(sum(co.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in co.values())

    def test_training_history(self, gateway):
        history = gateway.load_training_history(num_matches=100)
        assert len(history) == 100
        # Every historical match should have ground truth
        for m in history:
            assert m.result is not None
            assert m.result.outcome in (
                MatchOutcome.HOME_WIN, MatchOutcome.DRAW, MatchOutcome.AWAY_WIN,
            )


class TestPredictionScoring:
    """
    Predictions need to be scoreable against ground truth. This test
    proves the scoring math works at the model level (full metrics
    engine comes in Day 2/5).
    """

    def test_correct_prediction_high_probability(self):
        # Predict HOME_WIN with 70% probability; actual is HOME_WIN
        p = Prediction(
            match_id="X", runner_id="test",
            p_home_win=0.70, p_draw=0.20, p_away_win=0.10,
        )
        actual = MatchResult(home_goals=2, away_goals=1)
        assert p.predicted_outcome == actual.outcome  # correct

        # Brier score: sum (p_i - actual_i)^2 across all 3 outcomes
        # actual = home → (0.70-1)^2 + (0.20-0)^2 + (0.10-0)^2 = 0.09+0.04+0.01 = 0.14
        brier = (
            (p.p_home_win - 1) ** 2
            + (p.p_draw - 0) ** 2
            + (p.p_away_win - 0) ** 2
        )
        assert 0.13 < brier < 0.15

    def test_wrong_prediction_high_brier(self):
        # Predict HOME_WIN with high confidence; actual is AWAY_WIN
        p = Prediction(
            match_id="X", runner_id="test",
            p_home_win=0.70, p_draw=0.20, p_away_win=0.10,
        )
        # actual = away → (0.70-0)^2 + (0.20-0)^2 + (0.10-1)^2 = 0.49+0.04+0.81 = 1.34
        brier = (
            (p.p_home_win - 0) ** 2
            + (p.p_draw - 0) ** 2
            + (p.p_away_win - 1) ** 2
        )
        assert 1.3 < brier < 1.4
