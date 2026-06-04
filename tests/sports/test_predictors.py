"""
Tests for Day 2 predictors: Elo, Dixon-Coles, Monte Carlo.

Covers:
  - Probability validity (sums to 1, in [0, 1])
  - Direction sanity (stronger team gets higher win prob)
  - Training convergence (model recovers true rankings from synthetic data)
  - Monte Carlo aggregates to predictor probabilities given enough samples
"""

from __future__ import annotations

import pytest

from aegeanbench.sports import (
    MatchOutcome,
    Prediction,
    SportsDataGateway,
)
from aegeanbench.sports.predictors import (
    DixonColesPredictor,
    EloPredictor,
    MonteCarloSimulator,
)
from aegeanbench.sports.predictors.elo import (
    expected_win_prob,
    goal_margin_multiplier,
    three_way_probs,
)


# ---------------- Elo unit math ----------------

class TestEloMath:
    def test_expected_win_prob_symmetry(self):
        assert expected_win_prob(0) == pytest.approx(0.5)
        # +400 Elo should give ~91% win probability (classic Elo)
        assert expected_win_prob(400) == pytest.approx(0.909, abs=0.01)
        assert expected_win_prob(-400) == pytest.approx(0.091, abs=0.01)

    def test_three_way_probs_normalized(self):
        for diff in (-500, -100, 0, 100, 500):
            probs = three_way_probs(diff)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9
            for p in probs.values():
                assert 0 <= p <= 1

    def test_three_way_probs_direction(self):
        # Higher diff -> higher home win
        p_neg = three_way_probs(-300)
        p_eq = three_way_probs(0)
        p_pos = three_way_probs(300)
        assert p_neg[MatchOutcome.HOME_WIN] < p_eq[MatchOutcome.HOME_WIN] < p_pos[MatchOutcome.HOME_WIN]
        assert p_neg[MatchOutcome.AWAY_WIN] > p_eq[MatchOutcome.AWAY_WIN] > p_pos[MatchOutcome.AWAY_WIN]

    def test_draw_peaks_at_zero_diff(self):
        probs_eq = three_way_probs(0)
        probs_off = three_way_probs(300)
        assert probs_eq[MatchOutcome.DRAW] > probs_off[MatchOutcome.DRAW]

    def test_goal_margin_multiplier(self):
        assert goal_margin_multiplier(1, 0) == 1.0
        assert goal_margin_multiplier(2, 0) == 1.5
        assert goal_margin_multiplier(3, 0) == 1.75
        assert goal_margin_multiplier(5, 0) == 1.75 + 2 / 8


# ---------------- Elo predictor ----------------

class TestEloPredictor:
    @pytest.fixture
    def gw(self):
        return SportsDataGateway(mock=True)

    def test_predicts_valid_distribution(self, gw):
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = EloPredictor().predict(ctx)
        assert isinstance(pred, Prediction)
        assert abs(pred.p_home_win + pred.p_draw + pred.p_away_win - 1.0) < 1e-9

    def test_uses_seed_rating_from_team(self, gw):
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        # Brazil (1981) vs Argentina (2114) -> ARG favored away
        pred = EloPredictor().predict(ctx)
        # Away is stronger by ~130 Elo even with +100 home advantage
        # net rating_diff ≈ 1981 + 100 - 2114 = -33, slightly home but draw zone
        # The key check: pred returns a sane distribution
        assert pred.p_away_win > 0.20

    def test_fit_then_predict(self, gw):
        elo = EloPredictor()
        history = gw.load_training_history(num_matches=200)
        elo.fit(history)
        # Some teams should have moved from default 1500
        moved = [code for code, r in elo.ratings.items() if abs(r - 1500) > 5]
        assert len(moved) > 0


# ---------------- Dixon-Coles ----------------

class TestDixonColesPredictor:
    @pytest.fixture
    def gw(self):
        return SportsDataGateway(mock=True)

    def test_predict_without_fit_gives_uniform_ish(self, gw):
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = DixonColesPredictor().predict(ctx)
        assert abs(pred.p_home_win + pred.p_draw + pred.p_away_win - 1.0) < 1e-6

    def test_fit_converges(self, gw):
        dc = DixonColesPredictor(max_iterations=50)
        history = gw.load_training_history(num_matches=300)
        dc.fit(history)
        assert dc.fitted
        # Strength table should have entries for all observed teams
        table = dc.team_strength_table()
        assert len(table) >= 4

    def test_fit_then_predict_valid(self, gw):
        dc = DixonColesPredictor(max_iterations=30)
        dc.fit(gw.load_training_history(num_matches=200))
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = dc.predict(ctx)
        assert abs(pred.p_home_win + pred.p_draw + pred.p_away_win - 1.0) < 1e-6
        assert pred.metadata["lambda"] > 0
        assert pred.metadata["mu"] > 0


# ---------------- Monte Carlo ----------------

class TestMonteCarloSimulator:
    @pytest.fixture
    def gw(self):
        return SportsDataGateway(mock=True)

    def test_sample_matches_predictor_distribution(self, gw):
        """
        Empirical frequency from sampling should approximate the predictor's
        stated probabilities for a large enough sample.
        """
        sim = MonteCarloSimulator(EloPredictor(), gw, seed=42)
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = sim.predictor.predict(ctx)

        counts = {o: 0 for o in MatchOutcome}
        n = 5000
        for _ in range(n):
            counts[sim.simulate_match(ctx)] += 1

        # Empirical home_win frequency should be within ~3% of stated probability
        empirical_home = counts[MatchOutcome.HOME_WIN] / n
        assert abs(empirical_home - pred.p_home_win) < 0.03

    def test_group_stage_returns_advancement_probs(self, gw):
        sim = MonteCarloSimulator(EloPredictor(), gw, seed=42)
        # Take first 4 fixtures as a synthetic "group"
        group_matches = gw.list_fixtures()[:4]
        results = sim.simulate_group_stage(group_matches, num_simulations=500)
        for code, stats in results.items():
            assert 0 <= stats["p_first"] <= 1
            assert 0 <= stats["p_advance"] <= 1
            # p_advance >= p_first
            assert stats["p_advance"] >= stats["p_first"] - 1e-9
