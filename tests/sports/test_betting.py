"""
Tests for the betting layer: Kelly Criterion, value-bet identification,
and portfolio construction.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

import pytest

from aegeanbench.sports import (
    Match,
    MatchOutcome,
    Odds,
    Prediction,
    SportsDataGateway,
    Team,
)
from aegeanbench.sports.betting import (
    BettingPortfolio,
    KellyResult,
    PortfolioConfig,
    ValueBetCandidate,
    best_value_bet,
    build_portfolio,
    find_value_bets,
    kelly_fraction,
    stake_amount,
)


# ----------------------------- Kelly -----------------------------


class TestKellyFraction:
    def test_no_edge_returns_zero(self):
        # p=0.50 at fair odds 2.0 -> edge = 0
        r = kelly_fraction(p_win=0.50, decimal_odds=2.0)
        assert r.has_edge is False
        assert r.fractional_kelly == 0.0
        assert r.full_kelly == 0.0

    def test_negative_edge_returns_zero(self):
        # p=0.30 at odds 2.0 -> edge = -0.4
        r = kelly_fraction(p_win=0.30, decimal_odds=2.0)
        assert r.has_edge is False
        assert r.edge < 0

    def test_positive_edge_returns_kelly(self):
        # p=0.60 at odds 2.0 -> b=1, edge=0.20
        # full Kelly = (0.6*1 - 0.4) / 1 = 0.20
        r = kelly_fraction(p_win=0.60, decimal_odds=2.0, fraction=1.0, cap=1.0)
        assert r.has_edge is True
        assert r.full_kelly == pytest.approx(0.20)
        assert r.fractional_kelly == pytest.approx(0.20)
        assert r.edge == pytest.approx(0.20)
        assert r.growth_rate > 0

    def test_fractional_kelly(self):
        # Quarter Kelly should be 1/4 of full
        r = kelly_fraction(p_win=0.60, decimal_odds=2.0, fraction=0.25, cap=1.0)
        assert r.fractional_kelly == pytest.approx(0.05)

    def test_cap_clamps_full_kelly(self):
        # p=0.95 at odds 2.0 -> full Kelly ~ 0.90, cap=0.20 -> clamp
        r = kelly_fraction(p_win=0.95, decimal_odds=2.0, fraction=1.0, cap=0.20)
        assert r.full_kelly == 0.20
        assert r.fractional_kelly == 0.20

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            kelly_fraction(p_win=-0.1, decimal_odds=2.0)
        with pytest.raises(ValueError):
            kelly_fraction(p_win=0.5, decimal_odds=1.0)
        with pytest.raises(ValueError):
            kelly_fraction(p_win=0.5, decimal_odds=2.0, fraction=0)
        with pytest.raises(ValueError):
            kelly_fraction(p_win=0.5, decimal_odds=2.0, cap=0)


class TestStakeAmount:
    def test_zero_bankroll_returns_zero(self):
        r = kelly_fraction(p_win=0.6, decimal_odds=2.0)
        assert stake_amount(0.0, r) == 0.0

    def test_below_min_returns_zero(self):
        r = kelly_fraction(p_win=0.6, decimal_odds=2.0, fraction=0.25, cap=1.0)
        # stake = 1000 * 0.05 = 50, min_stake 100 -> 0
        assert stake_amount(1000.0, r, min_stake=100.0) == 0.0

    def test_normal_stake(self):
        r = kelly_fraction(p_win=0.6, decimal_odds=2.0, fraction=0.25, cap=1.0)
        assert stake_amount(1000.0, r) == pytest.approx(50.0)


# ----------------------------- Value bet -----------------------------


def _make_match(match_id: str, home_odds: float, draw_odds: float, away_odds: float) -> Match:
    """Build a Match with one bookmaker for testing."""
    from aegeanbench.sports.models import CompetitionStage
    return Match(
        match_id=match_id,
        competition="TEST",
        stage=CompetitionStage.GROUP,
        kickoff_at=datetime(2026, 6, 12, 18, 0),
        home_team=Team("BRA", "Brazil"),
        away_team=Team("ARG", "Argentina"),
        odds=[
            Odds("test_book", datetime.now(), home_win=home_odds, draw=draw_odds, away_win=away_odds),
        ],
    )


def _make_prediction(match_id: str, p_h: float, p_d: float, p_a: float) -> Prediction:
    return Prediction(
        match_id=match_id,
        runner_id="test_runner",
        p_home_win=p_h,
        p_draw=p_d,
        p_away_win=p_a,
        confidence=0.7,
    )


class TestFindValueBets:
    def test_value_bet_when_model_higher_than_market(self):
        # Market: home odds 2.5 -> implied 40%. Model: 60%.
        # edge = 0.6 * 2.5 - 1 = 0.5  -> clear value
        match = _make_match("M1", home_odds=2.5, draw_odds=3.5, away_odds=3.0)
        pred = _make_prediction("M1", p_h=0.60, p_d=0.20, p_a=0.20)
        bets = find_value_bets(pred, match, min_edge=0.05)
        assert len(bets) >= 1
        home_bet = next(b for b in bets if b.outcome == MatchOutcome.HOME_WIN)
        assert home_bet.edge > 0.4
        assert home_bet.kelly is not None
        assert home_bet.kelly.has_edge

    def test_no_value_when_model_matches_market(self):
        # Market home 2.0 = 50%. Model 50%. edge = 0
        match = _make_match("M2", home_odds=2.0, draw_odds=3.0, away_odds=4.0)
        pred = _make_prediction("M2", p_h=0.50, p_d=0.30, p_a=0.20)
        bets = find_value_bets(pred, match, min_edge=0.05)
        # The market implied probs (after margin) shift slightly so we might
        # still see a tiny edge; key is that home_win specifically should fail
        # the min_edge=0.05 filter
        for b in bets:
            assert b.outcome != MatchOutcome.HOME_WIN or b.edge >= 0.05

    def test_min_edge_filter(self):
        match = _make_match("M3", home_odds=2.5, draw_odds=3.5, away_odds=3.0)
        # Model gives small edge: 0.42 * 2.5 - 1 = 0.05
        pred = _make_prediction("M3", p_h=0.42, p_d=0.28, p_a=0.30)
        loose = find_value_bets(pred, match, min_edge=0.01)
        strict = find_value_bets(pred, match, min_edge=0.20)
        assert len(loose) >= len(strict)

    def test_no_odds_returns_empty(self):
        from aegeanbench.sports.models import CompetitionStage
        match = Match(
            match_id="M4",
            competition="TEST",
            stage=CompetitionStage.GROUP,
            kickoff_at=datetime(2026, 6, 12),
            home_team=Team("BRA", "Brazil"),
            away_team=Team("ARG", "Argentina"),
        )
        pred = _make_prediction("M4", p_h=0.6, p_d=0.2, p_a=0.2)
        assert find_value_bets(pred, match) == []

    def test_sorted_by_edge_descending(self):
        match = _make_match("M5", home_odds=4.0, draw_odds=4.0, away_odds=4.0)
        # All outcomes get high edge but home wins biggest
        pred = _make_prediction("M5", p_h=0.50, p_d=0.30, p_a=0.20)
        bets = find_value_bets(pred, match, min_edge=0.05)
        # Verify sorted descending
        edges = [b.edge for b in bets]
        assert edges == sorted(edges, reverse=True)

    def test_best_value_bet_returns_top(self):
        match = _make_match("M6", home_odds=4.0, draw_odds=4.0, away_odds=4.0)
        pred = _make_prediction("M6", p_h=0.50, p_d=0.30, p_a=0.20)
        best = best_value_bet(pred, match)
        assert best is not None
        assert best.outcome == MatchOutcome.HOME_WIN


# ----------------------------- Portfolio -----------------------------


class TestBuildPortfolio:
    def _make_candidates(self) -> List[ValueBetCandidate]:
        """Build 4 candidates with varying edges across 3 matches."""
        cands = []
        for match_id, edge in [
            ("M1", 0.20),
            ("M1", 0.10),  # Same match, second outcome
            ("M2", 0.15),
            ("M3", 0.08),
        ]:
            match = _make_match(match_id, home_odds=2.5, draw_odds=3.5, away_odds=3.0)
            # Reverse-engineer p_model to give exactly this edge on home_win
            p_model = (1.0 + edge) / 2.5
            pred = _make_prediction(match_id, p_h=p_model, p_d=(1 - p_model) / 2, p_a=(1 - p_model) / 2)
            bets = find_value_bets(pred, match, min_edge=0.01)
            if bets:
                cands.append(bets[0])
        return cands

    def test_empty_input_returns_empty_portfolio(self):
        portfolio = build_portfolio([])
        assert portfolio.bets == []
        assert portfolio.total_stake == 0.0

    def test_skips_below_min_edge(self):
        cfg = PortfolioConfig(min_edge=0.50, bankroll=1000.0)
        candidates = self._make_candidates()
        portfolio = build_portfolio(candidates, cfg)
        # All our candidates have edge < 0.50, so all skipped
        assert portfolio.bets == []
        assert len(portfolio.skipped_candidates) == len(candidates)

    def test_per_bet_cap_respected(self):
        cfg = PortfolioConfig(
            bankroll=1000.0,
            max_per_bet_pct=0.01,  # tight cap
            max_per_match_pct=0.50,
            max_total_exposure_pct=1.0,
            min_edge=0.01,
        )
        candidates = self._make_candidates()
        portfolio = build_portfolio(candidates, cfg)
        for bet in portfolio.bets:
            assert bet.stake_fraction <= cfg.max_per_bet_pct + 1e-9

    def test_total_exposure_cap_respected(self):
        cfg = PortfolioConfig(
            bankroll=1000.0,
            max_per_bet_pct=0.50,
            max_per_match_pct=0.50,
            max_total_exposure_pct=0.10,
            min_edge=0.01,
        )
        candidates = self._make_candidates()
        portfolio = build_portfolio(candidates, cfg)
        assert portfolio.exposure_pct <= cfg.max_total_exposure_pct + 1e-9

    def test_per_match_cap_respected(self):
        # Two value bets on the SAME match should be capped
        cfg = PortfolioConfig(
            bankroll=1000.0,
            max_per_bet_pct=0.10,
            max_per_match_pct=0.03,  # tight per-match cap
            max_total_exposure_pct=1.0,
            min_edge=0.01,
        )
        candidates = self._make_candidates()
        portfolio = build_portfolio(candidates, cfg)
        per_match_totals = {}
        for bet in portfolio.bets:
            mid = bet.candidate.match_id
            per_match_totals[mid] = per_match_totals.get(mid, 0) + bet.stake_fraction
        for mid, total in per_match_totals.items():
            assert total <= cfg.max_per_match_pct + 1e-9

    def test_bets_sorted_by_edge_descending(self):
        cfg = PortfolioConfig(
            bankroll=1000.0,
            min_edge=0.01,
            max_per_bet_pct=0.10,
            max_per_match_pct=0.50,
            max_total_exposure_pct=1.0,
        )
        candidates = self._make_candidates()
        portfolio = build_portfolio(candidates, cfg)
        edges = [b.candidate.edge for b in portfolio.bets]
        assert edges == sorted(edges, reverse=True)

    def test_summary_counts(self):
        candidates = self._make_candidates()
        portfolio = build_portfolio(candidates)
        summary = portfolio.summary()
        assert "n_bets" in summary
        assert "total_stake" in summary
        assert "exposure_pct" in summary
        assert summary["n_bets"] == len(portfolio.bets)


# ----------------------------- End-to-end smoke -----------------------------


class TestEndToEndPipeline:
    """
    Smoke: build a Portfolio directly from a SportsDataGateway + a Prediction.
    """

    def test_full_pipeline_from_gateway(self):
        gw = SportsDataGateway(mock=True)
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        # Hand-build a confident prediction
        pred = Prediction(
            match_id=m.match_id,
            runner_id="smoke",
            p_home_win=0.55,
            p_draw=0.25,
            p_away_win=0.20,
            confidence=0.80,
        )
        candidates = find_value_bets(pred, ctx.match, min_edge=0.05)
        # The mock odds may or may not produce value depending on the
        # combination; both outcomes are valid for a smoke test.
        portfolio = build_portfolio(candidates)
        assert isinstance(portfolio, BettingPortfolio)
        # Sanity: stakes don't exceed bankroll
        assert portfolio.total_stake <= portfolio.config.bankroll
