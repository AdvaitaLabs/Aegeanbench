"""
Tests for the Day 5 orchestrator: pipeline, evaluator, persistence.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from aegeanbench.sports import (
    Match,
    MatchOutcome,
    MatchResult,
    Prediction,
    SportsDataGateway,
)
from aegeanbench.sports.betting import (
    PortfolioConfig,
    SizedBet,
    ValueBetCandidate,
)
from aegeanbench.sports.orchestrator import (
    PipelineConfig,
    WorldCupPipeline,
    evaluate_betting,
    evaluate_runner,
    evaluate_runners,
    list_runs,
    load_run,
    save_run,
)
from aegeanbench.sports.predictors import (
    AegeanPredictor,
    DixonColesPredictor,
    EloPredictor,
)


# ----------------------------- Evaluator -----------------------------


class TestEvaluator:
    def _mk_match(self, match_id: str, outcome: MatchOutcome) -> Match:
        from aegeanbench.sports.models import CompetitionStage, Team
        # Encode outcome via goals
        if outcome == MatchOutcome.HOME_WIN:
            hg, ag = 2, 1
        elif outcome == MatchOutcome.AWAY_WIN:
            hg, ag = 1, 2
        else:
            hg, ag = 1, 1
        return Match(
            match_id=match_id,
            competition="TEST",
            stage=CompetitionStage.GROUP,
            kickoff_at=datetime(2026, 6, 12),
            home_team=Team("BRA", "Brazil"),
            away_team=Team("ARG", "Argentina"),
            result=MatchResult(home_goals=hg, away_goals=ag),
        )

    def test_perfect_prediction_brier_zero(self):
        """A perfect (1.0, 0, 0) prediction on HOME_WIN should yield brier=0."""
        match = self._mk_match("M1", MatchOutcome.HOME_WIN)
        pred = Prediction(
            match_id="M1", runner_id="oracle",
            p_home_win=1.0, p_draw=0.0, p_away_win=0.0, confidence=1.0,
        )
        eva = evaluate_runner([pred], [match])
        assert eva.mean_brier == pytest.approx(0.0)
        assert eva.hit_rate == 1.0
        assert eva.mean_log_loss < 1e-6

    def test_uniform_prediction_brier_known_value(self):
        """A uniform (1/3) prediction has brier = 2/3."""
        match = self._mk_match("M1", MatchOutcome.HOME_WIN)
        pred = Prediction(
            match_id="M1", runner_id="random",
            p_home_win=1/3, p_draw=1/3, p_away_win=1/3,
        )
        eva = evaluate_runner([pred], [match])
        # (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
        assert eva.mean_brier == pytest.approx(2 / 3)
        # Tied argmax falls back to first enum value (HOME_WIN), so a uniform
        # prediction "correctly" predicts HOME_WIN matches by accident.
        assert eva.hit_rate == 1.0

    def test_wrong_prediction_high_brier(self):
        """A confidently wrong prediction yields brier ~ 2."""
        match = self._mk_match("M1", MatchOutcome.HOME_WIN)
        pred = Prediction(
            match_id="M1", runner_id="bad",
            p_home_win=0.0, p_draw=0.0, p_away_win=1.0,
        )
        eva = evaluate_runner([pred], [match])
        assert eva.mean_brier == pytest.approx(2.0)
        assert eva.hit_rate == 0.0

    def test_skips_matches_without_result(self):
        from aegeanbench.sports.models import CompetitionStage, Team
        pending_match = Match(
            match_id="PENDING",
            competition="TEST",
            stage=CompetitionStage.GROUP,
            kickoff_at=datetime(2027, 6, 12),
            home_team=Team("BRA", "Brazil"),
            away_team=Team("ARG", "Argentina"),
        )
        played = self._mk_match("PLAYED", MatchOutcome.DRAW)
        preds = [
            Prediction(match_id="PENDING", runner_id="r",
                       p_home_win=0.4, p_draw=0.3, p_away_win=0.3),
            Prediction(match_id="PLAYED", runner_id="r",
                       p_home_win=0.4, p_draw=0.3, p_away_win=0.3),
        ]
        eva = evaluate_runner(preds, [pending_match, played])
        assert eva.n_evaluated == 1
        assert eva.n_skipped_no_result == 1

    def test_evaluate_runners_multi(self):
        match = self._mk_match("M1", MatchOutcome.HOME_WIN)
        preds_a = [Prediction("M1", "rA", 0.6, 0.2, 0.2)]
        preds_b = [Prediction("M1", "rB", 0.3, 0.3, 0.4)]
        results = evaluate_runners({"rA": preds_a, "rB": preds_b}, [match])
        assert results["rA"].hit_rate == 1.0
        assert results["rB"].hit_rate == 0.0
        assert results["rA"].mean_brier < results["rB"].mean_brier


# ----------------------------- Betting Evaluation -----------------------------


class TestBettingEvaluation:
    def test_winning_bet_returns_stake_times_odds(self):
        from aegeanbench.sports.models import CompetitionStage, Team
        match = Match(
            match_id="M1", competition="TEST",
            stage=CompetitionStage.GROUP,
            kickoff_at=datetime(2026, 6, 12),
            home_team=Team("BRA", "Brazil"),
            away_team=Team("ARG", "Argentina"),
            result=MatchResult(home_goals=2, away_goals=1),  # HOME_WIN
        )
        candidate = ValueBetCandidate(
            match_id="M1", runner_id="r",
            outcome=MatchOutcome.HOME_WIN,
            p_model=0.6, p_market=0.4, decimal_odds=2.5, edge=0.5,
        )
        bet = SizedBet(
            candidate=candidate,
            stake_amount=100.0, stake_fraction=0.1, expected_return=50.0,
        )
        eva = evaluate_betting([bet], [match])
        assert eva.n_won == 1
        assert eva.n_lost == 0
        assert eva.total_stake == 100.0
        assert eva.total_returned == 250.0  # 100 * 2.5
        assert eva.net_profit == 150.0
        assert eva.roi == 1.5

    def test_losing_bet_returns_nothing(self):
        from aegeanbench.sports.models import CompetitionStage, Team
        match = Match(
            match_id="M1", competition="TEST",
            stage=CompetitionStage.GROUP,
            kickoff_at=datetime(2026, 6, 12),
            home_team=Team("BRA", "Brazil"),
            away_team=Team("ARG", "Argentina"),
            result=MatchResult(home_goals=1, away_goals=2),  # AWAY_WIN
        )
        candidate = ValueBetCandidate(
            match_id="M1", runner_id="r",
            outcome=MatchOutcome.HOME_WIN,
            p_model=0.6, p_market=0.4, decimal_odds=2.5, edge=0.5,
        )
        bet = SizedBet(candidate=candidate, stake_amount=100.0, stake_fraction=0.1, expected_return=50.0)
        eva = evaluate_betting([bet], [match])
        assert eva.n_won == 0
        assert eva.n_lost == 1
        assert eva.net_profit == -100.0
        assert eva.roi == -1.0

    def test_skips_bets_on_unfinished_matches(self):
        from aegeanbench.sports.models import CompetitionStage, Team
        match = Match(
            match_id="M1", competition="TEST",
            stage=CompetitionStage.GROUP,
            kickoff_at=datetime(2027, 6, 12),
            home_team=Team("BRA", "Brazil"),
            away_team=Team("ARG", "Argentina"),
        )
        candidate = ValueBetCandidate(
            match_id="M1", runner_id="r",
            outcome=MatchOutcome.HOME_WIN,
            p_model=0.6, p_market=0.4, decimal_odds=2.5, edge=0.5,
        )
        bet = SizedBet(candidate=candidate, stake_amount=100.0, stake_fraction=0.1, expected_return=50.0)
        eva = evaluate_betting([bet], [match])
        assert eva.n_skipped_no_result == 1
        assert eva.n_won == 0 and eva.n_lost == 0


# ----------------------------- Persistence -----------------------------


class TestPersistence:
    def test_save_and_load_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            predictions = {
                "elo": [
                    Prediction("M1", "elo", 0.5, 0.3, 0.2),
                    Prediction("M2", "elo", 0.4, 0.3, 0.3),
                ],
            }
            run_dir = save_run(
                predictions_by_runner=predictions,
                portfolio=None,
                label="unit-test",
                runs_dir=runs_dir,
            )
            assert run_dir.exists()
            assert (run_dir / "manifest.json").exists()
            assert (run_dir / "predictions.json").exists()

            bundle = load_run(run_dir)
            assert bundle["manifest"]["label"] == "unit-test"
            assert "predictions" in bundle
            assert len(bundle["predictions"]["elo"]) == 2

    def test_list_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            save_run({"elo": []}, runs_dir=runs_dir, label="first")
            save_run({"elo": []}, runs_dir=runs_dir, label="second")
            runs = list_runs(runs_dir)
            assert len(runs) == 2


# ----------------------------- Pipeline -----------------------------


class TestWorldCupPipeline:
    def test_pipeline_basic_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = SportsDataGateway(mock=True)
            predictors = [EloPredictor(), AegeanPredictor(mock=True)]
            cfg = PipelineConfig(
                runs_dir=Path(tmp),
                portfolio_config=None,
                persist=True,
                persist_label="pipeline-test",
            )
            pipeline = WorldCupPipeline(gateway, predictors, cfg)
            result = pipeline.run()
            assert result.run_dir is not None
            assert result.run_dir.exists()
            assert set(result.runner_ids) == {"elo", "aegean"}
            for runner_id, preds in result.predictions_by_runner.items():
                assert len(preds) > 0

    def test_pipeline_with_portfolio(self):
        with tempfile.TemporaryDirectory() as tmp:
            gateway = SportsDataGateway(mock=True)
            predictors = [
                DixonColesPredictor(max_iterations=30),
                AegeanPredictor(mock=True),
            ]
            cfg = PipelineConfig(
                runs_dir=Path(tmp),
                betting_runner_id="dixon_coles",
                portfolio_config=PortfolioConfig(
                    bankroll=1000.0, min_edge=0.05,
                ),
                persist=True,
            )
            pipeline = WorldCupPipeline(gateway, predictors, cfg)
            result = pipeline.run()
            assert result.portfolio is not None

    def test_pipeline_train_idempotent(self):
        """Calling train() twice or run(train=True) twice should not crash."""
        gateway = SportsDataGateway(mock=True)
        pipeline = WorldCupPipeline(
            gateway, [EloPredictor()],
            PipelineConfig(persist=False, portfolio_config=None),
        )
        pipeline.train()
        pipeline.train()
        pipeline.run(train=False)
        pipeline.run(train=True)

    def test_pipeline_evaluate(self):
        """End-to-end: run pipeline, fake ground truth, evaluate."""
        with tempfile.TemporaryDirectory() as tmp:
            gateway = SportsDataGateway(mock=True)
            predictors = [EloPredictor()]
            cfg = PipelineConfig(
                runs_dir=Path(tmp),
                portfolio_config=None,
                persist=True,
            )
            pipeline = WorldCupPipeline(gateway, predictors, cfg)
            result = pipeline.run()
            # Fake ground truth: every match was a home win
            for m in result.matches:
                m.result = MatchResult(home_goals=2, away_goals=0)
            evaluation = pipeline.evaluate(result, matches_with_ground_truth=result.matches)
            assert "runner_evaluations" in evaluation
            runner_evals = evaluation["runner_evaluations"]
            assert "elo" in runner_evals
            # evaluation.json should now exist
            assert (result.run_dir / "evaluation.json").exists()


# ----------------------------- CLI smoke -----------------------------


class TestCLISmoke:
    """
    Lightweight smoke test - we don't shell out; we call main() with argv.
    """

    def test_cli_predict_no_persist(self, monkeypatch):
        # Force runs_dir to a temp directory so we don't pollute home
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "aegeanbench.sports.orchestrator.persistence.DEFAULT_RUNS_DIR",
                Path(tmp),
            )
            from aegeanbench.sports.cli import main
            rc = main([
                "predict", "--runners=elo,aegean", "--no-bet", "--no-persist",
            ])
            assert rc == 0

    def test_cli_list_runs_empty(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(
                "aegeanbench.sports.orchestrator.persistence.DEFAULT_RUNS_DIR",
                Path(tmp),
            )
            from aegeanbench.sports.cli import main
            rc = main(["list-runs"])
            assert rc == 0
