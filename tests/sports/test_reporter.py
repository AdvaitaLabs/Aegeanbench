"""
Tests for the Day 6 reporter: endpoint shapes + end-to-end build.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from aegeanbench.sports import (
    Match,
    MatchOutcome,
    MatchResult,
    Prediction,
    SportsDataGateway,
)
from aegeanbench.sports.betting import PortfolioConfig
from aegeanbench.sports.orchestrator import (
    PipelineConfig,
    WorldCupPipeline,
    save_run,
)
from aegeanbench.sports.predictors import (
    AegeanPredictor,
    DixonColesPredictor,
    EloPredictor,
)
from aegeanbench.sports.reporter import (
    RUNNER_REGISTRY,
    build_all_endpoints,
    build_leaderboard_endpoint,
    build_match_detail_endpoint,
    build_run_endpoint,
    build_runner_card_endpoint,
    build_runners_endpoint,
    build_tournaments_endpoint,
)


# ----------------------------- static endpoints -----------------------------


class TestStaticEndpoints:
    def test_runners_endpoint_lists_all_registered(self):
        out = build_runners_endpoint()
        assert "runners" in out
        ids = [r["id"] for r in out["runners"]]
        assert "elo" in ids
        assert "dixon_coles" in ids
        assert "aegean" in ids

    def test_tournaments_endpoint_includes_world_cup(self):
        out = build_tournaments_endpoint()
        ids = [t["id"] for t in out["tournaments"]]
        assert "fifa-world-cup-2026" in ids


# ----------------------------- leaderboard -----------------------------


class TestLeaderboardEndpoint:
    def _mk_run_with_eval(self, runner_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build a minimal run bundle with evaluation data attached."""
        runner_evals = {}
        for rid, m in runner_metrics.items():
            runner_evals[rid] = {
                "runner_id": rid,
                "n_evaluated": m["n"],
                "n_skipped_no_result": 0,
                "mean_brier": m["brier"],
                "mean_log_loss": m.get("log_loss", 0.5),
                "hit_rate": m["hit_rate"],
                "mean_confidence": 0.6,
                "per_match": [],
            }
        return {
            "manifest": {
                "run_id": "test_run",
                "competition": "FIFA World Cup 2026",
                "config": {"betting_runner_id": "aegean"},
            },
            "evaluation": {
                "evaluated_at": "2026-06-12T18:00:00",
                "runner_evaluations": runner_evals,
                "betting_evaluation": None,
            },
            "predictions": {},
        }

    def test_empty_runs_returns_empty_rows(self):
        out = build_leaderboard_endpoint([])
        assert out["rows"] == []

    def test_aggregates_metrics_across_runs(self):
        run_a = self._mk_run_with_eval({
            "elo": {"n": 10, "brier": 0.50, "hit_rate": 0.50},
            "aegean": {"n": 10, "brier": 0.40, "hit_rate": 0.60},
        })
        run_b = self._mk_run_with_eval({
            "elo": {"n": 10, "brier": 0.55, "hit_rate": 0.45},
            "aegean": {"n": 10, "brier": 0.42, "hit_rate": 0.58},
        })
        out = build_leaderboard_endpoint([run_a, run_b])
        rows = {r["runner_id"]: r for r in out["rows"]}
        # 20 matches each, weighted brier should be the mean of two equal-sized batches
        assert rows["elo"]["n_matches"] == 20
        assert rows["aegean"]["n_matches"] == 20
        assert rows["elo"]["mean_brier_score"] == pytest.approx((0.50 + 0.55) / 2, abs=1e-4)
        assert rows["aegean"]["mean_brier_score"] == pytest.approx((0.40 + 0.42) / 2, abs=1e-4)

    def test_rows_sorted_descending_by_composite(self):
        # Aegean wins on every metric -> ranks first
        run = self._mk_run_with_eval({
            "elo": {"n": 10, "brier": 0.60, "hit_rate": 0.40},
            "aegean": {"n": 10, "brier": 0.30, "hit_rate": 0.70},
        })
        out = build_leaderboard_endpoint([run])
        composite_scores = [r["composite_score"] for r in out["rows"]]
        assert composite_scores == sorted(composite_scores, reverse=True)
        assert out["rows"][0]["runner_id"] == "aegean"
        assert out["rows"][0]["rank"] == 1


# ----------------------------- run + match drill -----------------------------


class TestRunAndMatchEndpoints:
    @pytest.fixture
    def pipeline_run(self, tmp_path):
        gateway = SportsDataGateway(mock=True)
        predictors = [EloPredictor(), AegeanPredictor(mock=True)]
        cfg = PipelineConfig(
            runs_dir=tmp_path / "runs",
            persist=True,
            portfolio_config=None,
        )
        pipeline = WorldCupPipeline(gateway, predictors, cfg)
        result = pipeline.run()
        # Need ground truth for evaluation to be meaningful
        for m in result.matches:
            m.result = MatchResult(home_goals=2, away_goals=1)
        pipeline.evaluate(result, matches_with_ground_truth=result.matches)
        return result

    def test_run_endpoint_includes_all_fields(self, pipeline_run):
        from aegeanbench.sports.orchestrator.persistence import load_run
        bundle = load_run(pipeline_run.run_dir)
        out = build_run_endpoint(bundle)
        assert out["run_id"]
        assert "predictions" in out
        assert "evaluation" in out

    def test_match_detail_includes_all_predictors(self, pipeline_run):
        from aegeanbench.sports.orchestrator.persistence import load_run
        bundle = load_run(pipeline_run.run_dir)
        first_match_id = pipeline_run.matches[0].match_id
        out = build_match_detail_endpoint(bundle, first_match_id)
        assert out is not None
        runner_ids = {pp["runner_id"] for pp in out["predictions_per_predictor"]}
        assert runner_ids == {"elo", "aegean"}
        assert out["match_id"] == first_match_id

    def test_match_detail_unknown_id_returns_none(self, pipeline_run):
        from aegeanbench.sports.orchestrator.persistence import load_run
        bundle = load_run(pipeline_run.run_dir)
        assert build_match_detail_endpoint(bundle, "DOES_NOT_EXIST") is None


# ----------------------------- runner card -----------------------------


class TestRunnerCardEndpoint:
    def test_empty_runs_returns_metadata_only(self):
        out = build_runner_card_endpoint([], "elo")
        assert out["runner_id"] == "elo"
        assert out["metadata"]["name"] == "Elo Ratings"
        assert out["lifetime"]["n_matches_evaluated"] == 0
        assert out["history"] == []

    def test_aggregates_history(self):
        run = {
            "manifest": {"run_id": "r1"},
            "evaluation": {
                "evaluated_at": "2026-06-12T18:00:00",
                "runner_evaluations": {
                    "elo": {
                        "n_evaluated": 10,
                        "mean_brier": 0.50,
                        "mean_log_loss": 1.0,
                        "hit_rate": 0.60,
                        "mean_confidence": 0.55,
                        "per_match": [],
                    }
                },
            },
            "predictions": {"elo": [{"match_id": "M1", "p_home_win": 0.5,
                                      "p_draw": 0.3, "p_away_win": 0.2,
                                      "rationale": "test"}]},
        }
        out = build_runner_card_endpoint([run], "elo")
        assert out["lifetime"]["n_matches_evaluated"] == 10
        assert out["lifetime"]["mean_brier"] == 0.50
        assert len(out["sample_predictions"]) == 1


# ----------------------------- end-to-end build -----------------------------


class TestBuildAllEndpoints:
    def test_writes_expected_files(self, tmp_path):
        runs_dir = tmp_path / "runs"
        out_dir = tmp_path / "out"

        # Seed two persisted runs so the leaderboard has data
        save_run(
            predictions_by_runner={
                "elo": [Prediction("M1", "elo", 0.5, 0.3, 0.2)],
                "aegean": [Prediction("M1", "aegean", 0.45, 0.30, 0.25)],
            },
            portfolio=None,
            runs_dir=runs_dir,
            label="r1",
        )
        save_run(
            predictions_by_runner={
                "elo": [Prediction("M2", "elo", 0.4, 0.3, 0.3)],
                "aegean": [Prediction("M2", "aegean", 0.42, 0.29, 0.29)],
            },
            portfolio=None,
            runs_dir=runs_dir,
            label="r2",
        )

        counts = build_all_endpoints(output_dir=out_dir, runs_dir=runs_dir)

        # All static files exist
        assert (out_dir / "runners.json").exists()
        assert (out_dir / "tournaments.json").exists()
        assert (out_dir / "leaderboard.json").exists()
        # Per-run JSONs
        assert counts["runs"] == 2
        assert len(list((out_dir / "runs").glob("*.json"))) == 2
        # Per-match drill-downs
        assert counts["match_details"] >= 2  # one per (run, match) pair
        # Per-runner cards (one for every registered runner)
        assert counts["runner_cards"] == len(RUNNER_REGISTRY)

    def test_runners_json_well_formed(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        out_dir = tmp_path / "out"
        build_all_endpoints(output_dir=out_dir, runs_dir=runs_dir)
        with (out_dir / "runners.json").open() as f:
            data = json.load(f)
        assert "_meta" in data
        assert "runners" in data
        ids = {r["id"] for r in data["runners"]}
        assert {"elo", "dixon_coles", "aegean"}.issubset(ids)


# ----------------------------- optional server (smoke) -----------------------------


class TestServerCreateApp:
    def test_create_app_works_when_fastapi_available(self, tmp_path):
        """Skips silently if fastapi isn't installed; verifies app builds."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("fastapi not installed in this env")
        from aegeanbench.sports.reporter.server import create_app
        app = create_app(runs_dir=tmp_path)
        # Sanity check the route table
        routes = {r.path for r in app.router.routes}
        assert "/api/v1/runners" in routes
        assert "/api/v1/leaderboard" in routes
        assert "/api/v1/runners/{runner_id}/card" in routes
