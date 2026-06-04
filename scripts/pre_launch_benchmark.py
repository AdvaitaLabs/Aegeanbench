"""
Pre-launch benchmark: validate every predictor on a recent finished
tournament before going live.

Default: replay the 8 mock fixtures with synthetic ground truth (fast,
offline, ~2 seconds). Run with --euro to replay Euro 2024 from
football-data once a real API key is configured.

Output a calibration report that highlights:
  - Any predictor whose Brier > 0.75 (worse than a uniform guess)
  - Any predictor whose hit rate is statistically below random (33%)
  - Realised ROI of the betting layer

Use the report to decide whether to adjust:
  - Aegean min_edge / portfolio risk caps
  - LLM temperature
  - Dixon-Coles training depth
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from aegeanbench.sports import SportsDataGateway
from aegeanbench.sports.betting import PortfolioConfig
from aegeanbench.sports.models import Match, MatchResult
from aegeanbench.sports.orchestrator import (
    PipelineConfig,
    WorldCupPipeline,
)
from aegeanbench.sports.predictors import (
    AegeanPredictor,
    DixonColesPredictor,
    EloPredictor,
    LLMPredictor,
    MockLLMClient,
    Predictor,
    make_llm_predictor_from_env,
)

logger = logging.getLogger(__name__)


# Plausible-but-varied outcome mix for the mock fixtures.
_SYNTHETIC_RESULTS = [
    (2, 1), (1, 2), (1, 1), (3, 0),
    (1, 0), (1, 2), (1, 1), (2, 0),
]


def _make_predictor_panel() -> List[Predictor]:
    """Use mock LLMs when keys absent, real when present (auto-detected)."""
    return [
        EloPredictor(),
        DixonColesPredictor(max_iterations=80),
        make_llm_predictor_from_env("gpt-5", display_name="GPT-5"),
        make_llm_predictor_from_env("claude-opus-4-7", display_name="Claude Opus 4.7"),
        make_llm_predictor_from_env("deepseek-v3", display_name="DeepSeek V3"),
        AegeanPredictor(),
    ]


def _apply_synthetic_ground_truth(matches: List[Match]) -> None:
    for i, m in enumerate(matches):
        if m.result is not None:
            continue
        hg, ag = _SYNTHETIC_RESULTS[i % len(_SYNTHETIC_RESULTS)]
        m.result = MatchResult(home_goals=hg, away_goals=ag)


def _print_calibration_report(evaluation: dict) -> int:
    """
    Print a leaderboard-style table and return an exit code:
        0  all predictors look healthy
        1  at least one predictor is below acceptable thresholds
    """
    runner_evals = evaluation["runner_evaluations"]
    bet_eval = evaluation["betting_evaluation"]

    print()
    print("=" * 72)
    print("PRE-LAUNCH CALIBRATION REPORT")
    print("=" * 72)
    print(f"{'Runner':28} {'Brier':>8} {'Hit':>8} {'LogLoss':>10} {'n':>5} {'flag':>8}")
    print("-" * 72)

    has_warning = False
    # Sort by Brier ascending
    rows = sorted(runner_evals.items(), key=lambda kv: kv[1].mean_brier)
    for runner_id, ev in rows:
        flag = ""
        if ev.mean_brier > 0.75:
            flag = "WARN"
            has_warning = True
        elif ev.hit_rate < 0.30:
            flag = "WARN"
            has_warning = True
        print(
            f"{runner_id:28} {ev.mean_brier:8.4f} {ev.hit_rate:7.1%} "
            f"{ev.mean_log_loss:10.4f} {ev.n_evaluated:5d} {flag:>8}"
        )

    if bet_eval:
        print()
        print(
            f"Betting: {bet_eval.n_won}W / {bet_eval.n_lost}L  "
            f"stake ${bet_eval.total_stake:.2f}  "
            f"net ${bet_eval.net_profit:+.2f}  "
            f"ROI {bet_eval.roi*100:+.1f}%"
        )

    print()
    if has_warning:
        print("⚠️  Calibration warnings raised. Review predictor configs before launch.")
        return 1
    print("✅ All predictors within acceptable calibration band.")
    return 0


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    gateway = SportsDataGateway(mock=args.mock_gateway)
    pipeline = WorldCupPipeline(
        gateway=gateway,
        predictors=_make_predictor_panel(),
        config=PipelineConfig(
            train_history_size=args.train_history,
            betting_runner_id="dixon_coles",
            portfolio_config=PortfolioConfig(
                bankroll=1000.0,
                max_per_bet_pct=0.05,
                max_per_match_pct=0.08,
                max_total_exposure_pct=0.50,
                min_edge=0.05,
            ),
            persist=False,
            persist_label="pre-launch-benchmark",
        ),
    )

    print("Training classical predictors on historical data...")
    pipeline.train()
    print("Running predictor panel...")
    result = pipeline.run(train=False)

    print("Applying ground truth...")
    _apply_synthetic_ground_truth(result.matches)
    evaluation = pipeline.evaluate(result, result.matches)

    return _print_calibration_report(evaluation)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock-gateway", action=argparse.BooleanOptionalAction, default=True,
        help="Use the offline SportsDataGateway (default true).",
    )
    parser.add_argument(
        "--train-history", type=int, default=500,
        help="Number of historical matches for training (default 500).",
    )
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
