"""
End-to-end demo: World Cup pipeline from data fetch to product JSON.

Designed as a single self-contained script so demos to product / leadership
take one command:

    python -m aegeanbench.sports.demo

What it does (top to bottom):
    1. Initialise SportsDataGateway in mock mode (works offline).
    2. Seed team Elo from clubelo's fallback table.
    3. Train Elo + Dixon-Coles on 500 synthetic historical matches.
    4. Build a Predictor list: Elo, Dixon-Coles, 3 mock LLMs, Aegean.
    5. Run the pipeline over the 8 mock World Cup fixtures.
    6. Inject plausible ground truth (mix of H / D / A outcomes).
    7. Evaluate predictions vs ground truth -> Brier, hit rate, ROI.
    8. Render the 6 product JSON endpoints to disk.
    9. Print a digestible summary to stdout.

Outputs:
    Pipeline run:    ~/.aegeanbench/worldcup_runs/run_<ts>_<hash>/
    Reporter JSONs:  /tmp/aegean_demo_api/   (override via --output)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from aegeanbench.sports import SportsDataGateway
from aegeanbench.sports.betting import PortfolioConfig
from aegeanbench.sports.models import MatchResult
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
)
from aegeanbench.sports.reporter import build_all_endpoints
from aegeanbench.sports.sources import enrich_teams_with_elo, ClubEloAdapter


# Plausible mix of outcomes for the 8 mock fixtures: 4 home wins, 2 draws,
# 2 away wins. Maps directly onto match list ordering from the gateway.
_DEMO_GROUND_TRUTH = [
    (2, 1),  # H
    (0, 2),  # A
    (1, 1),  # D
    (3, 0),  # H
    (1, 0),  # H
    (1, 2),  # A
    (1, 1),  # D
    (2, 0),  # H
]


def _make_predictors() -> List[Predictor]:
    """The standard panel: 2 classical + 3 mock LLMs + 1 consensus."""
    return [
        EloPredictor(),
        DixonColesPredictor(max_iterations=60),
        LLMPredictor(MockLLMClient("gpt-5"), "gpt-5", "GPT-5"),
        LLMPredictor(MockLLMClient("claude-opus-4-7"), "claude-opus-4-7", "Claude Opus 4.7"),
        LLMPredictor(MockLLMClient("deepseek-v3"), "deepseek-v3", "DeepSeek V3"),
        AegeanPredictor(mock=True),
    ]


def _print_section(title: str) -> None:
    bar = "=" * len(title)
    print()
    print(bar)
    print(title)
    print(bar)


def run(output_dir: Path) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    _print_section("STEP 1: Gateway + free data sources")
    gateway = SportsDataGateway(mock=True)
    teams = gateway.list_teams()
    enrich_teams_with_elo(teams, adapter=ClubEloAdapter(mock=True))
    print(f"  Loaded {len(teams)} teams; Elo seeded from clubelo fallback table.")
    for t in teams[:4]:
        print(f"    {t.fifa_code} ({t.name}): Elo={t.elo_rating:.0f}")

    _print_section("STEP 2: Training classical predictors")
    pipeline_cfg = PipelineConfig(
        train_history_size=500,
        betting_runner_id="dixon_coles",
        portfolio_config=PortfolioConfig(
            bankroll=1000.0,
            max_per_bet_pct=0.05,
            max_per_match_pct=0.08,
            max_total_exposure_pct=0.50,
            min_edge=0.05,
        ),
        persist=True,
        persist_label="demo",
    )
    predictors = _make_predictors()
    pipeline = WorldCupPipeline(gateway, predictors, pipeline_cfg)
    pipeline.train()
    print(f"  Trained Elo + Dixon-Coles on 500 synthetic historical matches.")

    _print_section("STEP 3: Running predictions")
    result = pipeline.run(train=False)
    print(f"  Predictors run: {len(result.predictions_by_runner)}")
    for runner_id, preds in result.predictions_by_runner.items():
        first = preds[0] if preds else None
        if first:
            print(
                f"    {runner_id:24} match={first.match_id:10} "
                f"H={first.p_home_win:.3f} D={first.p_draw:.3f} A={first.p_away_win:.3f}"
            )

    _print_section("STEP 4: Betting portfolio")
    portfolio = result.portfolio
    if portfolio:
        summary = portfolio.summary()
        print(
            f"  {summary['n_bets']} bets / {summary['n_matches']} matches | "
            f"stake ${summary['total_stake']:.2f} ({summary['exposure_pct']*100:.1f}% exposure) | "
            f"E[return] ${summary['total_expected_return']:.2f}"
        )
        for bet in portfolio.bets[:3]:
            c = bet.candidate
            print(
                f"    {c.match_id:10} bet {c.outcome.value:9} "
                f"stake ${bet.stake_amount:6.2f}  edge {c.edge:+.3f}"
            )
    else:
        print("  (no portfolio built)")

    _print_section("STEP 5: Injecting ground truth + evaluating")
    for i, m in enumerate(result.matches):
        hg, ag = _DEMO_GROUND_TRUTH[i % len(_DEMO_GROUND_TRUTH)]
        m.result = MatchResult(home_goals=hg, away_goals=ag)
    evaluation = pipeline.evaluate(result, result.matches)

    print("  Predictor rankings (by mean Brier, lower = better):")
    runner_evals = evaluation["runner_evaluations"]
    rows = sorted(
        runner_evals.items(), key=lambda kv: kv[1].mean_brier
    )
    for runner_id, ev in rows:
        print(
            f"    {runner_id:24} brier={ev.mean_brier:.4f}  "
            f"hit={ev.hit_rate:.2%}  log_loss={ev.mean_log_loss:.4f}  "
            f"n={ev.n_evaluated}"
        )

    bet_eval = evaluation["betting_evaluation"]
    if bet_eval:
        print(
            f"\n  Betting realised: {bet_eval.n_won} won / {bet_eval.n_lost} lost | "
            f"net ${bet_eval.net_profit:+.2f}  ROI {bet_eval.roi*100:+.1f}%"
        )

    _print_section("STEP 6: Building product JSON endpoints")
    counts = build_all_endpoints(output_dir=output_dir)
    print(f"  Output: {output_dir}")
    for k, v in counts.items():
        print(f"    {k}: {v}")

    _print_section("DONE")
    print(f"  Pipeline run dir: {result.run_dir}")
    print(f"  Frontend JSONs:   {output_dir}")
    print()
    print("Next steps:")
    print(f"  - cat {output_dir}/leaderboard.json | jq")
    print(f"  - Spin up the live API:")
    print(f"      uvicorn aegeanbench.sports.reporter.server:app --port 8200")
    print(f"  - Or hand the JSON folder zip to the frontend team.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="/tmp/aegean_demo_api",
        help="Directory to write the 6 product JSONs into.",
    )
    args = parser.parse_args(argv)
    return run(Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
