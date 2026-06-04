"""
Command-line entry point for the World Cup pipeline.

Usage:
    python -m aegeanbench.sports.cli predict
    python -m aegeanbench.sports.cli predict --runners elo,dixon-coles,aegean
    python -m aegeanbench.sports.cli list-runs
    python -m aegeanbench.sports.cli evaluate <run_id>

Runs are stored under ~/.aegeanbench/worldcup_runs/. The CLI is intentionally
thin: it wires command-line flags into PipelineConfig and a list of
predictors, then invokes WorldCupPipeline.

For the sprint, predictors default to the mock-friendly set
(elo + dixon-coles + aegean-mock). Real LLM predictors get added once
API keys are configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence

from aegeanbench.sports import SportsDataGateway
from aegeanbench.sports.betting import PortfolioConfig
from aegeanbench.sports.orchestrator import (
    DEFAULT_RUNS_DIR,
    PipelineConfig,
    WorldCupPipeline,
    list_runs,
    load_run,
)
from aegeanbench.sports.predictors import (
    AegeanPredictor,
    DixonColesPredictor,
    EloPredictor,
    LLMPredictor,
    MockLLMClient,
    Predictor,
)

logger = logging.getLogger(__name__)


# Predictor registry keyed by CLI-friendly short name
def _make_predictor(name: str) -> Predictor:
    name = name.lower().strip()
    if name == "elo":
        return EloPredictor()
    if name in ("dc", "dixon-coles", "dixon_coles"):
        return DixonColesPredictor(max_iterations=80)
    if name == "aegean":
        return AegeanPredictor(mock=True)
    if name == "gpt-5":
        return LLMPredictor(MockLLMClient("gpt-5"), runner_id="gpt-5", display_name="GPT-5")
    if name in ("claude", "claude-opus-4-7"):
        return LLMPredictor(
            MockLLMClient("claude-opus-4-7"),
            runner_id="claude-opus-4-7",
            display_name="Claude Opus 4.7",
        )
    if name in ("deepseek", "deepseek-v3"):
        return LLMPredictor(
            MockLLMClient("deepseek-v3"),
            runner_id="deepseek-v3",
            display_name="DeepSeek V3",
        )
    raise ValueError(f"Unknown predictor: {name!r}")


DEFAULT_RUNNERS = "elo,dixon-coles,aegean"


# ---------------------- subcommands ----------------------


def cmd_predict(args: argparse.Namespace) -> int:
    """Run the full pipeline and persist a new run."""
    runner_names = [n.strip() for n in args.runners.split(",") if n.strip()]
    predictors = [_make_predictor(n) for n in runner_names]
    gateway = SportsDataGateway(mock=args.mock)

    portfolio_cfg = (
        PortfolioConfig(
            bankroll=args.bankroll,
            max_per_bet_pct=args.max_per_bet,
            max_per_match_pct=args.max_per_match,
            max_total_exposure_pct=args.max_total_exposure,
            min_edge=args.min_edge,
        )
        if args.bet
        else None
    )

    cfg = PipelineConfig(
        competition=args.competition,
        train_history_size=args.train_history,
        betting_runner_id=args.betting_runner,
        portfolio_config=portfolio_cfg,
        persist=args.persist,
        persist_label=args.label,
    )

    pipeline = WorldCupPipeline(gateway, predictors, cfg)
    result = pipeline.run()

    print(f"Run dir: {result.run_dir}")
    print(f"Predictors: {result.runner_ids}")
    for runner_id, preds in result.predictions_by_runner.items():
        print(f"  {runner_id}: {len(preds)} predictions")
    if result.portfolio is not None:
        s = result.portfolio.summary()
        print(
            f"Portfolio: {s['n_bets']} bets across {s['n_matches']} matches | "
            f"stake ${s['total_stake']:.2f} | exposure {s['exposure_pct']*100:.1f}% | "
            f"E[return] ${s['total_expected_return']:.2f}"
        )
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    """List all known runs."""
    runs = list_runs()
    if not runs:
        print("(no runs found)")
        return 0
    for m in runs[: args.limit]:
        print(
            f"{m['run_id']:50}  {m.get('created_at', '?'):26}  "
            f"matches={m.get('n_matches', '?')}  runners={len(m.get('runner_ids', []))}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show the contents of one run."""
    run_dir = Path(args.run_dir) if "/" in args.run_id else DEFAULT_RUNS_DIR / args.run_id
    bundle = load_run(run_dir)
    print(json.dumps(bundle, indent=2, ensure_ascii=False, default=str))
    return 0


# ---------------------- parser ----------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aegeanbench-worldcup", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    pred = sub.add_parser("predict", help="Run the prediction + betting pipeline")
    pred.add_argument("--runners", default=DEFAULT_RUNNERS,
                      help=f"Comma-separated predictor names. Default: {DEFAULT_RUNNERS}")
    pred.add_argument("--competition", default="FIFA World Cup 2026")
    pred.add_argument("--mock", action=argparse.BooleanOptionalAction, default=True,
                      help="Use mock data sources (default: True)")
    pred.add_argument("--train-history", type=int, default=500,
                      help="Number of historical matches for training (default: 500)")
    pred.add_argument("--bet", action=argparse.BooleanOptionalAction, default=True,
                      help="Build a betting portfolio (default: True)")
    pred.add_argument("--betting-runner", default="aegean",
                      help="Which runner's predictions to bet on (default: aegean)")
    pred.add_argument("--bankroll", type=float, default=1000.0)
    pred.add_argument("--max-per-bet", type=float, default=0.05)
    pred.add_argument("--max-per-match", type=float, default=0.08)
    pred.add_argument("--max-total-exposure", type=float, default=0.50)
    pred.add_argument("--min-edge", type=float, default=0.05)
    pred.add_argument("--persist", action=argparse.BooleanOptionalAction, default=True)
    pred.add_argument("--label", default=None)
    pred.set_defaults(func=cmd_predict)

    lst = sub.add_parser("list-runs", help="List all persisted runs")
    lst.add_argument("--limit", type=int, default=20)
    lst.set_defaults(func=cmd_list_runs)

    show = sub.add_parser("show", help="Dump one run's JSON")
    show.add_argument("run_id", help="Run ID or path")
    show.add_argument("--run-dir", default=None)
    show.set_defaults(func=cmd_show)

    return p


def main(argv: Sequence[str] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
