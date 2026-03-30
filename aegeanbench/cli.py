#!/usr/bin/env python3
"""
AegeanBench CLI — run the benchmark suite from the command line.

Usage examples:

  # Run all cases with mock runner (no LLM needed)
  python -m aegeanbench.cli run

  # Run only consensus cases
  python -m aegeanbench.cli run --category consensus

  # Run only hard cases
  python -m aegeanbench.cli run --difficulty hard

  # Run against real aegean-consensus
  python -m aegeanbench.cli run --runner aegean

  # Show available cases
  python -m aegeanbench.cli list

  # Show a case detail
  python -m aegeanbench.cli show C-HARD-001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

from aegeanbench.core.models import (
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkSuiteResult,
    Difficulty,
)
from aegeanbench.datasets import load_full_suite
from aegeanbench.metrics.engine import MetricsEngine


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

COLORS = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "grey":   "\033[90m",
}


def c(color: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def print_banner() -> None:
    print(c("cyan", "\n" + "=" * 60))
    print(c("bold", "  AegeanBench v0.1.0"))
    print(c("grey", "  The first multi-agent consensus benchmark"))
    print(c("cyan", "=" * 60 + "\n"))


def print_result_line(result: BenchmarkResult) -> None:
    icon    = c("green", "PASS") if result.correct else c("red", "FAIL")
    latency = c("grey", f"{result.latency_s:.3f}s")
    diff    = c("yellow", result.difficulty.value.upper())
    cat     = c("cyan", result.category.value[:4].upper())
    name    = result.case_name[:48]
    print(f"  [{icon}] [{cat}] [{diff}] {name:<50} {latency}")


def print_suite_summary(suite_result: BenchmarkSuiteResult, engine: MetricsEngine,
                        results: List[BenchmarkResult]) -> None:
    print(c("cyan", "\n" + "─" * 60))
    print(c("bold", "  SUMMARY"))
    print(c("cyan", "─" * 60))

    total = suite_result.total_cases
    print(f"  Cases:    {suite_result.passed} passed / "
          f"{suite_result.failed} failed / "
          f"{suite_result.errored} errored / {total} total")
    print(f"  Accuracy: {c('bold', f'{suite_result.accuracy:.1%}')}")
    print(f"  Latency:  p50={suite_result.p50_latency_s:.3f}s  "
          f"p95={suite_result.p95_latency_s:.3f}s")

    # Consensus metrics
    con = [r for r in results if r.consensus_metrics]
    if con:
        print(c("cyan", "\n  [Consensus Metrics]"))
        print(f"  consensus_rate:      {suite_result.consensus_rate:.1%}")
        print(f"  mean_confidence:     {suite_result.mean_consensus_conf:.3f}")
        print(f"  disagreement_score:  {suite_result.mean_disagreement:.3f}")
        print(f"  early_stop_rate:     {suite_result.early_stop_rate:.1%}")
        print(f"  quorum_efficiency:   {suite_result.mean_quorum_efficiency:.3f}")
        print(f"  outlier_detect_rate: {suite_result.outlier_detection_rate:.1%}")

    # Collaboration metrics
    col = [r for r in results if r.collaboration_metrics]
    if col:
        print(c("cyan", "\n  [Collaboration Metrics]"))
        print(f"  subtask_completion:  {suite_result.mean_subtask_completion:.1%}")
        print(f"  milestone_hit:       {suite_result.mean_milestone_hit:.1%}")
        print(f"  coordination:        {suite_result.mean_coordination:.3f}")

    # Risk metrics
    risk = [r for r in results if r.risk_metrics]
    if risk:
        print(c("cyan", "\n  [Risk Assessment Metrics]"))
        print(f"  risk_accuracy:       {suite_result.risk_accuracy:.1%}")
        print(f"  risk_level_accuracy: {suite_result.risk_level_accuracy:.1%}")
        print(f"  f1_approve:          {suite_result.risk_f1_approve:.3f}")
        print(f"  f1_reject:           {suite_result.risk_f1_reject:.3f}")
        print(f"  validator_agreement: {suite_result.mean_validator_agreement:.3f}")
        print(f"  pre_screen_rate:     {suite_result.pre_screen_rate:.1%}")

    # By difficulty
    print(c("cyan", "\n  [By Difficulty]"))
    for diff, stats in engine.breakdown_by_difficulty(results).items():
        bar = int(stats["accuracy"] * 20)
        bar_str = "█" * bar + "░" * (20 - bar)
        print(f"  {diff.upper():<8}: [{bar_str}] {stats['accuracy']:.1%} "
              f"({stats['correct']}/{stats['total']})")

    # By category
    print(c("cyan", "\n  [By Category]"))
    for cat, stats in engine.breakdown_by_category(results).items():
        bar = int(stats["accuracy"] * 20)
        bar_str = "█" * bar + "░" * (20 - bar)
        print(f"  {cat.upper():<14}: [{bar_str}] {stats['accuracy']:.1%} "
              f"({stats['correct']}/{stats['total']})")

    print(c("cyan", "\n" + "─" * 60 + "\n"))


# ─────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> None:
    print_banner()

    suite = load_full_suite()
    cases = suite.cases

    # Filters
    if args.category:
        cases = [c for c in cases if c.category.value == args.category]
    if args.difficulty:
        cases = [c for c in cases if c.difficulty.value == args.difficulty]
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if args.case_id:
        cases = [c for c in cases if c.case_id == args.case_id]

    if not cases:
        print(c("yellow", "No cases matched the filters. Use `list` to see available cases."))
        return

    print(f"Running {c('bold', str(len(cases)))} cases with runner={c('cyan', args.runner)}\n")

    # Build runner
    if args.runner == "mock":
        from aegeanbench.runners.mock_runner import MockRunner
        runner = MockRunner(seed=args.seed)
    elif args.runner == "aegean":
        from aegeanbench.runners.aegean_runner import AegeanRunner
        runner = AegeanRunner(fallback_to_mock=True)
    elif args.runner == "http":
        from aegeanbench.runners.http_runner import HttpRunner
        try:
            runner = HttpRunner(base_url=args.http_url)
        except RuntimeError as e:
            print(c("red", f"Error: {e}"))
            return
    else:
        print(c("red", f"Unknown runner: {args.runner}"))
        return

    results: List[BenchmarkResult] = []
    start_total = time.perf_counter()

    for case in cases:
        result = runner.run_case(case)
        results.append(result)
        print_result_line(result)

    elapsed = time.perf_counter() - start_total
    print(c("grey", f"\n  Total wall time: {elapsed:.2f}s  |  "
               f"tokens: {sum(r.tokens_total for r in results):,}"))

    engine       = MetricsEngine()
    suite_result = engine.aggregate(suite.suite_id, suite.name, results)
    suite_result.results = results

    print_suite_summary(suite_result, engine, results)
    _print_token_summary(suite_result)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path  = os.path.join(args.output_dir, f"run_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(
            suite_result.model_dump(mode="json"),
            f, indent=2, default=str,
        )
    print(c("grey", f"  Results saved: {out_path}\n"))


def _print_token_summary(suite_result: BenchmarkSuiteResult) -> None:
    print(c("cyan", "\n  [Token Consumption]"))
    print(f"  total_tokens:          {suite_result.total_tokens:,}")
    print(f"    prompt:              {suite_result.total_tokens_prompt:,}")
    print(f"    completion:          {suite_result.total_tokens_completion:,}")
    print(f"  tokens_saved:          {suite_result.total_tokens_saved:,}  "
          f"(early-stop + pre-screen)")
    print(f"  token_efficiency_rate: {suite_result.token_efficiency_rate:.1%}  "
          f"(saved / total+saved)")
    print(f"  mean_tokens/case:      {suite_result.mean_tokens_per_case:,.0f}")
    print(f"  mean_tokens/correct:   {suite_result.mean_tokens_per_correct:,.0f}")
    print(f"  p50/p95_tokens:        {suite_result.p50_tokens_per_case:,.0f} / "
          f"{suite_result.p95_tokens_per_case:,.0f}")


def cmd_list(args: argparse.Namespace) -> None:
    suite = load_full_suite()
    cases = suite.cases

    if args.category:
        cases = [c for c in cases if c.category.value == args.category]
    if args.difficulty:
        cases = [c for c in cases if c.difficulty.value == args.difficulty]

    print(f"\n{'ID':<16} {'CATEGORY':<16} {'DIFF':<8} {'NAME'}")
    print("─" * 80)
    for case in cases:
        diff_col = {
            "easy":   c("green", "easy"),
            "medium": c("yellow", "medium"),
            "hard":   c("red", "hard"),
        }.get(case.difficulty.value, case.difficulty.value)
        print(f"  {case.case_id:<14} {case.category.value:<16} {diff_col:<20} {case.name}")
    print(f"\n  {len(cases)} cases total.\n")


def cmd_show(args: argparse.Namespace) -> None:
    suite = load_full_suite()
    matches = [c for c in suite.cases if c.case_id == args.case_id]
    if not matches:
        print(c("red", f"Case '{args.case_id}' not found."))
        return
    case = matches[0]
    print(json.dumps(case.model_dump(mode="json"), indent=2, default=str))


# ─────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m aegeanbench.cli",
        description="AegeanBench — multi-agent consensus benchmark CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run benchmark cases")
    run_p.add_argument("--runner",     default="mock",    choices=["mock", "aegean", "http"],
                       help="Runner backend: mock (no LLM), aegean (Python import), http (REST API)")
    run_p.add_argument("--http-url",   default="http://localhost:8000", dest="http_url",
                       help="Base URL for HTTP runner (default: http://localhost:8000)")
    run_p.add_argument("--category",  default=None,
                       choices=[e.value for e in BenchmarkCategory],
                       help="Filter by category")
    run_p.add_argument("--difficulty", default=None,
                       choices=[e.value for e in Difficulty],
                       help="Filter by difficulty")
    run_p.add_argument("--tag",        default=None,      help="Filter by tag")
    run_p.add_argument("--case-id",    default=None,      dest="case_id",
                       help="Run a single case by ID")
    run_p.add_argument("--output-dir", default="results", dest="output_dir",
                       help="Directory to save JSON results (default: results)")
    run_p.add_argument("--seed",       default=42,        type=int,
                       help="Random seed for mock runner")

    # list
    lst_p = sub.add_parser("list", help="List available cases")
    lst_p.add_argument("--category",  default=None,
                       choices=[e.value for e in BenchmarkCategory])
    lst_p.add_argument("--difficulty", default=None,
                       choices=[e.value for e in Difficulty])

    # show
    show_p = sub.add_parser("show", help="Show details of a single case")
    show_p.add_argument("case_id", help="Case ID, e.g. C-HARD-001")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "show":
        cmd_show(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

