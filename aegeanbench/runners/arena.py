"""
Multi-model arena — run the SAME benchmark cases through several models and
rank them. This is what turns the frontend leaderboard from MOCK into real
numbers: each model (gpt-5.4 / claude-opus-4-6 / grok-4.3-fast) makes a
prediction per case, scored against the realized outcome.

`predict_fn(case, model_id) -> EventPrediction | None` is injected (default:
drive Loka's sim over HTTP with a `model` param). Scoring reuses
aegeanbench.scoring, so the arena inherits direction / value-error / CI logic.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from aegeanbench.core.models import BenchmarkCase, EventPrediction
from aegeanbench.scoring import score_prediction

PredictFn = Callable[[BenchmarkCase, str], Optional[EventPrediction]]


def models_from_env(env: str = "BENCHMARK_MODELS",
                    default: Optional[List[str]] = None) -> List[str]:
    """Parse BENCHMARK_MODELS ('gpt-5.4,claude-opus-4-6,grok-4.3-fast' or the
    'model:name:cost,...' arena format) into a list of model ids."""
    raw = (os.environ.get(env) or "").strip()
    if not raw:
        return default or ["claude-opus-4-6", "gpt-5.4", "grok-4.3-fast"]
    return [p.strip().split(":")[0].strip() for p in raw.split(",") if p.strip()]


def _median(xs: List[float]) -> Optional[float]:
    s = sorted(xs)
    if not s:
        return None
    n = len(s)
    return s[n // 2] if n % 2 else round((s[n // 2 - 1] + s[n // 2]) / 2, 2)


def run_arena(cases: List[BenchmarkCase], model_ids: List[str],
              predict_fn: PredictFn) -> Dict[str, Any]:
    """
    Run every (case, model) and produce a leaderboard + per-case breakdown.
    A case with no event_ground_truth is skipped. A model whose predict_fn
    raises/returns None on a case is scored as a miss for that case.
    """
    scored: Dict[str, List[Any]] = {m: [] for m in model_ids}
    per_case: List[Dict[str, Any]] = []

    for case in cases:
        gt = case.event_ground_truth
        if gt is None:
            continue
        row = {"case_id": case.case_id, "name": case.name,
               "actual": gt.direction_label, "models": {}}
        for m in model_ids:
            try:
                pred = predict_fn(case, m)
            except Exception:
                pred = None
            metrics = score_prediction(pred, gt)
            scored[m].append(metrics)
            row["models"][m] = {
                "prediction": metrics.predicted_direction,
                "correct": metrics.direction_correct,
                "value_error_pct": metrics.value_error_pct,
                "within_ci": metrics.within_ci,
                "case_score": metrics.case_score,
            }
        per_case.append(row)

    leaderboard = []
    for m in model_ids:
        ms = scored[m]
        n = len(ms) or 1
        correct = sum(1 for x in ms if x.direction_correct)
        errs = [x.value_error_pct for x in ms if x.value_error_pct is not None]
        cis = [x for x in ms if x.within_ci is not None]
        ci_hit = sum(1 for x in cis if x.within_ci)
        leaderboard.append({
            "model": m, "n": len(ms), "correct": correct,
            "accuracy": round(correct / n, 4),
            "median_error_pct": _median(errs),
            "ci_hit": ci_hit, "ci_total": len(cis),
            "ci_pct": round(ci_hit / len(cis), 4) if cis else None,
            "mean_case_score": round(sum(x.case_score for x in ms) / n, 4),
        })
    leaderboard.sort(key=lambda r: (-r["accuracy"],
                                    r["median_error_pct"] if r["median_error_pct"] is not None else 1e9))
    return {"models": model_ids, "leaderboard": leaderboard, "per_case": per_case}
