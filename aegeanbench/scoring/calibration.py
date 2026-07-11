"""
Calibration analysis (Phase 3) for probabilistic survey forecasts.

The survey subsystem produces a *distribution* per question (e.g.
{"A党":0.46,"B党":0.28,...}); the realized outcome is one class. This module
scores how well-calibrated those distributions are:

  * brier_score       — multiclass Brier (lower = better)
  * accuracy          — argmax == actual
  * reliability_bins  — predicted-prob vs observed-frequency curve
  * stratified        — accuracy/Brier per stratum (which segment predicts worst)

This is the "敢报准确率" backbone: run a suite of historical cases, get these
numbers, and feed the worst strata back into sampling/prompt/weight correction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForecastRecord(BaseModel):
    distribution: Dict[str, float] = Field(default_factory=dict)  # class -> prob
    actual:       str
    stratum:      Optional[str] = None      # geo / demographic segment label


def _norm(dist: Dict[str, float]) -> Dict[str, float]:
    tot = sum(v for v in dist.values() if v is not None) or 1.0
    return {k: (v or 0.0) / tot for k, v in dist.items()}


def brier_score(records: List[ForecastRecord]) -> float:
    """Mean multiclass Brier score. 0 = perfect; higher = worse."""
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        dist = _norm(r.distribution)
        classes = set(dist) | {r.actual}
        total += sum((dist.get(c, 0.0) - (1.0 if c == r.actual else 0.0)) ** 2
                     for c in classes)
    return round(total / len(records), 4)


def accuracy(records: List[ForecastRecord]) -> float:
    if not records:
        return 0.0
    hits = 0
    for r in records:
        if r.distribution and max(r.distribution, key=r.distribution.get) == r.actual:
            hits += 1
    return round(hits / len(records), 4)


def reliability_bins(records: List[ForecastRecord], n_bins: int = 10) -> List[Dict[str, Any]]:
    """
    Reliability curve: over all (class, predicted_prob) pairs, bin by predicted
    prob and compare mean predicted vs observed frequency of that class actually
    occurring. Well-calibrated → mean_pred ≈ observed in every bin.
    """
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "sum_pred": 0.0,
             "hits": 0, "count": 0} for i in range(n_bins)]
    for r in records:
        dist = _norm(r.distribution)
        for cls, p in dist.items():
            idx = min(n_bins - 1, int(p * n_bins))
            b = bins[idx]
            b["sum_pred"] += p
            b["hits"] += 1 if cls == r.actual else 0
            b["count"] += 1
    out = []
    for b in bins:
        if b["count"]:
            out.append({"range": [round(b["lo"], 2), round(b["hi"], 2)],
                        "count": b["count"],
                        "mean_pred": round(b["sum_pred"] / b["count"], 4),
                        "observed": round(b["hits"] / b["count"], 4)})
    return out


def stratified(records: List[ForecastRecord]) -> Dict[str, Dict[str, Any]]:
    """Accuracy + Brier per stratum. Reveals which segment predicts worst."""
    groups: Dict[str, List[ForecastRecord]] = {}
    for r in records:
        groups.setdefault(r.stratum or "all", []).append(r)
    return {s: {"n": len(rs), "accuracy": accuracy(rs), "brier": brier_score(rs)}
            for s, rs in groups.items()}


def record_from_summary(summary: Dict[str, Any], qid: str, actual: str,
                        *, stratum: Optional[str] = None) -> Optional[ForecastRecord]:
    """Build a ForecastRecord from a survey `summarize` dict + the true class."""
    dist = ((summary.get("questions") or {}).get(qid) or {}).get("distribution")
    if not dist:
        return None
    return ForecastRecord(distribution=dist, actual=actual, stratum=stratum)


# ─────────────────────────────────────────────────────────────────────────
# Competitor confidence calibration (live arena)
# ─────────────────────────────────────────────────────────────────────────
# An LLM's self-reported confidence (0.68, 0.72, …) is a guess, not a
# probability. Once the arena has accumulated enough (confidence, correct)
# pairs per competitor, we can replace the raw number with the observed hit
# rate of that competitor's confidence bin — Laplace-smoothed so small samples
# shrink toward 0.5 instead of screaming 0% or 100%.

CONF_BINS = [(0.0, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.01)]


def fit_confidence_table(history: List[Dict[str, Any]],
                         min_samples: int = 8) -> Dict[str, List[Dict[str, Any]]]:
    """
    history rows: {"competitor", "confidence", "correct"} (extra keys ignored).
    Returns {competitor: [{lo, hi, n, observed}]} — only for competitors with
    at least `min_samples` usable rows. observed = (hits+1)/(n+2) per bin.
    """
    by_comp: Dict[str, List[Dict[str, Any]]] = {}
    for row in history:
        comp, conf = row.get("competitor"), row.get("confidence")
        if not comp or not isinstance(conf, (int, float)):
            continue
        by_comp.setdefault(comp, []).append(row)

    tables: Dict[str, List[Dict[str, Any]]] = {}
    for comp, rows in by_comp.items():
        if len(rows) < min_samples:
            continue
        bins = []
        for lo, hi in CONF_BINS:
            hit = n = 0
            for r in rows:
                if lo <= float(r["confidence"]) < hi:
                    n += 1
                    hit += 1 if r.get("correct") else 0
            bins.append({"lo": lo, "hi": hi, "n": n,
                         "observed": round((hit + 1) / (n + 2), 4)})
        tables[comp] = bins
    return tables


def calibrate_confidence(tables: Dict[str, List[Dict[str, Any]]],
                         competitor: str, confidence: Optional[float]) -> Optional[float]:
    """Raw self-reported confidence → observed accuracy of its bin, or None
    when the competitor has no fitted table / the bin is empty."""
    if confidence is None:
        return None
    for b in tables.get(competitor) or []:
        if b["lo"] <= float(confidence) < b["hi"]:
            return b["observed"] if b["n"] > 0 else None
    return None


def calibration_report(records: List[ForecastRecord]) -> Dict[str, Any]:
    """One-shot bundle. `worst_stratum` = lowest-accuracy segment for triage."""
    strat = stratified(records)
    worst = min(strat.items(), key=lambda kv: kv[1]["accuracy"])[0] if strat else None
    return {
        "n": len(records),
        "accuracy": accuracy(records),
        "brier_score": brier_score(records),
        "reliability": reliability_bins(records),
        "stratified": strat,
        "worst_stratum": worst,
    }
