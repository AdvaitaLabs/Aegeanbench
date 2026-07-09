"""
Event/scenario backtest scoring.

Turns a Loka report into a scored EventMetrics against the case's realized
ground truth, and computes the memorization gap for real/anonymized twins.

The bridge to Loka is the machine-readable block Loka's ReportAgent appends
to every report in benchmark mode:

    <!-- AEGEANBENCH:PREDICTION_START -->
    ```json
    { "prediction": { "direction": ..., "point_estimate": ..., "ci_80": [...],
                      "confidence": ..., "rationale": ... } }
    ```
    <!-- AEGEANBENCH:PREDICTION_END -->

We extract that block (never parse the prose) so scoring is deterministic.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from aegeanbench.core.models import EventGroundTruth, EventMetrics, EventPrediction

# The prediction block Loka emits. Tolerant of whitespace; the json fence is
# optional (we grab the first {...} after the START marker either way).
_BLOCK_RE = re.compile(
    r"<!--\s*AEGEANBENCH:PREDICTION_START\s*-->(.*?)<!--\s*AEGEANBENCH:PREDICTION_END\s*-->",
    re.DOTALL,
)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Canonical polarity buckets so "decline" ≈ "sharp_decline" ≈ "down" ≈ "drop".
_POLARITY = {
    "up": {"up", "rise", "rising", "increase", "increased", "gain", "bull", "positive"},
    "down": {"down", "decline", "decrease", "drop", "fall", "falling", "bear", "negative",
             "sharp_decline", "negative_spike"},
    "flat": {"flat", "neutral", "unchanged", "stable"},
}


def _canon(label: Optional[str]) -> str:
    """Normalize a direction/label to a comparable token."""
    if not label:
        return ""
    return re.sub(r"[\s\-]+", "_", str(label).strip().lower())


def _polarity(token: str) -> Optional[str]:
    for pol, words in _POLARITY.items():
        if token in words:
            return pol
    return None


def _direction_match(predicted: Optional[str], truth: Optional[str]) -> bool:
    """
    True if the predicted direction agrees with the ground-truth direction.
    Matches on: exact, substring either way, or shared polarity bucket.
    """
    p, t = _canon(predicted), _canon(truth)
    if not p or not t:
        return False
    if p == t or p in t or t in p:
        return True
    pp, tp = _polarity(p), _polarity(t)
    return pp is not None and pp == tp


def extract_prediction(report_markdown: Optional[str]) -> Optional[EventPrediction]:
    """Pull the AEGEANBENCH:PREDICTION block out of a Loka report. None if absent/malformed."""
    if not report_markdown:
        return None
    block = _BLOCK_RE.search(report_markdown)
    region = block.group(1) if block else report_markdown
    m = _JSON_RE.search(region)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    pred = obj.get("prediction", obj) if isinstance(obj, dict) else {}
    if not isinstance(pred, dict):
        return None
    ci = pred.get("ci_80")
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        ci = None
    try:
        return EventPrediction(
            direction=pred.get("direction"),
            point_estimate=pred.get("point_estimate"),
            unit=pred.get("unit"),
            ci_80=list(ci) if ci is not None else None,
            confidence=float(pred.get("confidence") or 0.0),
            rationale=str(pred.get("rationale") or ""),
        )
    except (ValueError, TypeError):
        return None


def score_prediction(
    prediction: Optional[EventPrediction],
    ground_truth: EventGroundTruth,
    *,
    is_anonymized: bool = False,
) -> EventMetrics:
    """Grade one prediction against the realized outcome."""
    m = EventMetrics(
        ground_truth_direction=ground_truth.direction_label,
        actual_value=ground_truth.actual_value,
        is_anonymized=is_anonymized,
    )
    if prediction is None:
        return m  # no parsable prediction → all-miss (case_score 0.0)

    m.predicted_direction = prediction.direction
    m.predicted_value = prediction.point_estimate
    m.direction_correct = _direction_match(prediction.direction, ground_truth.direction_label)

    actual = ground_truth.actual_value
    if actual is not None and prediction.point_estimate is not None and actual != 0:
        m.value_error_pct = round(
            abs(prediction.point_estimate - actual) / abs(actual) * 100.0, 2
        )
    if actual is not None and prediction.ci_80:
        lo, hi = min(prediction.ci_80), max(prediction.ci_80)
        m.within_ci = lo <= actual <= hi

    # Composite 0..1 case score used for the memorization gap.
    if m.within_ci is None:
        # No numeric target → direction is the whole story.
        m.case_score = 1.0 if m.direction_correct else 0.0
    else:
        m.case_score = round(
            (0.6 if m.direction_correct else 0.0) + (0.4 if m.within_ci else 0.0), 4
        )
    return m


def gap_level(gap: float) -> str:
    """Map a memorization gap to the plan's low/medium/high bands."""
    if gap <= 0.05:
        return "low"
    if gap <= 0.15:
        return "medium"
    return "high"


def apply_memorization_gap(real: EventMetrics, anon: EventMetrics) -> float:
    """
    Set memorization_gap = real.case_score - anon.case_score on both metrics
    (clamped at 0 — a negative gap just means the anon twin scored higher,
    which is still "no evidence of memorization"). Returns the gap.
    """
    gap = round(max(0.0, real.case_score - anon.case_score), 4)
    level = gap_level(gap)
    for m in (real, anon):
        m.memorization_gap = gap
        m.gap_level = level
    return gap
