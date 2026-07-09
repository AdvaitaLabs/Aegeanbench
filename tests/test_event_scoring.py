"""Unit tests for event/scenario backtest scoring (Loka benchmark, P1)."""

import json

from aegeanbench.core.models import EventGroundTruth, EventPrediction
from aegeanbench.scoring import (
    extract_prediction,
    score_prediction,
    apply_memorization_gap,
    gap_level,
)


def _report_with_prediction(pred: dict) -> str:
    """Build a Loka-style report markdown ending in the AEGEANBENCH block."""
    body = json.dumps({"prediction": pred}, ensure_ascii=False, indent=2)
    return (
        "# 报告正文\n\n... a lot of prose ...\n\n"
        "\n\n<!-- AEGEANBENCH:PREDICTION_START -->\n"
        "```json\n" + body + "\n```\n"
        "<!-- AEGEANBENCH:PREDICTION_END -->\n"
    )


# ── extraction ────────────────────────────────────────────────

def test_extract_prediction_from_block():
    md = _report_with_prediction({
        "direction": "decline", "point_estimate": -52, "unit": "percent",
        "ci_80": [-68, -38], "confidence": 0.81, "rationale": "败选致机构捐赠骤降",
    })
    p = extract_prediction(md)
    assert p is not None
    assert p.direction == "decline"
    assert p.point_estimate == -52
    assert p.ci_80 == [-68, -38]
    assert p.confidence == 0.81


def test_extract_returns_none_without_block():
    assert extract_prediction("# just a report, no prediction block") is None
    assert extract_prediction("") is None
    assert extract_prediction(None) is None


def test_extract_tolerates_bad_ci():
    md = _report_with_prediction({"direction": "up", "ci_80": [1]})  # malformed CI
    p = extract_prediction(md)
    assert p is not None and p.ci_80 is None and p.direction == "up"


# ── scoring: numeric + direction ──────────────────────────────

def test_score_numeric_case_all_correct():
    gt = EventGroundTruth(direction_label="sharp_decline", actual_value=-58,
                          unit="percent", confidence="approx")
    pred = EventPrediction(direction="decline", point_estimate=-52, ci_80=[-68, -38])
    m = score_prediction(pred, gt)
    assert m.direction_correct is True          # decline ⊆ sharp_decline / same polarity
    assert m.within_ci is True                  # -58 ∈ [-68, -38]
    assert abs(m.value_error_pct - 10.34) < 0.1
    assert m.case_score == 1.0


def test_score_numeric_out_of_ci():
    gt = EventGroundTruth(direction_label="up", actual_value=1_000_000)
    pred = EventPrediction(direction="up", point_estimate=780_000, ci_80=[600_000, 900_000])
    m = score_prediction(pred, gt)
    assert m.direction_correct is True
    assert m.within_ci is False                 # 1.0M not in [600k, 900k]
    assert m.case_score == 0.6                  # direction only


# ── scoring: categorical (no numeric target) ──────────────────

def test_score_categorical_case():
    gt = EventGroundTruth(direction_label="suspended", confidence="high")
    m_ok = score_prediction(EventPrediction(direction="suspended"), gt)
    assert m_ok.direction_correct is True and m_ok.within_ci is None
    assert m_ok.case_score == 1.0

    m_bad = score_prediction(EventPrediction(direction="continue_restructured"), gt)
    assert m_bad.direction_correct is False and m_bad.case_score == 0.0


def test_score_missing_prediction_is_all_miss():
    gt = EventGroundTruth(direction_label="delayed")
    m = score_prediction(None, gt)
    assert m.direction_correct is False and m.case_score == 0.0


# ── memorization gap ──────────────────────────────────────────

def test_memorization_gap_high_when_anon_fails():
    gt = EventGroundTruth(direction_label="suspended", confidence="high")
    real = score_prediction(EventPrediction(direction="suspended"), gt, is_anonymized=False)
    anon = score_prediction(EventPrediction(direction="continue_restructured"), gt, is_anonymized=True)
    gap = apply_memorization_gap(real, anon)
    assert gap == 1.0 and real.gap_level == "high" and anon.gap_level == "high"


def test_memorization_gap_low_when_twins_agree():
    gt = EventGroundTruth(direction_label="target_achieved")
    real = score_prediction(EventPrediction(direction="target_achieved"), gt, is_anonymized=False)
    anon = score_prediction(EventPrediction(direction="target_achieved"), gt, is_anonymized=True)
    gap = apply_memorization_gap(real, anon)
    assert gap == 0.0 and real.gap_level == "low"


def test_gap_level_bands():
    assert gap_level(0.03) == "low"
    assert gap_level(0.10) == "medium"
    assert gap_level(0.30) == "high"
