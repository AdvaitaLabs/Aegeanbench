"""Competitor confidence calibration (live arena): fit + lookup."""

from aegeanbench.scoring import calibrate_confidence, fit_confidence_table


def _rows(comp, pairs):
    return [{"competitor": comp, "confidence": c, "correct": ok} for c, ok in pairs]


def test_fit_requires_min_samples():
    hist = _rows("loka", [(0.7, True)] * 5)          # below default min_samples=8
    assert fit_confidence_table(hist) == {}
    assert calibrate_confidence({}, "loka", 0.7) is None


def test_fit_and_calibrate_shrinks_overconfidence():
    # 10 forecasts at ~0.9 confidence but only 5 correct → calibrated ≈ 0.5
    hist = _rows("loka", [(0.9, i < 5) for i in range(10)])
    tables = fit_confidence_table(hist)
    cal = calibrate_confidence(tables, "loka", 0.9)
    assert cal is not None and abs(cal - 0.5) < 0.01   # (5+1)/(10+2)
    # a bin with no samples yields None, not a made-up number
    assert calibrate_confidence(tables, "loka", 0.6) is None


def test_calibrate_ignores_unknown_competitor_and_none():
    hist = _rows("loka", [(0.8, True)] * 8)
    tables = fit_confidence_table(hist)
    assert calibrate_confidence(tables, "gpt-5.4", 0.8) is None
    assert calibrate_confidence(tables, "loka", None) is None


def test_laplace_smoothing_on_small_bins():
    # 8 rows all in one bin, all correct → (8+1)/(8+2) = 0.9, not 1.0
    hist = _rows("loka", [(0.72, True)] * 8)
    tables = fit_confidence_table(hist)
    assert calibrate_confidence(tables, "loka", 0.75) == 0.9
