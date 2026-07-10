"""Phase 3 calibration tests: Brier / accuracy / reliability / stratified."""

from aegeanbench.scoring import calibration as cal
from aegeanbench.scoring.calibration import ForecastRecord


def test_brier_perfect_and_worst():
    perfect = [ForecastRecord(distribution={"A": 1.0, "B": 0.0}, actual="A")]
    assert cal.brier_score(perfect) == 0.0
    worst = [ForecastRecord(distribution={"A": 0.0, "B": 1.0}, actual="A")]
    assert cal.brier_score(worst) == 2.0        # multiclass Brier max


def test_brier_partial():
    recs = [ForecastRecord(distribution={"A": 0.7, "B": 0.3}, actual="A")]
    assert abs(cal.brier_score(recs) - 0.18) < 1e-6   # .09 + .09


def test_accuracy_argmax():
    recs = [ForecastRecord(distribution={"A": 0.6, "B": 0.4}, actual="A"),
            ForecastRecord(distribution={"A": 0.6, "B": 0.4}, actual="B")]
    assert cal.accuracy(recs) == 0.5


def test_stratified_finds_worst_segment():
    recs = [
        ForecastRecord(distribution={"A": 0.9, "B": 0.1}, actual="A", stratum="CA"),
        ForecastRecord(distribution={"A": 0.9, "B": 0.1}, actual="A", stratum="CA"),
        ForecastRecord(distribution={"A": 0.9, "B": 0.1}, actual="B", stratum="TX"),  # wrong
    ]
    rep = cal.calibration_report(recs)
    assert rep["stratified"]["CA"]["accuracy"] == 1.0
    assert rep["stratified"]["TX"]["accuracy"] == 0.0
    assert rep["worst_stratum"] == "TX"


def test_reliability_bins_shape():
    recs = [ForecastRecord(distribution={"A": 0.8, "B": 0.2}, actual="A") for _ in range(5)]
    bins = cal.reliability_bins(recs, n_bins=10)
    # 0.8 lands in bin [0.8,0.9], observed freq of A there = 1.0; 0.2 in [0.2,0.3], obs 0
    hi = next(b for b in bins if b["range"] == [0.8, 0.9])
    assert hi["mean_pred"] == 0.8 and hi["observed"] == 1.0


def test_record_from_summary():
    summary = {"questions": {"vote": {"distribution": {"A党": 0.55, "B党": 0.45}}}}
    r = cal.record_from_summary(summary, "vote", "A党", stratum="US")
    assert r.actual == "A党" and r.distribution["A党"] == 0.55 and r.stratum == "US"
    assert cal.record_from_summary({"questions": {}}, "vote", "A党") is None
