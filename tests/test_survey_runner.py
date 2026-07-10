"""Tests for SurveyRunner — extraction + scoring via an injected results fetcher."""

from aegeanbench.core.models import (
    BenchmarkCase, BenchmarkCategory, ConsensusOutcome, Difficulty, EventGroundTruth,
)
from aegeanbench.runners.survey_runner import SurveyRunner, extract_prediction


def _case(gt_dir, mode="geo_leader", qid="vote", is_anon=False, cid="US-ELEC", twin=None):
    return BenchmarkCase(
        case_id=cid, name="US election", description="who wins",
        category=BenchmarkCategory.EVENT, difficulty=Difficulty.HARD,
        scenario_request={"marginals": {"age": {"18-24": 1.0}}, "questions": [],
                          "predict": {"qid": qid, "mode": mode}},
        event_ground_truth=EventGroundTruth(direction_label=gt_dir, confidence="high"),
        is_anonymized=is_anon, twin_case_id=twin,
    )


# ── extraction ────────────────────────────────────────────────

def test_extract_geo_leader():
    results = {"questions": {"vote": {"geo_rollup": {"leader": "B党"}}}}
    p = extract_prediction(results, {"qid": "vote", "mode": "geo_leader"})
    assert p.direction == "B党"


def test_extract_top_choice_with_ci():
    results = {"questions": {"vote": {"distribution": {"A党": 0.4, "B党": 0.6},
                                      "ci_80": {"B党": [0.55, 0.65]}}}}
    p = extract_prediction(results, {"qid": "vote", "mode": "top_choice"})
    assert p.direction == "B党" and p.ci_80 == [0.55, 0.65]


def test_extract_mean():
    results = {"questions": {"pol": {"mean": 3.8}}}
    p = extract_prediction(results, {"qid": "pol", "mode": "mean"})
    assert p.point_estimate == 3.8


def test_extract_missing_returns_none():
    assert extract_prediction({}, {"qid": "x", "mode": "mean"}) is None


# ── run_case scoring ──────────────────────────────────────────

def test_run_case_scores_geo_leader():
    case = _case("B党")
    runner = SurveyRunner(results_fetcher=lambda c: {"questions": {"vote": {"geo_rollup": {"leader": "B党"}}}})
    res = runner.run_case(case)
    assert res.outcome == ConsensusOutcome.CONVERGED
    assert res.correct is True and res.final_answer == "B党"


def test_run_case_wrong_prediction():
    case = _case("A党")
    runner = SurveyRunner(results_fetcher=lambda c: {"questions": {"vote": {"geo_rollup": {"leader": "B党"}}}})
    res = runner.run_case(case)
    assert res.correct is False


def test_run_case_fetch_error_is_captured():
    def boom(c):
        raise RuntimeError("loka down")
    res = SurveyRunner(results_fetcher=boom).run_case(_case("B党"))
    assert res.outcome == ConsensusOutcome.ERROR and res.error == "loka down"


def test_run_suite_pairs_memorization_gap():
    real = _case("suspended", mode="geo_leader", cid="C-R", twin="C-A", is_anon=False)
    anon = _case("suspended", mode="geo_leader", cid="C-A", twin="C-R", is_anon=True)
    fetch = {
        "C-R": {"questions": {"vote": {"geo_rollup": {"leader": "suspended"}}}},   # real correct
        "C-A": {"questions": {"vote": {"geo_rollup": {"leader": "continue"}}}},     # anon wrong
    }
    runner = SurveyRunner(results_fetcher=lambda c: fetch[c.case_id])
    out = {r.case_id: r for r in runner.run_suite([real, anon])}
    assert out["C-R"].event_metrics.memorization_gap == 1.0
    assert out["C-R"].event_metrics.gap_level == "high"
