"""Survey calibration dataset + end-to-end scoring via SurveyRunner (mock fetch)."""

from aegeanbench.core.models import BenchmarkCategory
from aegeanbench.datasets.survey_cases import load_survey_suite
from aegeanbench.runners.survey_runner import SurveyRunner
from aegeanbench.scoring import calibration as cal


def test_survey_suite_loads():
    suite = load_survey_suite()
    ids = {c.case_id for c in suite.cases}
    assert {"SV-BREXIT-2016", "SV-USELEC-2016"} <= ids
    for c in suite.cases:
        assert c.category == BenchmarkCategory.EVENT
        assert "predict" in c.scenario_request
        assert c.event_ground_truth.confidence == "high"     # outcomes are real


def _mock_results(case):
    """Stand in for a real Loka run: return a summarize dict per case."""
    if case.case_id == "SV-BREXIT-2016":
        return {"questions": {"vote": {"distribution": {"Leave": 0.52, "Remain": 0.48}}}}
    # US electoral: geo roll-up leader Trump
    return {"questions": {"vote": {
        "geo_rollup": {"leader": "Trump",
                       "per_unit": {"PA": {"winner": "Trump"}, "MI": {"winner": "Trump"},
                                    "WI": {"winner": "Trump"}}},
        "distribution": {"Trump": 0.51, "Clinton": 0.49}}}}


def test_survey_runner_scores_suite_correct():
    suite = load_survey_suite()
    runner = SurveyRunner(results_fetcher=_mock_results)
    results = runner.run_suite(suite.cases)
    by_id = {r.case_id: r for r in results}
    assert by_id["SV-BREXIT-2016"].correct is True and by_id["SV-BREXIT-2016"].final_answer == "Leave"
    assert by_id["SV-USELEC-2016"].correct is True and by_id["SV-USELEC-2016"].final_answer == "Trump"


def test_calibration_over_survey_suite():
    suite = load_survey_suite()
    results = SurveyRunner(results_fetcher=_mock_results).run_suite(suite.cases)
    # build forecast records from each run's stored distribution + real outcome
    records = []
    for c, r in zip(suite.cases, results):
        rec = cal.record_from_summary(r.raw_output["results"], "vote",
                                      c.event_ground_truth.direction_label,
                                      stratum=c.case_id)
        if rec:
            records.append(rec)
    rep = cal.calibration_report(records)
    assert rep["n"] == 2 and rep["accuracy"] == 1.0       # both predicted correctly
