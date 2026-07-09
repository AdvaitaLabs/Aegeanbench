"""LokaRunner tests — scoring + twin pairing via an injected report fetcher."""

import json

from aegeanbench.core.models import BenchmarkCategory, ConsensusOutcome
from aegeanbench.datasets.event_cases import load_event_suite
from aegeanbench.runners.loka_runner import LokaRunner


def _report(pred: dict) -> str:
    body = json.dumps({"prediction": pred}, ensure_ascii=False)
    return ("报告正文……\n\n<!-- AEGEANBENCH:PREDICTION_START -->\n```json\n"
            + body + "\n```\n<!-- AEGEANBENCH:PREDICTION_END -->\n")


def _cgi_pair():
    suite = load_event_suite()
    by_id = {c.case_id: c for c in suite.cases}
    return by_id["CF-CGI-002-R"], by_id["CF-CGI-002-A"]


def test_run_case_scores_from_report():
    real, _ = _cgi_pair()
    runner = LokaRunner(report_fetcher=lambda c: _report({"direction": "suspended", "confidence": 0.9}))
    res = runner.run_case(real)
    assert res.category == BenchmarkCategory.EVENT
    assert res.outcome == ConsensusOutcome.CONVERGED
    assert res.correct is True
    assert res.final_answer == "suspended"
    assert res.event_metrics.case_score == 1.0


def test_run_suite_pairs_memorization_gap():
    real, anon = _cgi_pair()
    # real nails it; anon (identity masked) drifts → high gap
    canned = {
        real.case_id: _report({"direction": "suspended", "confidence": 0.9}),
        anon.case_id: _report({"direction": "continue_restructured", "confidence": 0.5}),
    }
    runner = LokaRunner(report_fetcher=lambda c: canned[c.case_id])
    results = runner.run_suite([real, anon])

    by_id = {r.case_id: r for r in results}
    assert by_id[real.case_id].event_metrics.memorization_gap == 1.0
    assert by_id[anon.case_id].event_metrics.memorization_gap == 1.0   # set on both twins
    assert by_id[real.case_id].event_metrics.gap_level == "high"


def test_missing_prediction_block_is_errored_miss():
    real, _ = _cgi_pair()
    runner = LokaRunner(report_fetcher=lambda c: "报告里没有预测块")
    res = runner.run_case(real)
    assert res.outcome == ConsensusOutcome.ERROR
    assert res.correct is False
    assert res.event_metrics.case_score == 0.0


def test_fetcher_exception_is_captured_not_raised():
    real, _ = _cgi_pair()
    def boom(_c):
        raise RuntimeError("loka down")
    runner = LokaRunner(report_fetcher=boom)
    res = runner.run_case(real)
    assert res.outcome == ConsensusOutcome.ERROR
    assert res.error == "loka down"
