"""Dataset + engine-aggregation tests for the Loka event benchmark (P1)."""

from aegeanbench.core.models import (
    BenchmarkCategory,
    BenchmarkResult,
    ConsensusOutcome,
    EventPrediction,
)
from aegeanbench.datasets.event_cases import load_event_suite
from aegeanbench.metrics.engine import MetricsEngine
from aegeanbench.scoring import score_prediction, apply_memorization_gap


# ── dataset shape ─────────────────────────────────────────────

def test_suite_loads_twin_pairs():
    suite = load_event_suite()
    assert len(suite.cases) == 20                      # 10 scenarios × (real + anon)
    assert all(c.category == BenchmarkCategory.EVENT for c in suite.cases)
    reals = [c for c in suite.cases if not c.is_anonymized]
    anons = [c for c in suite.cases if c.is_anonymized]
    assert len(reals) == len(anons) == 10


def test_twins_cross_link_and_share_ground_truth():
    suite = load_event_suite()
    by_id = {c.case_id: c for c in suite.cases}
    for c in suite.cases:
        twin = by_id[c.twin_case_id]                   # every twin resolves
        assert twin.twin_case_id == c.case_id          # links are symmetric
        # twins share the same realized outcome
        assert twin.event_ground_truth.direction_label == c.event_ground_truth.direction_label
        assert twin.is_anonymized != c.is_anonymized


def test_internal_only_filter():
    full = load_event_suite(include_internal=True)
    safe = load_event_suite(include_internal=False)
    assert len(full.cases) == 20 and len(safe.cases) == 18   # drops CF-SENT-004 pair
    assert not any("internal_only" in c.tags for c in safe.cases)


def test_anon_masks_entity_but_keeps_facts_count():
    suite = load_event_suite()
    by_id = {c.case_id: c for c in suite.cases}
    real = by_id["AD-PROJ-001-R"]
    anon = by_id["AD-PROJ-001-A"]
    assert "Louvre" in real.scenario_request["entity_shown"]
    assert "Louvre" not in anon.scenario_request["entity_shown"]
    # same number of facts (identity masked, structure preserved)
    assert len(real.scenario_request["public_facts"]) == len(anon.scenario_request["public_facts"])


# ── engine aggregation over event results ─────────────────────

def _result(case, pred_dict):
    pred = EventPrediction(**pred_dict) if pred_dict else None
    m = score_prediction(pred, case.event_ground_truth, is_anonymized=case.is_anonymized)
    return BenchmarkResult(
        case_id=case.case_id, case_name=case.name,
        category=BenchmarkCategory.EVENT, difficulty=case.difficulty,
        outcome=ConsensusOutcome.CONVERGED, correct=m.direction_correct,
        event_metrics=m,
    )


def test_engine_aggregates_event_metrics_and_gap():
    suite = load_event_suite()
    by_id = {c.case_id: c for c in suite.cases}

    # CGI pair: real nails "suspended", anon drifts to "continue" → gap high.
    real = _result(by_id["CF-CGI-002-R"], {"direction": "suspended", "confidence": 0.9})
    anon = _result(by_id["CF-CGI-002-A"], {"direction": "continue_restructured", "confidence": 0.5})
    apply_memorization_gap(real.event_metrics, anon.event_metrics)

    sr = MetricsEngine().aggregate("s1", "event-mini", [real, anon])
    assert sr.event_direction_accuracy == 0.5          # 1 of 2 correct
    assert sr.event_mean_memorization_gap == 1.0        # real 1.0 - anon 0.0
    assert real.event_metrics.gap_level == "high"
