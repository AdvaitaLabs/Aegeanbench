"""Multi-model arena tests: leaderboard ranking across models (mock predict_fn)."""

import os

from aegeanbench.core.models import EventPrediction
from aegeanbench.datasets.event_cases import load_event_suite
from aegeanbench.runners import arena


def test_models_from_env(monkeypatch):
    monkeypatch.delenv("BENCHMARK_MODELS", raising=False)
    assert arena.models_from_env() == ["claude-opus-4-6", "gpt-5.4", "grok-4.3-fast"]
    monkeypatch.setenv("BENCHMARK_MODELS", "gpt-5.4,claude-opus-4-6:Claude:0.1,grok-4.3-fast")
    assert arena.models_from_env() == ["gpt-5.4", "claude-opus-4-6", "grok-4.3-fast"]


def test_arena_ranks_models():
    # Real-name (non-anon) event cases with ground truth.
    cases = [c for c in load_event_suite().cases if not c.is_anonymized][:5]

    def predict(case, model):
        gt = case.event_ground_truth.direction_label
        if model == "good":                       # always correct
            return EventPrediction(direction=gt)
        if model == "ok":                          # correct on half
            return EventPrediction(direction=gt if hash(case.case_id) % 2 else "wrong-x")
        return EventPrediction(direction="always-wrong")   # never correct

    out = arena.run_arena(cases, ["good", "ok", "bad"], predict)
    lb = {r["model"]: r for r in out["leaderboard"]}
    assert lb["good"]["accuracy"] == 1.0
    assert lb["bad"]["accuracy"] == 0.0
    assert 0.0 <= lb["ok"]["accuracy"] <= 1.0
    # ranked best-first
    assert out["leaderboard"][0]["model"] == "good"
    assert out["leaderboard"][-1]["model"] == "bad"
    assert len(out["per_case"]) == len(cases)


def test_arena_predict_failure_is_a_miss():
    cases = [c for c in load_event_suite().cases if not c.is_anonymized][:2]

    def boom(case, model):
        raise RuntimeError("model down")

    out = arena.run_arena(cases, ["m1"], boom)
    assert out["leaderboard"][0]["accuracy"] == 0.0    # all misses, no crash
