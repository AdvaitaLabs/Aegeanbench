"""
SurveyRunner — benchmark the Loka population/survey subsystem (the camel-free
Aaru-style path) against realized outcomes.

Per case it drives Loka:
    1. POST /api/population/build   (marginals + geo from the case)
    2. POST /api/survey/run         (the case's question(s))
    3. GET  /api/survey/<id>/results (weighted dist / geo roll-up / means)
then extracts a prediction and scores it against the case's EventGroundTruth
(reusing the event scoring from aegeanbench.scoring).

The HTTP flow is injectable via `results_fetcher` so extraction + scoring are
unit-testable without a live Loka. Unlike the OASIS LokaRunner, this only needs
Loka's lightweight camel-free backend running.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkResult,
    ConsensusOutcome,
    EventPrediction,
)
from aegeanbench.scoring import apply_memorization_gap, score_prediction

# (case) -> aggregated results dict (same shape as GET /api/survey/<id>/results .data)
ResultsFetcher = Callable[[BenchmarkCase], Dict[str, Any]]


def extract_prediction(results: Dict[str, Any], spec: Dict[str, Any]) -> Optional[EventPrediction]:
    """
    Turn a survey `summarize` result into an EventPrediction per `spec`:
      {"qid": <question id>, "mode": "geo_leader" | "top_choice" | "mean"}
    """
    if not results or not spec:
        return None
    q = (results.get("questions") or {}).get(spec.get("qid"))
    if not q:
        return None
    mode = spec.get("mode", "top_choice")

    if mode == "geo_leader":
        leader = (q.get("geo_rollup") or {}).get("leader")
        return EventPrediction(direction=leader) if leader is not None else None
    if mode == "mean":
        m = q.get("mean")
        return EventPrediction(point_estimate=m) if m is not None else None
    # top_choice (default)
    dist = q.get("distribution") or {}
    if not dist:
        return None
    winner = max(dist, key=dist.get)
    ci = (q.get("ci_80") or {}).get(winner)
    return EventPrediction(direction=winner, point_estimate=dist[winner],
                           ci_80=list(ci) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None)


class SurveyRunner:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5001",
        *,
        timeout_s: float = 600.0,
        results_fetcher: Optional[ResultsFetcher] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._fetch = results_fetcher or self._http_run

    # ─────────────────────── public API ───────────────────────

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        if case.category != BenchmarkCategory.EVENT or case.event_ground_truth is None:
            raise ValueError(f"SurveyRunner needs an EVENT case with ground truth: {case.case_id}")

        spec = (case.scenario_request or {}).get("predict", {})
        error: Optional[str] = None
        results: Dict[str, Any] = {}
        t0 = time.time()
        try:
            results = self._fetch(case) or {}
        except Exception as e:  # noqa: BLE001
            error = str(e)
        latency = time.time() - t0

        prediction = extract_prediction(results, spec)
        metrics = score_prediction(prediction, case.event_ground_truth,
                                   is_anonymized=case.is_anonymized)
        outcome = ConsensusOutcome.CONVERGED if prediction is not None else ConsensusOutcome.ERROR
        return BenchmarkResult(
            case_id=case.case_id, case_name=case.name,
            category=BenchmarkCategory.EVENT, difficulty=case.difficulty,
            outcome=outcome, correct=metrics.direction_correct,
            final_answer=(prediction.direction if prediction else None),
            latency_s=round(latency, 3), event_metrics=metrics,
            raw_output={"results": results, "is_anonymized": case.is_anonymized,
                        "twin_case_id": case.twin_case_id},
            error=error,
        )

    def run_suite(self, cases: List[BenchmarkCase]) -> List[BenchmarkResult]:
        results = [self.run_case(c) for c in cases]
        self._pair_gaps(cases, results)
        return results

    @staticmethod
    def _pair_gaps(cases, results):
        by_id = {r.case_id: r for r in results}
        seen = set()
        for c in cases:
            if c.is_anonymized or not c.twin_case_id:
                continue
            key = frozenset((c.case_id, c.twin_case_id))
            if key in seen:
                continue
            real, anon = by_id.get(c.case_id), by_id.get(c.twin_case_id)
            if real and anon and real.event_metrics and anon.event_metrics:
                apply_memorization_gap(real.event_metrics, anon.event_metrics)
                seen.add(key)

    # ─────────────────────── HTTP flow ────────────────────────

    def _http_run(self, case: BenchmarkCase) -> Dict[str, Any]:
        """
        Drive Loka's lightweight survey backend. Validate against a running
        Loka before trusting scores; extraction/scoring are unit-tested via an
        injected results_fetcher. Expects case.scenario_request to carry:
          {marginals, size?, geo?, questions, predict}
        """
        import requests
        req = case.scenario_request or {}
        pop_id = f"bench-{case.case_id}"

        # Electoral / multi-geo cases build one sub-population per GeoScope leaf.
        if req.get("geo") and req.get("unit_marginals"):
            build = requests.post(f"{self.base_url}/api/population/build-geo", timeout=self.timeout_s,
                                  json={"geo": req["geo"], "unit_marginals": req["unit_marginals"],
                                        "sizes": req.get("sizes"),
                                        "default_size": req.get("default_size", 500),
                                        "population_id": pop_id, "async": False})
        else:
            build = requests.post(f"{self.base_url}/api/population/build", timeout=self.timeout_s,
                                  json={"marginals": req["marginals"], "size": req.get("size", 1000),
                                        "geo_id": req.get("geo_id"), "population_id": pop_id})
        build.raise_for_status()

        run = requests.post(f"{self.base_url}/api/survey/run", timeout=self.timeout_s,
                            json={"population_id": pop_id, "questions": req["questions"],
                                  "sample_size": req.get("sample_size")})
        run.raise_for_status()
        survey_id = (run.json().get("data") or {}).get("survey_id")
        if not survey_id:
            raise RuntimeError(f"no survey_id for {case.case_id}")

        res = requests.get(f"{self.base_url}/api/survey/{survey_id}/results", timeout=self.timeout_s)
        res.raise_for_status()
        return res.json().get("data") or {}
