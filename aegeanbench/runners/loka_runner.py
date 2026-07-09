"""
LokaRunner — drives the Loka (MiroFish-based) multi-agent simulation for
event/scenario backtest cases and scores the result.

Flow per case:
    1. POST the scenario (question + point-in-time public_facts) to Loka's
       /api/project/run and get a project id.
    2. Poll /api/project/<id>/status until the pipeline completes.
    3. Fetch the report markdown and extract the machine-readable
       AEGEANBENCH:PREDICTION block Loka appends in benchmark mode.
    4. Score the prediction against the case's realized EventGroundTruth.

After a suite runs, real/anon twins are paired to compute the memorization
gap.

IMPORTANT — the Loka server MUST run in no-leak benchmark mode, i.e. started
with ``LOKA_BENCHMARK_NO_LEAK=true`` (blocks live web_search) and
``LOKA_EMIT_PREDICTION=true`` (emits the prediction block). Otherwise the
backtest leaks post-analysis-date info and/or has no block to score.

The HTTP flow is injectable via ``report_fetcher`` so scoring/pairing can be
unit-tested without a live Loka. The default HTTP path (`_http_fetch_report`)
should be validated end-to-end against a running Loka before trusting scores.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkResult,
    ConsensusOutcome,
)
from aegeanbench.scoring import (
    apply_memorization_gap,
    extract_prediction,
    score_prediction,
)

# (case) -> report markdown string
ReportFetcher = Callable[[BenchmarkCase], str]


class LokaRunner:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5001",
        *,
        poll_interval: float = 3.0,
        timeout_s: float = 1800.0,
        max_rounds: int = 30,
        report_fetcher: Optional[ReportFetcher] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout_s = timeout_s
        self.max_rounds = max_rounds
        self._fetch_report: ReportFetcher = report_fetcher or self._http_fetch_report

    # ─────────────────────── public API ───────────────────────

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run one event case through Loka and score it."""
        if case.category != BenchmarkCategory.EVENT or case.event_ground_truth is None:
            raise ValueError(f"LokaRunner only handles EVENT cases with ground truth: {case.case_id}")

        error: Optional[str] = None
        report_md = ""
        t0 = time.time()
        try:
            report_md = self._fetch_report(case) or ""
        except Exception as e:  # noqa: BLE001
            error = str(e)
        latency = time.time() - t0

        prediction = extract_prediction(report_md)
        metrics = score_prediction(
            prediction, case.event_ground_truth, is_anonymized=case.is_anonymized
        )
        outcome = ConsensusOutcome.CONVERGED if prediction is not None else ConsensusOutcome.ERROR

        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=BenchmarkCategory.EVENT,
            difficulty=case.difficulty,
            outcome=outcome,
            correct=metrics.direction_correct,
            final_answer=(prediction.direction if prediction else None),
            latency_s=round(latency, 3),
            event_metrics=metrics,
            raw_output={"report_markdown": report_md,
                        "is_anonymized": case.is_anonymized,
                        "twin_case_id": case.twin_case_id},
            error=error,
        )

    def run_suite(self, cases: List[BenchmarkCase]) -> List[BenchmarkResult]:
        """Run every case, then pair real/anon twins to fill memorization_gap."""
        results = [self.run_case(c) for c in cases]
        self._pair_memorization_gaps(cases, results)
        return results

    # ─────────────────────── twin pairing ─────────────────────

    @staticmethod
    def _pair_memorization_gaps(
        cases: List[BenchmarkCase], results: List[BenchmarkResult]
    ) -> None:
        by_id = {r.case_id: r for r in results}
        case_by_id = {c.case_id: c for c in cases}
        done: set = set()
        for c in cases:
            if c.is_anonymized or not c.twin_case_id:
                continue  # drive pairing from the real twin only
            pair_key = frozenset((c.case_id, c.twin_case_id))
            if pair_key in done:
                continue
            real_res = by_id.get(c.case_id)
            anon_res = by_id.get(c.twin_case_id)
            if (real_res and anon_res and real_res.event_metrics and anon_res.event_metrics
                    and c.twin_case_id in case_by_id):
                apply_memorization_gap(real_res.event_metrics, anon_res.event_metrics)
                done.add(pair_key)

    # ─────────────────────── HTTP flow ────────────────────────

    def _http_fetch_report(self, case: BenchmarkCase) -> str:
        """
        Drive Loka over HTTP and return the report markdown.

        NOTE: validate this against a live Loka before trusting scores — the
        exact request/response shape may need tweaking to match the deployed
        /api/project endpoints. Scoring/pairing are covered by unit tests via
        an injected report_fetcher; this method is the only un-mocked part.
        """
        import requests  # local import so the module loads without requests

        req = case.scenario_request or {}
        facts = req.get("public_facts") or []
        # Only T-relative facts go in; the Loka server must run in no-leak mode
        # so it cannot pull post-analysis-date info to fill gaps.
        requirement = req.get("question") or case.description
        doc = "# Scenario facts (as of {})\n\n{}".format(
            req.get("analysis_date", "the analysis date"),
            "\n".join(f"- {f}" for f in facts),
        )

        run = requests.post(
            f"{self.base_url}/api/project/run",
            json={"requirement": requirement,
                  "project_name": case.case_id,
                  "document": doc,
                  "max_rounds": self.max_rounds},
            timeout=60,
        )
        run.raise_for_status()
        project_id = (run.json().get("data") or {}).get("project_id") \
            or run.json().get("project_id")
        if not project_id:
            raise RuntimeError(f"no project_id in run response for {case.case_id}")

        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            st = requests.get(f"{self.base_url}/api/project/{project_id}/status", timeout=30)
            st.raise_for_status()
            sd = st.json().get("data") or st.json()
            status = sd.get("status")
            if status in ("completed", "done", "finished"):
                break
            if status in ("failed", "error"):
                raise RuntimeError(f"Loka run failed for {case.case_id}: {sd}")
            time.sleep(self.poll_interval)
        else:
            raise TimeoutError(f"Loka run timed out for {case.case_id}")

        data = requests.get(f"{self.base_url}/api/project/{project_id}/data", timeout=60)
        data.raise_for_status()
        dd = data.json().get("data") or data.json()
        # report markdown may be surfaced under a few keys depending on version
        return (dd.get("report_markdown")
                or (dd.get("report") or {}).get("markdown_content")
                or dd.get("markdown_content")
                or "")
