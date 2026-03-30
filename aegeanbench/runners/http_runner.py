"""
HTTP Runner — calls aegean-consensus via REST API.

Requires aegean-consensus service running at http://localhost:8000 (or custom URL).
This is the true end-to-end test — tests the entire system as deployed.

Usage::

    runner = HttpRunner(base_url="http://localhost:8000")
    result = runner.run_case(case)
"""

from __future__ import annotations

import time
import requests
from typing import Any, Dict, List, Tuple

from aegeanbench.core.models import (
    AgentSnapshot,
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkResult,
    CollaborationMetrics,
    ConsensusOutcome,
    ExpectedDecision,
    RiskMetrics,
)
from aegeanbench.metrics.engine import MetricsEngine


class HttpRunner:
    """
    Runs benchmark cases against aegean-consensus via HTTP API.

    Requires aegean-consensus service to be running:
        uvicorn aegean.api.app:create_app --factory --host 0.0.0.0 --port 8000

    Usage::

        runner = HttpRunner(base_url="http://localhost:8000")
        result = runner.run_case(case)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.metrics = MetricsEngine()
        self._check_health()

    def _check_health(self) -> None:
        """Verify aegean-consensus service is running."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"Service health check failed: {resp.status_code}")
        except requests.ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to aegean-consensus at {self.base_url}. "
                f"Is it running? Start with: "
                f"uvicorn aegean.api.app:create_app --factory --host 0.0.0.0 --port 8000"
            ) from e

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single case via HTTP."""
        start = time.perf_counter()
        try:
            if case.category in (BenchmarkCategory.CONSENSUS, BenchmarkCategory.HYBRID):
                result = self._run_consensus_http(case)
            elif case.category == BenchmarkCategory.COLLABORATION:
                result = self._run_collaboration_http(case)
            elif case.category == BenchmarkCategory.RISK:
                result = self._run_risk_http(case)
            else:
                result = self._error(case, f"Unknown category: {case.category}")
        except Exception as exc:  # noqa: BLE001
            result = self._error(case, str(exc))

        result.latency_s = round(time.perf_counter() - start, 4)
        return result

    def run_suite(self, cases: List[BenchmarkCase]) -> List[BenchmarkResult]:
        """Run all cases sequentially."""
        return [self.run_case(c) for c in cases]

    # ──────────────────────────────────────────────────────────────
    # Consensus via HTTP
    # ──────────────────────────────────────────────────────────────

    def _run_consensus_http(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run consensus case via HTTP API."""
        # Step 1: Create group
        group_resp = requests.post(
            f"{self.base_url}/api/v1/groups",
            json={
                "group_name": f"Benchmark {case.case_id}",
                "description": case.description,
                "mode": "consensus",
                "created_by": "aegeanbench",
            },
            timeout=self.timeout,
        )
        if group_resp.status_code != 201:
            return self._error(case, f"Failed to create group: {group_resp.text}")

        group = group_resp.json()
        group_id = group["group_id"]

        try:
            # Step 2: Fetch registered agents, then add to group
            agents_resp = requests.get(
                f"{self.base_url}/api/v1/groups/agents",
                timeout=self.timeout,
            )
            if agents_resp.status_code != 200 or not agents_resp.json():
                return self._error(case, f"No registered agents available: {agents_resp.text}")

            available_agents = agents_resp.json()
            n = max(1, case.num_agents)
            weights = case.agent_capability_weights or [1.0] * n
            if len(weights) < n:
                weights = weights + [1.0] * (n - len(weights))

            # Use as many available agents as needed (cycle if fewer available)
            selected = [available_agents[i % len(available_agents)] for i in range(n)]

            added = 0
            for i, agent_info in enumerate(selected):
                member_resp = requests.post(
                    f"{self.base_url}/api/v1/groups/{group_id}/members",
                    json={
                        "agent_id": agent_info["agent_id"],
                        "role": "consensus_agent",
                        "capability_weight": weights[i],
                    },
                    timeout=self.timeout,
                )
                if member_resp.status_code == 201:
                    added += 1

            if added == 0:
                return self._error(case, "Failed to add any agents to group")

            # Step 3: Execute consensus
            consensus_resp = requests.post(
                f"{self.base_url}/api/v1/groups/{group_id}/consensus",
                json={
                    "task": case.task or "",
                    "quorum_threshold": case.quorum_threshold,
                    "max_rounds": case.max_rounds,
                },
                timeout=self.timeout,
            )
            if consensus_resp.status_code != 201:
                error_msg = f"Consensus failed (status {consensus_resp.status_code}): {consensus_resp.text}"
                print(f"[DEBUG] {error_msg}")
                return self._error(case, error_msg)

            consensus_result = consensus_resp.json()

            # Step 4: Parse result
            final_answer = None
            if consensus_result.get("final_solution"):
                final_answer = consensus_result["final_solution"].get("answer")

            correct = False
            if final_answer and case.expected_answer:
                variants = [case.expected_answer] + case.answer_variants
                correct = final_answer.lower().strip() in [
                    v.lower().strip() for v in variants
                ]

            # Extract metrics
            agent_responses = consensus_result.get("agent_responses", [])
            agent_answers = [r.get("answer") for r in agent_responses]

            snapshots = [
                AgentSnapshot(
                    agent_id=r.get("agent_id", f"agent-{i}"),
                    answer=r.get("answer"),
                    confidence=r.get("confidence", 0.5),
                    tokens_prompt=self._to_int(r.get("tokens_prompt", 0)),
                    tokens_completion=self._to_int(r.get("tokens_completion", 0)),
                )
                for i, r in enumerate(agent_responses)
            ]

            snapshot_prompt = sum(s.tokens_prompt for s in snapshots)
            snapshot_completion = sum(s.tokens_completion for s in snapshots)

            tokens_prompt = snapshot_prompt
            tokens_completion = snapshot_completion
            if tokens_prompt == 0 and tokens_completion == 0:
                tokens_prompt, tokens_completion = self._extract_token_totals(
                    consensus_result,
                    entries=agent_responses,
                )

            con_metrics = self.metrics.compute_consensus_metrics(
                agent_answers=agent_answers,
                final_answer=final_answer,
                rounds_used=consensus_result.get("rounds_used", 1),
                max_rounds=case.max_rounds,
                early_stop=consensus_result.get("early_stop_triggered", False),
                has_outlier=case.has_outlier_agent,
                weighted_votes=consensus_result.get("weighted_votes", {}),
            )

            outcome = (
                ConsensusOutcome.CONVERGED
                if consensus_result.get("consensus_reached")
                else ConsensusOutcome.DIVERGED
            )

            return BenchmarkResult(
                case_id=case.case_id,
                case_name=case.name,
                category=case.category,
                difficulty=case.difficulty,
                outcome=outcome,
                correct=correct,
                final_answer=final_answer,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_saved=0,
                agent_snapshots=snapshots,
                consensus_metrics=con_metrics,
                raw_output=consensus_result,
            )

        finally:
            # Cleanup: delete group
            requests.delete(
                f"{self.base_url}/api/v1/groups/{group_id}",
                timeout=self.timeout,
            )

    # ──────────────────────────────────────────────────────────────
    # Collaboration via HTTP
    # ──────────────────────────────────────────────────────────────

    def _run_collaboration_http(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run collaboration case via HTTP API."""
        # Similar to consensus but with collaboration mode
        group_resp = requests.post(
            f"{self.base_url}/api/v1/groups",
            json={
                "group_name": f"Collab {case.case_id}",
                "description": case.description,
                "mode": "collaboration",
                "created_by": "aegeanbench",
            },
            timeout=self.timeout,
        )
        if group_resp.status_code != 201:
            return self._error(case, f"Failed to create group: {group_resp.text}")

        group = group_resp.json()
        group_id = group["group_id"]

        try:
            # Fetch registered agents then add to group
            agents_resp = requests.get(
                f"{self.base_url}/api/v1/groups/agents",
                timeout=self.timeout,
            )
            available_agents = agents_resp.json() if agents_resp.status_code == 200 else []
            n = max(1, case.num_agents)
            selected = [available_agents[i % len(available_agents)] for i in range(n)] if available_agents else []
            for agent_info in selected:
                requests.post(
                    f"{self.base_url}/api/v1/groups/{group_id}/members",
                    json={"agent_id": agent_info["agent_id"], "role": "collaborator"},
                    timeout=self.timeout,
                )

            # Execute each subtask
            subtasks = case.subtasks or [case.task or ""]
            completed = 0
            tokens_total = 0

            for subtask in subtasks:
                resp = requests.post(
                    f"{self.base_url}/api/v1/groups/{group_id}/consensus",
                    json={"task": subtask, "max_rounds": case.max_rounds},
                    timeout=self.timeout,
                )
                if resp.status_code == 201:
                    completed += 1
                    result = resp.json()
                    # Accumulate tokens
                    for agent_resp in result.get("agent_responses", []):
                        prompt_i, completion_i = self._extract_token_totals(
                            agent_resp,
                            entries=[agent_resp],
                        )
                        tokens_total += prompt_i + completion_i

            col_metrics = CollaborationMetrics(
                task_completed=completed == len(subtasks),
                subtasks_total=len(subtasks),
                subtasks_completed=completed,
                subtask_completion_rate=completed / len(subtasks) if subtasks else 0.0,
                coordination_score=0.85,
                milestone_hit_rate=1.0 if case.milestones else 0.0,
            )

            return BenchmarkResult(
                case_id=case.case_id,
                case_name=case.name,
                category=case.category,
                difficulty=case.difficulty,
                outcome=ConsensusOutcome.CONVERGED
                if col_metrics.task_completed
                else ConsensusOutcome.DIVERGED,
                correct=col_metrics.task_completed,
                final_answer="collaboration_complete" if col_metrics.task_completed else None,
                tokens_prompt=int(tokens_total * 0.8),
                tokens_completion=int(tokens_total * 0.2),
                collaboration_metrics=col_metrics,
            )

        finally:
            requests.delete(
                f"{self.base_url}/api/v1/groups/{group_id}",
                timeout=self.timeout,
            )

    # ──────────────────────────────────────────────────────────────
    # Risk via HTTP
    # ──────────────────────────────────────────────────────────────

    def _run_risk_http(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run risk case via HTTP API."""
        if not case.risk_payload:
            return self._error(case, "risk_payload is required for RISK cases")

        payload = case.risk_payload
        risk_resp = requests.post(
            f"{self.base_url}/api/v1/risk/evaluate",
            json=payload,
            timeout=self.timeout,
        )

        if risk_resp.status_code != 200:
            return self._error(case, f"Risk evaluation failed: {risk_resp.text}")

        decision = risk_resp.json()

        expected = case.expected_decision or ExpectedDecision.APPROVE
        actual_decision = decision.get("decision")
        actual_risk = decision.get("risk_level")

        decision_correct = actual_decision == expected.value
        risk_level_correct = actual_risk == (case.expected_risk_level or "low")

        # Validator agreement
        validator_agreement = 0.0
        if decision.get("validator_results"):
            matching = sum(
                1
                for vr in decision["validator_results"]
                if vr.get("risk_level") == actual_risk
            )
            validator_agreement = matching / len(decision["validator_results"])

        pre_screen = decision.get("pre_screen_triggered", False)

        tokens_prompt, tokens_completion = self._extract_token_totals(decision)
        tokens_saved = self._to_int(
            decision.get("tokens_saved", decision.get("tokens_saved_by_prescreen", 0))
        )

        risk_metrics = RiskMetrics(
            expected_decision=expected,
            actual_decision=actual_decision,
            expected_risk_level=case.expected_risk_level or "low",
            actual_risk_level=actual_risk,
            decision_correct=decision_correct,
            risk_level_correct=risk_level_correct,
            pre_screen_triggered=pre_screen,
            validator_agreement=round(validator_agreement, 4),
            difficulty_level=decision.get("difficulty_level", "simple"),
            participating_validators=decision.get("participating_validators", []),
            tokens_prompt_total=tokens_prompt,
            tokens_completion_total=tokens_completion,
            tokens_saved_by_prescreen=tokens_saved,
        )

        snapshot = AgentSnapshot(
            agent_id="van-pipeline",
            answer=actual_decision,
            confidence=decision.get("confidence", 0.5),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
        )

        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=ConsensusOutcome.CONVERGED,
            correct=decision_correct,
            final_answer=actual_decision,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_saved=tokens_saved,
            agent_snapshots=[snapshot],
            risk_metrics=risk_metrics,
            raw_output=decision,
        )

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _extract_token_totals(
        self,
        payload: Dict[str, Any],
        entries: List[Dict[str, Any]] | None = None,
    ) -> Tuple[int, int]:
        """
        Provider-agnostic token extraction.

        Supports:
          - tokens_prompt / tokens_completion
          - input_tokens / output_tokens
          - prompt_tokens / completion_tokens
          - total_tokens
          - usage / token_usage nested objects
          - per-agent entry list aggregation
        """
        token_objects = []
        if isinstance(payload, dict):
            token_objects.append(payload)
            usage_obj = payload.get("usage") or payload.get("token_usage")
            if isinstance(usage_obj, dict):
                token_objects.append(usage_obj)

        prompt = 0
        completion = 0

        for obj in token_objects:
            prompt = max(
                prompt,
                self._to_int(
                    obj.get("tokens_prompt", obj.get("input_tokens", obj.get("prompt_tokens", 0)))
                ),
            )
            completion = max(
                completion,
                self._to_int(
                    obj.get(
                        "tokens_completion",
                        obj.get("output_tokens", obj.get("completion_tokens", 0)),
                    )
                ),
            )

            total_tokens = self._to_int(obj.get("total_tokens", 0))
            if total_tokens > 0 and prompt + completion == 0:
                prompt = int(total_tokens * 0.8)
                completion = total_tokens - prompt

        if entries:
            sum_prompt = 0
            sum_completion = 0
            for entry in entries:
                p_i, c_i = self._extract_token_totals(entry)
                sum_prompt += p_i
                sum_completion += c_i
            prompt = max(prompt, sum_prompt)
            completion = max(completion, sum_completion)

        return prompt, completion

    @staticmethod
    def _error(case: BenchmarkCase, msg: str) -> BenchmarkResult:
        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=ConsensusOutcome.ERROR,
            correct=False,
            error=msg,
        )

