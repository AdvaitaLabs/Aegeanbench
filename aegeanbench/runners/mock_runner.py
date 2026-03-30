"""
Mock Runner — runs benchmark cases without a real LLM.

Simulates agent behaviour deterministically so the benchmark
framework itself can be validated without API keys or cost.

Behaviour:
  - Consensus cases  : agents "answer" from the case's answer_variants;
                       outlier agents return a wrong answer.
  - Collaboration    : all subtasks are marked completed.
  - Risk cases       : decision is looked up from expected_decision.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

from aegeanbench.core.models import (
    AgentSnapshot,
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkResult,
    CollaborationMetrics,
    ConsensusOutcome,
    Difficulty,
    ExpectedDecision,
    RiskMetrics,
)
from aegeanbench.metrics.engine import MetricsEngine


class MockRunner:
    """
    Deterministic mock runner for framework testing.

    Usage::

        runner = MockRunner(seed=42)
        result = runner.run_case(case)
    """

    def __init__(self, seed: int = 42, base_latency_s: float = 0.05):
        self.rng            = random.Random(seed)
        self.base_latency_s = base_latency_s
        self.metrics        = MetricsEngine()

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single benchmark case and return a BenchmarkResult."""
        start = time.perf_counter()
        try:
            if case.category in (BenchmarkCategory.CONSENSUS, BenchmarkCategory.HYBRID):
                result = self._run_consensus(case)
            elif case.category == BenchmarkCategory.COLLABORATION:
                result = self._run_collaboration(case)
            elif case.category == BenchmarkCategory.RISK:
                result = self._run_risk(case)
            else:
                result = self._error_result(case, "Unknown category")
        except Exception as exc:  # noqa: BLE001
            result = self._error_result(case, str(exc))

        result.latency_s = round(time.perf_counter() - start, 4)
        return result

    def run_suite(self, cases: List[BenchmarkCase]) -> List[BenchmarkResult]:
        """Run all cases in a suite."""
        return [self.run_case(c) for c in cases]

    # ──────────────────────────────────────────────────────────────
    # Consensus simulation
    # ──────────────────────────────────────────────────────────────

    def _run_consensus(self, case: BenchmarkCase) -> BenchmarkResult:
        n             = max(1, case.num_agents)
        correct_ans   = case.expected_answer or (case.answer_variants[0] if case.answer_variants else "unknown")
        wrong_ans     = "WRONG_ANSWER"
        outlier_idx   = case.outlier_agent_idx if case.has_outlier_agent else None
        num_outliers  = case.metadata.get("num_outliers", 1) if case.has_outlier_agent else 0
        outlier_idxs  = set(case.metadata.get("outlier_indices", [outlier_idx] if outlier_idx is not None else []))

        weights = case.agent_capability_weights or [1.0] * n
        if len(weights) < n:
            weights = weights + [1.0] * (n - len(weights))

        snapshots: List[AgentSnapshot] = []
        agent_answers: List[str]       = []

        for i in range(n):
            is_outlier = i in outlier_idxs
            answer     = wrong_ans if is_outlier else correct_ans
            if case.inject_noise and not is_outlier:
                # Occasionally paraphrase
                variants = case.answer_variants or [correct_ans]
                answer   = self.rng.choice(variants)

            # Simulate that max-rounds-exhaustion cases diverge
            if case.metadata.get("expect_consensus") is False:
                answer = self.rng.choice(["A", "B", "C", "D"])

            snapshots.append(AgentSnapshot(
                agent_id=f"mock-agent-{i}",
                answer=answer,
                confidence=self.rng.uniform(0.7, 0.99) if not is_outlier else self.rng.uniform(0.3, 0.6),
                capability_weight=weights[i],
                was_outlier=is_outlier,
                latency_s=round(self.base_latency_s * self.rng.uniform(0.8, 1.5), 4),
            ))
            agent_answers.append(answer)

        # Determine consensus
        from collections import Counter
        vote_counts   = Counter(agent_answers)
        top_ans, top_n = vote_counts.most_common(1)[0]
        quorum_needed = max(1, round(n * case.quorum_threshold))

        reached  = top_n >= quorum_needed
        final    = top_ans if reached else None

        # Cap rounds used
        rounds_used = 1 if reached else case.max_rounds
        early_stop  = reached and rounds_used < case.max_rounds

        # Use expect_consensus override
        if case.metadata.get("expect_consensus") is False:
            reached    = False
            final      = None
            rounds_used = case.max_rounds
            early_stop  = False

        outcome = ConsensusOutcome.CONVERGED if reached else ConsensusOutcome.DIVERGED
        if early_stop:
            outcome = ConsensusOutcome.EARLY_STOP

        correct = False
        if final and case.expected_answer:
            correct = final.lower().strip() in [v.lower().strip() for v in
                      ([case.expected_answer] + case.answer_variants)]

        con_metrics = self.metrics.compute_consensus_metrics(
            agent_answers=agent_answers,
            final_answer=final,
            rounds_used=rounds_used,
            max_rounds=case.max_rounds,
            early_stop=early_stop,
            has_outlier=case.has_outlier_agent,
            weighted_votes={k: v for k, v in vote_counts.items()},
        )

        # Simulate token consumption:
        # Each agent call = ~200 prompt tokens + ~50 completion tokens per round
        # Early stop saves tokens from cancelled agents
        agents_called   = len(snapshots)
        prompt_per_call = 200 + len(case.task or "") // 4
        comp_per_call   = 50
        tokens_prompt     = agents_called * rounds_used * prompt_per_call
        tokens_completion = agents_called * rounds_used * comp_per_call
        # Tokens saved = agents cancelled by early-stop × remaining rounds
        cancelled = max(0, n - case.num_agents + (n - len(snapshots)))
        tokens_saved = cancelled * (case.max_rounds - rounds_used) * (
            prompt_per_call + comp_per_call
        ) if early_stop else 0

        # Update snapshots with per-agent token counts
        for snap in snapshots:
            snap.tokens_prompt     = prompt_per_call * rounds_used
            snap.tokens_completion = comp_per_call   * rounds_used

        # Update ConsensusMetrics token fields
        con_metrics.tokens_prompt_total     = tokens_prompt
        con_metrics.tokens_completion_total = tokens_completion
        con_metrics.tokens_per_round        = [
            agents_called * (prompt_per_call + comp_per_call)
        ] * rounds_used
        total_tok = tokens_prompt + tokens_completion
        con_metrics.token_efficiency = round(
            1.0 / total_tok if total_tok > 0 and correct else 0.0, 6
        )

        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=outcome,
            correct=correct,
            final_answer=final,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_saved=tokens_saved,
            agent_snapshots=snapshots,
            consensus_metrics=con_metrics,
            raw_output={
                "has_outlier_agent": case.has_outlier_agent,
                "agent_answers": agent_answers,
                "vote_counts": dict(vote_counts),
            },
        )

    # ──────────────────────────────────────────────────────────────
    # Collaboration simulation
    # ──────────────────────────────────────────────────────────────

    def _run_collaboration(self, case: BenchmarkCase) -> BenchmarkResult:
        n         = max(1, case.num_agents)
        subtasks  = case.subtasks or []
        milestones = case.milestones or []

        # In mock mode, assume all subtasks completed unless redundancy test
        max_redundancy  = case.metadata.get("max_allowed_redundancy", 0.0)
        redundancy_rate = self.rng.uniform(0.0, max_redundancy * 0.8) if max_redundancy else 0.0
        completed       = len(subtasks)
        coordination    = self.rng.uniform(0.75, 0.98)

        snapshots = [
            AgentSnapshot(
                agent_id=f"mock-agent-{i}",
                answer=f"Completed subtask {i}",
                confidence=self.rng.uniform(0.8, 0.99),
                latency_s=round(self.base_latency_s * self.rng.uniform(1.0, 2.0), 4),
            )
            for i in range(n)
        ]

        col_metrics = CollaborationMetrics(
            task_completed=True,
            subtasks_total=len(subtasks),
            subtasks_completed=completed,
            subtask_completion_rate=1.0,
            coordination_score=round(coordination, 4),
            redundancy_rate=round(redundancy_rate, 4),
            milestone_hit_rate=1.0 if milestones else 0.0,
        )

        # Token simulation for collaboration: each subtask = one agent LLM call
        prompt_per_subtask = 300
        comp_per_subtask   = 120
        tokens_prompt     = len(subtasks) * prompt_per_subtask
        tokens_completion = len(subtasks) * comp_per_subtask
        for snap in snapshots:
            snap.tokens_prompt     = prompt_per_subtask
            snap.tokens_completion = comp_per_subtask

        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=ConsensusOutcome.CONVERGED,
            correct=True,
            final_answer="all_subtasks_completed",
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_saved=0,
            agent_snapshots=snapshots,
            collaboration_metrics=col_metrics,
        )

    # ──────────────────────────────────────────────────────────────
    # Risk simulation
    # ──────────────────────────────────────────────────────────────

    def _run_risk(self, case: BenchmarkCase) -> BenchmarkResult:
        expected = case.expected_decision or ExpectedDecision.APPROVE
        actual   = expected.value  # Mock: always return expected (perfect oracle)

        expect_pre_screen = case.metadata.get("expect_pre_screen", False)

        risk_metrics = RiskMetrics(
            expected_decision=expected,
            actual_decision=actual,
            expected_risk_level=case.expected_risk_level or "low",
            actual_risk_level=case.expected_risk_level or "low",
            decision_correct=True,
            risk_level_correct=True,
            pre_screen_triggered=expect_pre_screen,
            validator_agreement=self.rng.uniform(0.8, 1.0),
            difficulty_level=case.metadata.get("expected_difficulty", "simple"),
            participating_validators=case.metadata.get(
                "expected_validators", ["amount", "identity"]
            ),
        )

        snapshot = AgentSnapshot(
            agent_id="van-pipeline",
            answer=actual,
            confidence=self.rng.uniform(0.75, 0.99),
            latency_s=round(self.base_latency_s * self.rng.uniform(1.0, 3.0), 4),
        )

        # Token simulation for risk:
        # Pre-screen cases use 0 LLM tokens (deterministic rules)
        # Other cases: each active validator = ~400 prompt + 100 completion tokens
        num_validators  = len(case.metadata.get("expected_validators", ["amount", "identity"]))
        if expect_pre_screen:
            tokens_prompt     = 0
            tokens_completion = 0
            tokens_saved      = num_validators * 500  # saved by not calling LLM
            risk_metrics.tokens_saved_by_prescreen = tokens_saved
        else:
            tokens_prompt     = num_validators * 400
            tokens_completion = num_validators * 100
            tokens_saved      = 0
        risk_metrics.tokens_prompt_total     = tokens_prompt
        risk_metrics.tokens_completion_total = tokens_completion
        snapshot.tokens_prompt     = tokens_prompt
        snapshot.tokens_completion = tokens_completion

        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=ConsensusOutcome.CONVERGED,
            correct=True,
            final_answer=actual,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            tokens_saved=tokens_saved,
            agent_snapshots=[snapshot],
            risk_metrics=risk_metrics,
        )

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _error_result(case: BenchmarkCase, msg: str) -> BenchmarkResult:
        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=ConsensusOutcome.ERROR,
            correct=False,
            error=msg,
        )

