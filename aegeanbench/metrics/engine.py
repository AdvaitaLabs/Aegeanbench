"""
AegeanBench Metrics Engine.

Computes all benchmark metrics from raw BenchmarkResult objects
and aggregates them into BenchmarkSuiteResult.

Novel metrics introduced by AegeanBench (absent from AutoGenBench / MultiAgentBench):
  - consensus_rate          : fraction of cases where consensus was reached
  - disagreement_score      : normalised agent disagreement (0=full agreement)
  - quorum_efficiency       : how early the system terminated vs max allowed
  - early_stop_rate         : fraction of cases that used early termination
  - outlier_detection_rate  : fraction of outlier cases where outlier was identified
  - risk_f1_{approve,reject}: F1 score on risk decision classification
  - pre_screen_rate         : fraction of risk cases resolved without LLM
  - validator_agreement     : mean fraction of validators aligned with final decision
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Optional

from aegeanbench.core.models import (
    BenchmarkCategory,
    BenchmarkResult,
    BenchmarkSuiteResult,
    ConsensusMetrics,
    ConsensusOutcome,
    Difficulty,
)


class MetricsEngine:
    """
    Computes per-result and aggregate metrics for an AegeanBench run.

    Usage::

        engine = MetricsEngine()
        suite_result = engine.aggregate(suite_id, suite_name, results)
    """

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def aggregate(
        self,
        suite_id: str,
        suite_name: str,
        results: List[BenchmarkResult],
    ) -> BenchmarkSuiteResult:
        """Aggregate a list of BenchmarkResults into a BenchmarkSuiteResult."""
        if not results:
            return BenchmarkSuiteResult(
                suite_id=suite_id,
                suite_name=suite_name,
                total_cases=0,
            )

        total   = len(results)
        passed  = sum(1 for r in results if r.correct)
        failed  = sum(1 for r in results if not r.correct and r.outcome != ConsensusOutcome.ERROR)
        errored = sum(1 for r in results if r.outcome == ConsensusOutcome.ERROR)

        latencies = sorted(r.latency_s for r in results)
        mean_lat  = sum(latencies) / total
        p50_lat   = self._percentile(latencies, 50)
        p95_lat   = self._percentile(latencies, 95)

        # ── Token metrics ────────────────────────────────────────
        total_tokens_prompt     = sum(r.tokens_prompt for r in results)
        total_tokens_completion = sum(r.tokens_completion for r in results)
        total_tokens            = total_tokens_prompt + total_tokens_completion
        total_tokens_saved      = sum(r.tokens_saved for r in results)
        token_counts            = sorted(r.tokens_total for r in results)
        mean_tokens_per_case    = total_tokens / total if total else 0.0
        p50_tokens              = self._percentile(token_counts, 50)
        p95_tokens              = self._percentile(token_counts, 95)
        token_efficiency_rate   = (
            total_tokens_saved / (total_tokens + total_tokens_saved)
            if (total_tokens + total_tokens_saved) > 0 else 0.0
        )
        correct_results = [r for r in results if r.correct]
        mean_tokens_per_correct = (
            sum(r.tokens_total for r in correct_results) / len(correct_results)
            if correct_results else 0.0
        )

        # ── Consensus metrics ────────────────────────────────────
        con_results = [r for r in results if r.category == BenchmarkCategory.CONSENSUS
                       and r.consensus_metrics is not None]

        consensus_rate         = 0.0
        mean_consensus_conf    = 0.0
        mean_disagreement      = 0.0
        early_stop_rate        = 0.0
        mean_quorum_efficiency = 0.0
        outlier_detection_rate = 0.0

        if con_results:
            n = len(con_results)
            consensus_rate      = sum(1 for r in con_results
                                      if r.consensus_metrics.quorum_reached) / n
            mean_consensus_conf = sum(r.consensus_metrics.consensus_confidence
                                      for r in con_results) / n
            mean_disagreement   = sum(r.consensus_metrics.disagreement_score
                                      for r in con_results) / n
            early_stop_rate     = sum(1 for r in con_results
                                      if r.consensus_metrics.early_stop_triggered) / n
            mean_quorum_efficiency = sum(r.consensus_metrics.quorum_efficiency
                                         for r in con_results) / n

            outlier_cases = [r for r in con_results
                             if r.raw_output.get("has_outlier_agent", False)]
            if outlier_cases:
                outlier_detection_rate = sum(
                    1 for r in outlier_cases if r.consensus_metrics.outlier_detected
                ) / len(outlier_cases)

        # ── Collaboration metrics ────────────────────────────────
        col_results = [r for r in results if r.category == BenchmarkCategory.COLLABORATION
                       and r.collaboration_metrics is not None]

        mean_subtask_completion = 0.0
        mean_milestone_hit      = 0.0
        mean_coordination       = 0.0

        if col_results:
            n = len(col_results)
            mean_subtask_completion = sum(r.collaboration_metrics.subtask_completion_rate
                                          for r in col_results) / n
            mean_milestone_hit      = sum(r.collaboration_metrics.milestone_hit_rate
                                          for r in col_results) / n
            mean_coordination       = sum(r.collaboration_metrics.coordination_score
                                          for r in col_results) / n

        # ── Risk metrics ─────────────────────────────────────────
        risk_results = [r for r in results if r.category == BenchmarkCategory.RISK
                        and r.risk_metrics is not None]

        risk_accuracy            = 0.0
        risk_level_accuracy      = 0.0
        risk_f1_approve          = 0.0
        risk_f1_reject           = 0.0
        mean_validator_agreement = 0.0
        pre_screen_rate          = 0.0

        if risk_results:
            n = len(risk_results)
            risk_accuracy       = sum(1 for r in risk_results
                                      if r.risk_metrics.decision_correct) / n
            risk_level_accuracy = sum(1 for r in risk_results
                                      if r.risk_metrics.risk_level_correct) / n
            mean_validator_agreement = sum(r.risk_metrics.validator_agreement
                                           for r in risk_results) / n
            pre_screen_rate     = sum(1 for r in risk_results
                                      if r.risk_metrics.pre_screen_triggered) / n

            risk_f1_approve = self._f1_for_class(
                risk_results, "approve",
                lambda r: r.risk_metrics.expected_decision.value,
                lambda r: r.risk_metrics.actual_decision or "",
            )
            risk_f1_reject = self._f1_for_class(
                risk_results, "reject",
                lambda r: r.risk_metrics.expected_decision.value,
                lambda r: r.risk_metrics.actual_decision or "",
            )

        return BenchmarkSuiteResult(
            suite_id=suite_id,
            suite_name=suite_name,
            total_cases=total,
            passed=passed,
            failed=failed,
            errored=errored,
            accuracy=passed / total,
            mean_latency_s=round(mean_lat, 4),
            p50_latency_s=round(p50_lat, 4),
            p95_latency_s=round(p95_lat, 4),
            # token aggregates
            total_tokens_prompt=total_tokens_prompt,
            total_tokens_completion=total_tokens_completion,
            total_tokens=total_tokens,
            total_tokens_saved=total_tokens_saved,
            mean_tokens_per_case=round(mean_tokens_per_case, 1),
            p50_tokens_per_case=round(p50_tokens, 1),
            p95_tokens_per_case=round(p95_tokens, 1),
            token_efficiency_rate=round(token_efficiency_rate, 4),
            mean_tokens_per_correct=round(mean_tokens_per_correct, 1),
            # consensus
            consensus_rate=round(consensus_rate, 4),
            mean_consensus_conf=round(mean_consensus_conf, 4),
            mean_disagreement=round(mean_disagreement, 4),
            early_stop_rate=round(early_stop_rate, 4),
            mean_quorum_efficiency=round(mean_quorum_efficiency, 4),
            outlier_detection_rate=round(outlier_detection_rate, 4),
            # collaboration
            mean_subtask_completion=round(mean_subtask_completion, 4),
            mean_milestone_hit=round(mean_milestone_hit, 4),
            mean_coordination=round(mean_coordination, 4),
            # risk
            risk_accuracy=round(risk_accuracy, 4),
            risk_level_accuracy=round(risk_level_accuracy, 4),
            risk_f1_approve=round(risk_f1_approve, 4),
            risk_f1_reject=round(risk_f1_reject, 4),
            mean_validator_agreement=round(mean_validator_agreement, 4),
            pre_screen_rate=round(pre_screen_rate, 4),
            results=results,
        )

    def compute_consensus_metrics(
        self,
        agent_answers: List[str],
        final_answer: Optional[str],
        rounds_used: int,
        max_rounds: int,
        early_stop: bool,
        has_outlier: bool,
        weighted_votes: Optional[dict] = None,
    ) -> ConsensusMetrics:
        """
        Compute ConsensusMetrics from raw agent answers.

        Args:
            agent_answers    : list of answer strings, one per agent
            final_answer     : the consensus answer (or None if not reached)
            rounds_used      : how many rounds were executed
            max_rounds       : max rounds allowed
            early_stop       : whether early termination fired
            has_outlier      : whether any agent is known to be an outlier
            weighted_votes   : optional dict {answer: weight}
        """
        if not agent_answers:
            return ConsensusMetrics()

        n = len(agent_answers)

        # Vote counts
        vote_counts: dict = defaultdict(int)
        for a in agent_answers:
            vote_counts[a] += 1

        top_answer, top_count = max(vote_counts.items(), key=lambda x: x[1])
        quorum_reached  = final_answer is not None
        quorum_size     = top_count if quorum_reached else 0

        # Disagreement score = 1 - (top_count / n)
        disagreement = 1.0 - (top_count / n) if n > 0 else 1.0

        # Consensus confidence: mean confidence of agents who voted for top answer
        # (we use 1.0 as default when no explicit confidences are available)
        consensus_confidence = top_count / n if quorum_reached else 0.0

        # Quorum efficiency: 1 - (rounds_used / max_rounds)  — higher is better
        quorum_efficiency = max(0.0, 1.0 - (rounds_used / max_rounds)) if max_rounds > 0 else 0.0

        # Outlier detection: if there IS an outlier, did the majority override it?
        outlier_detected = False
        if has_outlier and quorum_reached:
            # If consensus was reached despite an outlier, we consider it detected
            outlier_detected = True

        return ConsensusMetrics(
            quorum_reached=quorum_reached,
            quorum_size=quorum_size,
            consensus_confidence=round(consensus_confidence, 4),
            disagreement_score=round(disagreement, 4),
            rounds_used=rounds_used,
            max_rounds_allowed=max_rounds,
            early_stop_triggered=early_stop,
            quorum_efficiency=round(quorum_efficiency, 4),
            outlier_detected=outlier_detected,
            stability_horizon_met=quorum_reached,
            weighted_votes=weighted_votes or {},
        )

    # ──────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _percentile(sorted_values: List[float], pct: int) -> float:
        if not sorted_values:
            return 0.0
        idx = math.ceil(pct / 100 * len(sorted_values)) - 1
        return sorted_values[max(0, idx)]

    @staticmethod
    def _f1_for_class(
        results: list,
        cls: str,
        get_true,
        get_pred,
    ) -> float:
        """Compute F1 for a single class label."""
        tp = sum(1 for r in results if get_true(r) == cls and get_pred(r) == cls)
        fp = sum(1 for r in results if get_true(r) != cls and get_pred(r) == cls)
        fn = sum(1 for r in results if get_true(r) == cls and get_pred(r) != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def breakdown_by_difficulty(
        self,
        results: List[BenchmarkResult],
    ) -> dict:
        """Return accuracy broken down by difficulty level."""
        buckets: dict = defaultdict(list)
        for r in results:
            buckets[r.difficulty.value].append(r)
        return {
            diff: {
                "total": len(rs),
                "correct": sum(1 for r in rs if r.correct),
                "accuracy": sum(1 for r in rs if r.correct) / len(rs) if rs else 0.0,
            }
            for diff, rs in buckets.items()
        }

    def breakdown_by_category(
        self,
        results: List[BenchmarkResult],
    ) -> dict:
        """Return accuracy broken down by category."""
        buckets: dict = defaultdict(list)
        for r in results:
            buckets[r.category.value].append(r)
        return {
            cat: {
                "total": len(rs),
                "correct": sum(1 for r in rs if r.correct),
                "accuracy": sum(1 for r in rs if r.correct) / len(rs) if rs else 0.0,
            }
            for cat, rs in buckets.items()
        }

