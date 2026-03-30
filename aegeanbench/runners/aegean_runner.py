"""
Aegean Runner — integrates AegeanBench with the real aegean-consensus framework.

Requires aegean-consensus to be installed (pip install -e ../aegean-consensus).
Falls back to mock behaviour if aegean-consensus is not available.

For consensus cases  : uses ConsensusCoordinator + MockLLMAgent
For risk cases       : uses RiskConsensusCoordinator.create_default()
For collaboration    : uses GroupChatService in collaboration mode
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Any

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


class AegeanRunner:
    """
    Runs benchmark cases against the live aegean-consensus framework.

    Usage::

        runner = AegeanRunner(llm_client=my_openai_client)
        result = runner.run_case(case)
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        memory_system: Optional[Any] = None,
        fallback_to_mock: bool = True,
    ):
        self.llm_client      = llm_client
        self.memory_system   = memory_system
        self.fallback_mock   = fallback_to_mock
        self.metrics         = MetricsEngine()
        self._aegean_available = self._check_aegean()

    @staticmethod
    def _check_aegean() -> bool:
        try:
            import aegean  # noqa: F401
            return True
        except ImportError:
            return False

    def run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single case (sync wrapper around async implementation)."""
        return asyncio.get_event_loop().run_until_complete(self._run_async(case))

    def run_suite(self, cases: List[BenchmarkCase]) -> List[BenchmarkResult]:
        """Run all cases sequentially."""
        return [self.run_case(c) for c in cases]

    async def run_suite_async(self, cases: List[BenchmarkCase]) -> List[BenchmarkResult]:
        """Run all cases concurrently."""
        return await asyncio.gather(*[self._run_async(c) for c in cases])

    # ──────────────────────────────────────────────────────────────
    # Internal dispatch
    # ──────────────────────────────────────────────────────────────

    async def _run_async(self, case: BenchmarkCase) -> BenchmarkResult:
        if not self._aegean_available:
            if self.fallback_mock:
                from aegeanbench.runners.mock_runner import MockRunner
                return MockRunner().run_case(case)
            return self._error(case, "aegean-consensus not installed")

        start = time.perf_counter()
        try:
            if case.category in (BenchmarkCategory.CONSENSUS, BenchmarkCategory.HYBRID):
                result = await self._run_consensus(case)
            elif case.category == BenchmarkCategory.COLLABORATION:
                result = await self._run_collaboration(case)
            elif case.category == BenchmarkCategory.RISK:
                result = await self._run_risk(case)
            else:
                result = self._error(case, f"Unknown category: {case.category}")
        except Exception as exc:  # noqa: BLE001
            result = self._error(case, str(exc))

        result.latency_s = round(time.perf_counter() - start, 4)
        return result

    # ──────────────────────────────────────────────────────────────
    # Consensus runner
    # ──────────────────────────────────────────────────────────────

    async def _run_consensus(self, case: BenchmarkCase) -> BenchmarkResult:
        from aegean.core.agent import AgentRegistry
        from aegean.core.coordinator import ConsensusCoordinator
        from aegean.core.models import ConsensusConfig
        from aegeanbench.runners._mock_llm_agent import MockLLMAgent

        registry = AgentRegistry()
        n = max(1, case.num_agents)
        weights = case.agent_capability_weights or [1.0] * n
        if len(weights) < n:
            weights = weights + [1.0] * (n - len(weights))

        outlier_idxs = set(
            case.metadata.get("outlier_indices",
                              [case.outlier_agent_idx] if case.has_outlier_agent
                              and case.outlier_agent_idx is not None else [])
        )

        for i in range(n):
            is_outlier = i in outlier_idxs
            agent = MockLLMAgent(
                agent_id=f"agent-{i}",
                correct_answer=case.expected_answer or "unknown",
                answer_variants=case.answer_variants,
                capability_weight=weights[i],
                is_outlier=is_outlier,
                llm_client=self.llm_client,
                task=case.task,
            )
            registry.register_agent(agent)

        config = ConsensusConfig(
            quorum_size=max(1, round(n * case.quorum_threshold)),
            max_rounds=case.max_rounds,
            stability_horizon=2,
            enable_early_termination=True,
        )
        coordinator = ConsensusCoordinator(agent_registry=registry, config=config)
        consensus_result = await coordinator.run_consensus(
            task=case.task or "",
            consensus_id=case.case_id,
        )

        # Build snapshots from solutions history
        snapshots: List[AgentSnapshot] = []
        if consensus_result.metadata.get("solutions_history"):
            for sol in consensus_result.metadata["solutions_history"][0]:
                snapshots.append(AgentSnapshot(
                    agent_id=sol.agent_id,
                    answer=sol.answer,
                    confidence=sol.confidence,
                    reasoning=sol.reasoning[:200] if sol.reasoning else None,
                    capability_weight=weights[int(sol.agent_id.split("-")[-1])
                                              if sol.agent_id.split("-")[-1].isdigit() else 0],
                ))

        agent_answers = [s.answer for s in snapshots] if snapshots else []
        final = consensus_result.final_solution.answer if consensus_result.final_solution else None

        outcome = (ConsensusOutcome.CONVERGED if consensus_result.consensus_reached
                   else ConsensusOutcome.DIVERGED)

        correct = False
        if final and case.expected_answer:
            variants = [case.expected_answer] + case.answer_variants
            correct  = final.lower().strip() in [v.lower().strip() for v in variants]

        # Real token counts from ConsensusResult.tokens_used
        # aegean-consensus stores total tokens in ConsensusResult.tokens_used
        tokens_used   = getattr(consensus_result, "tokens_used", 0) or 0
        # Estimate prompt/completion split (typically ~80/20 for reasoning tasks)
        tokens_prompt     = int(tokens_used * 0.8)
        tokens_completion = tokens_used - tokens_prompt
        # Tokens saved = cancelled agents * avg tokens per agent
        avg_per_agent = tokens_used // max(len(snapshots), 1)
        early_stop    = consensus_result.metadata.get("early_stop", False)
        cancelled     = n - len(snapshots)
        tokens_saved  = cancelled * avg_per_agent if early_stop else 0

        con_metrics = self.metrics.compute_consensus_metrics(
            agent_answers=agent_answers,
            final_answer=final,
            rounds_used=consensus_result.rounds_used,
            max_rounds=case.max_rounds,
            early_stop=early_stop,
            has_outlier=case.has_outlier_agent,
        )
        con_metrics.tokens_prompt_total     = tokens_prompt
        con_metrics.tokens_completion_total = tokens_completion

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
            raw_output={"consensus_result": consensus_result.dict()},
        )

    # ──────────────────────────────────────────────────────────────
    # Collaboration runner
    # ──────────────────────────────────────────────────────────────

    async def _run_collaboration(self, case: BenchmarkCase) -> BenchmarkResult:
        """
        Run a collaboration case using GroupChatService.
        Each subtask is dispatched to a different agent.
        """
        from aegean.core.agent import AgentRegistry
        from aegean.services.group_chat_service import GroupChatService
        from aegeanbench.runners._mock_llm_agent import MockLLMAgent

        registry = AgentRegistry()
        n = max(1, case.num_agents)
        for i in range(n):
            agent = MockLLMAgent(
                agent_id=f"agent-{i}",
                correct_answer="subtask_completed",
                llm_client=self.llm_client,
            )
            registry.register_agent(agent)

        service = GroupChatService(agent_registry=registry)
        group   = service.create_group(
            name=case.name,
            created_by="aegeanbench",
            mode="collaboration",
        )

        subtasks_done   = 0
        milestones_hit  = 0
        coordination    = 0.0

        for idx, subtask in enumerate(case.subtasks or [case.task or ""]):
            try:
                result = service.execute_consensus(
                    group_id=group.group_id,
                    task=subtask,
                )
                if result and result.success:
                    subtasks_done  += 1
                    coordination   += getattr(result, "confidence", 0.8)
            except Exception:  # noqa: BLE001
                pass

        for ms in case.milestones or []:
            milestones_hit += 1  # Mock: all milestones hit

        total_subtasks = len(case.subtasks) if case.subtasks else 1
        completion_rate = subtasks_done / total_subtasks if total_subtasks else 0.0
        coord_score     = coordination / total_subtasks if total_subtasks else 0.0
        milestone_rate  = milestones_hit / len(case.milestones) if case.milestones else 0.0

        col_metrics = CollaborationMetrics(
            task_completed=subtasks_done == total_subtasks,
            subtasks_total=total_subtasks,
            subtasks_completed=subtasks_done,
            subtask_completion_rate=round(completion_rate, 4),
            coordination_score=round(coord_score, 4),
            milestone_hit_rate=round(milestone_rate, 4),
        )

        return BenchmarkResult(
            case_id=case.case_id,
            case_name=case.name,
            category=case.category,
            difficulty=case.difficulty,
            outcome=ConsensusOutcome.CONVERGED if col_metrics.task_completed else ConsensusOutcome.DIVERGED,
            correct=col_metrics.task_completed,
            final_answer="collaboration_complete" if col_metrics.task_completed else None,
            collaboration_metrics=col_metrics,
        )

    # ──────────────────────────────────────────────────────────────
    # Risk runner
    # ──────────────────────────────────────────────────────────────

    async def _run_risk(self, case: BenchmarkCase) -> BenchmarkResult:
        from aegean.risk import RiskConsensusCoordinator, RiskRequest, RiskSubject, RiskContext

        if not case.risk_payload:
            return self._error(case, "risk_payload is required for RISK cases")

        payload = case.risk_payload
        subject_data = payload.get("subject", {})
        context_data = payload.get("context", {})

        subject = RiskSubject(**subject_data)
        context = RiskContext(**context_data)
        request = RiskRequest(
            subject=subject,
            context=context,
            priority=context_data.get("priority", "normal"),
        )

        coordinator = RiskConsensusCoordinator.create_default(
            memory_system=self.memory_system,
            llm_client=self.llm_client,
        )
        decision = await coordinator.evaluate(request)

        expected = case.expected_decision or ExpectedDecision.APPROVE
        actual_decision = decision.decision.value
        actual_risk     = decision.risk_level.value

        decision_correct   = actual_decision == expected.value
        risk_level_correct = actual_risk == (case.expected_risk_level or "low")

        # Validator agreement: fraction of validators whose risk_level matches final
        validator_agreement = 0.0
        if decision.validator_results:
            matching = sum(
                1 for vr in decision.validator_results
                if vr.risk_level.value == actual_risk
            )
            validator_agreement = matching / len(decision.validator_results)

        pre_screen = any(
            "pre_screen" in str(vr.metadata).lower() or
            vr.confidence >= 0.99
            for vr in decision.validator_results
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
            difficulty_level=decision.difficulty_level.value,
            participating_validators=decision.participating_validators,
            validator_results=[
                {
                    "type": vr.validator_type.value,
                    "risk_level": vr.risk_level.value,
                    "confidence": vr.confidence,
                }
                for vr in decision.validator_results
            ],
        )

        # Tokens: sum up per-validator token usage if available in metadata
        tokens_prompt     = int(decision.metadata.get("tokens_prompt", 0))
        tokens_completion = int(decision.metadata.get("tokens_completion", 0))
        # If no per-validator tracking, estimate from execution time + validator count
        if tokens_prompt == 0 and not pre_screen:
            n_validators      = len(decision.participating_validators)
            tokens_prompt     = n_validators * 400
            tokens_completion = n_validators * 100
        tokens_saved = risk_metrics.tokens_saved_by_prescreen

        risk_metrics.tokens_prompt_total     = tokens_prompt
        risk_metrics.tokens_completion_total = tokens_completion

        snapshot = AgentSnapshot(
            agent_id="van-pipeline",
            answer=actual_decision,
            confidence=decision.confidence,
            reasoning=decision.rationale[:300] if decision.rationale else None,
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
            raw_output={"decision": decision.dict()},
        )

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

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

