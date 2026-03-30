"""
AegeanBench test suite.

Tests:
  - Dataset loading (all suites load without errors)
  - MockRunner correctness (easy cases pass, outlier cases handled)
  - MetricsEngine aggregation (consensus, collaboration, risk)
  - CLI smoke test (run --category consensus --difficulty easy)
"""

from __future__ import annotations

import pytest
from aegeanbench.core.models import (
    BenchmarkCategory,
    ConsensusOutcome,
    Difficulty,
    ExpectedDecision,
)
from aegeanbench.datasets import (
    load_consensus_suite,
    load_collaboration_suite,
    load_risk_suite,
    load_full_suite,
)
from aegeanbench.metrics.engine import MetricsEngine
from aegeanbench.runners.mock_runner import MockRunner


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

class TestDatasetLoading:
    def test_consensus_suite_loads(self):
        suite = load_consensus_suite()
        assert suite.name
        assert len(suite.cases) > 0
        for case in suite.cases:
            assert case.case_id.startswith("C-")
            assert case.category == BenchmarkCategory.CONSENSUS or \
                   case.category == BenchmarkCategory.HYBRID

    def test_collaboration_suite_loads(self):
        suite = load_collaboration_suite()
        assert len(suite.cases) > 0
        for case in suite.cases:
            assert case.case_id.startswith("L-")
            assert case.category in (
                BenchmarkCategory.COLLABORATION,
                BenchmarkCategory.HYBRID,
            )

    def test_risk_suite_loads(self):
        suite = load_risk_suite()
        assert len(suite.cases) > 0
        for case in suite.cases:
            assert case.case_id.startswith("R-")
            assert case.category == BenchmarkCategory.RISK
            assert case.expected_decision is not None
            assert case.risk_payload is not None

    def test_full_suite_combines_all(self):
        full       = load_full_suite()
        consensus  = load_consensus_suite()
        collab     = load_collaboration_suite()
        risk       = load_risk_suite()
        expected   = len(consensus.cases) + len(collab.cases) + len(risk.cases)
        assert len(full.cases) == expected

    def test_suite_filter_by_category(self):
        full  = load_full_suite()
        risk  = full.filter_by_category(BenchmarkCategory.RISK)
        assert all(c.category == BenchmarkCategory.RISK for c in risk)

    def test_suite_filter_by_difficulty(self):
        full = load_full_suite()
        easy = full.filter_by_difficulty(Difficulty.EASY)
        assert all(c.difficulty == Difficulty.EASY for c in easy)

    def test_suite_filter_by_tag(self):
        full     = load_full_suite()
        outliers = full.filter_by_tag("outlier")
        assert len(outliers) > 0


# ─────────────────────────────────────────────
# MockRunner — consensus cases
# ─────────────────────────────────────────────

class TestMockRunnerConsensus:
    def setup_method(self):
        self.runner = MockRunner(seed=42)

    def test_easy_case_passes(self):
        suite = load_consensus_suite()
        easy  = [c for c in suite.cases if c.difficulty == Difficulty.EASY
                 and c.expected_answer is not None][0]
        result = self.runner.run_case(easy)
        assert result.correct is True
        assert result.outcome in (ConsensusOutcome.CONVERGED, ConsensusOutcome.EARLY_STOP)

    def test_outlier_case_still_converges(self):
        suite        = load_consensus_suite()
        outlier_case = next(c for c in suite.cases if c.has_outlier_agent
                            and c.expected_answer is not None)
        result = self.runner.run_case(outlier_case)
        # Mock runner: majority overrides outlier
        assert result.correct is True
        assert result.consensus_metrics is not None
        assert result.consensus_metrics.outlier_detected is True

    def test_no_consensus_expected_case(self):
        suite = load_consensus_suite()
        no_con = next((c for c in suite.cases
                       if c.metadata.get("expect_consensus") is False), None)
        if no_con is None:
            pytest.skip("No no-consensus case in suite")
        result = self.runner.run_case(no_con)
        assert result.outcome == ConsensusOutcome.DIVERGED

    def test_consensus_metrics_populated(self):
        suite  = load_consensus_suite()
        case   = suite.cases[0]
        result = self.runner.run_case(case)
        m = result.consensus_metrics
        assert m is not None
        assert 0.0 <= m.disagreement_score <= 1.0
        assert 0.0 <= m.quorum_efficiency <= 1.0
        assert m.rounds_used >= 1

    def test_agent_snapshots_populated(self):
        suite  = load_consensus_suite()
        case   = [c for c in suite.cases if c.num_agents >= 3][0]
        result = self.runner.run_case(case)
        assert len(result.agent_snapshots) == case.num_agents
        for snap in result.agent_snapshots:
            assert snap.agent_id
            assert snap.answer is not None

    def test_latency_recorded(self):
        suite  = load_consensus_suite()
        result = self.runner.run_case(suite.cases[0])
        assert result.latency_s >= 0.0


# ─────────────────────────────────────────────
# MockRunner — collaboration cases
# ─────────────────────────────────────────────

class TestMockRunnerCollaboration:
    def setup_method(self):
        self.runner = MockRunner(seed=42)

    def test_collaboration_case_completes(self):
        suite  = load_collaboration_suite()
        case   = suite.cases[0]
        result = self.runner.run_case(case)
        assert result.correct is True
        assert result.collaboration_metrics is not None
        assert result.collaboration_metrics.task_completed is True

    def test_subtask_completion_rate_is_one(self):
        suite  = load_collaboration_suite()
        result = self.runner.run_case(suite.cases[0])
        assert result.collaboration_metrics.subtask_completion_rate == 1.0

    def test_milestone_hit_rate_populated(self):
        suite = load_collaboration_suite()
        case  = next(c for c in suite.cases if c.milestones)
        result = self.runner.run_case(case)
        assert result.collaboration_metrics.milestone_hit_rate >= 0.0


# ─────────────────────────────────────────────
# MockRunner — risk cases
# ─────────────────────────────────────────────

class TestMockRunnerRisk:
    def setup_method(self):
        self.runner = MockRunner(seed=42)

    def test_risk_case_correct(self):
        suite  = load_risk_suite()
        result = self.runner.run_case(suite.cases[0])
        assert result.correct is True
        assert result.risk_metrics is not None
        assert result.risk_metrics.decision_correct is True

    def test_pre_screen_case_flagged(self):
        suite       = load_risk_suite()
        crit_cases  = [c for c in suite.cases
                       if c.metadata.get("expect_pre_screen", False)]
        assert len(crit_cases) > 0
        for case in crit_cases:
            result = self.runner.run_case(case)
            assert result.risk_metrics.pre_screen_triggered is True

    def test_all_risk_cases_run(self):
        suite   = load_risk_suite()
        results = self.runner.run_suite(suite.cases)
        assert len(results) == len(suite.cases)
        for r in results:
            assert r.outcome != ConsensusOutcome.ERROR, f"Case {r.case_id} errored: {r.error}"


# ─────────────────────────────────────────────
# MetricsEngine
# ─────────────────────────────────────────────

class TestMetricsEngine:
    def setup_method(self):
        self.runner  = MockRunner(seed=42)
        self.engine  = MetricsEngine()
        full         = load_full_suite()
        self.results = self.runner.run_suite(full.cases)

    def test_aggregate_returns_suite_result(self):
        sr = self.engine.aggregate("test-suite", "Test", self.results)
        assert sr.total_cases == len(self.results)
        assert sr.passed + sr.failed + sr.errored == sr.total_cases

    def test_accuracy_between_zero_and_one(self):
        sr = self.engine.aggregate("test-suite", "Test", self.results)
        assert 0.0 <= sr.accuracy <= 1.0

    def test_consensus_rate_populated(self):
        sr = self.engine.aggregate("test-suite", "Test", self.results)
        assert 0.0 <= sr.consensus_rate <= 1.0

    def test_risk_f1_populated(self):
        sr = self.engine.aggregate("test-suite", "Test", self.results)
        assert 0.0 <= sr.risk_f1_approve <= 1.0
        assert 0.0 <= sr.risk_f1_reject  <= 1.0

    def test_breakdown_by_difficulty(self):
        breakdown = self.engine.breakdown_by_difficulty(self.results)
        for diff, stats in breakdown.items():
            assert "total" in stats
            assert "accuracy" in stats
            assert 0.0 <= stats["accuracy"] <= 1.0

    def test_breakdown_by_category(self):
        breakdown = self.engine.breakdown_by_category(self.results)
        cats      = {r.category.value for r in self.results}
        assert set(breakdown.keys()) == cats

    def test_latency_percentiles(self):
        sr = self.engine.aggregate("test-suite", "Test", self.results)
        assert sr.p50_latency_s <= sr.p95_latency_s
        assert sr.mean_latency_s >= 0.0

    def test_compute_consensus_metrics_all_agree(self):
        m = self.engine.compute_consensus_metrics(
            agent_answers=["A", "A", "A"],
            final_answer="A",
            rounds_used=1,
            max_rounds=5,
            early_stop=True,
            has_outlier=False,
        )
        assert m.quorum_reached is True
        assert m.disagreement_score == 0.0
        assert m.early_stop_triggered is True
        assert m.quorum_efficiency > 0.0

    def test_compute_consensus_metrics_with_outlier(self):
        m = self.engine.compute_consensus_metrics(
            agent_answers=["A", "A", "A", "A", "B"],
            final_answer="A",
            rounds_used=2,
            max_rounds=5,
            early_stop=False,
            has_outlier=True,
        )
        assert m.quorum_reached is True
        assert m.outlier_detected is True
        assert 0.0 < m.disagreement_score < 1.0

    def test_compute_consensus_metrics_no_consensus(self):
        m = self.engine.compute_consensus_metrics(
            agent_answers=["A", "B", "C", "D"],
            final_answer=None,
            rounds_used=5,
            max_rounds=5,
            early_stop=False,
            has_outlier=False,
        )
        assert m.quorum_reached is False
        assert m.consensus_confidence == 0.0


# ─────────────────────────────────────────────
# Full suite smoke test
# ─────────────────────────────────────────────

class TestFullSuiteSmoke:
    def test_full_suite_runs_without_errors(self):
        runner  = MockRunner(seed=0)
        suite   = load_full_suite()
        results = runner.run_suite(suite.cases)
        errored = [r for r in results if r.outcome == ConsensusOutcome.ERROR]
        assert errored == [], f"Cases errored: {[r.case_id for r in errored]}"

    def test_full_suite_accuracy_above_threshold(self):
        """Mock runner with oracle answers should achieve near-perfect accuracy
        (only no-consensus edge cases are expected to be 'wrong')."""
        runner  = MockRunner(seed=0)
        suite   = load_full_suite()
        results = runner.run_suite(suite.cases)
        engine  = MetricsEngine()
        sr      = engine.aggregate(suite.suite_id, suite.name, results)
        # Allow for edge cases with no expected answer
        assert sr.accuracy >= 0.7, f"Accuracy too low: {sr.accuracy:.1%}"

