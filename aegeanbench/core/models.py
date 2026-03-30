"""
AegeanBench core data models.

All benchmark cases, results, and metrics are typed here.
Designed to be framework-agnostic — works with aegean-consensus
out-of-the-box but can be adapted to any multi-agent system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import uuid


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class BenchmarkCategory(str, Enum):
    """Top-level benchmark category."""
    CONSENSUS     = "consensus"
    COLLABORATION = "collaboration"
    HYBRID        = "hybrid"
    RISK          = "risk"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class ExpectedDecision(str, Enum):
    APPROVE   = "approve"
    REJECT    = "reject"
    CHALLENGE = "challenge"
    REVIEW    = "review"


class ConsensusOutcome(str, Enum):
    CONVERGED  = "converged"
    DIVERGED   = "diverged"
    EARLY_STOP = "early_stop"
    TIMEOUT    = "timeout"
    ERROR      = "error"


# ─────────────────────────────────────────────
# Agent Snapshot
# ─────────────────────────────────────────────

class AgentSnapshot(BaseModel):
    """Per-agent state captured during a benchmark run."""
    agent_id:          str
    answer:            Optional[str] = None
    confidence:        float         = 1.0
    reasoning:         Optional[str] = None
    capability_weight: float         = 1.0
    rounds_active:     int           = 1
    was_outlier:       bool          = False
    latency_s:         float         = 0.0
    # Token tracking (per-agent, per-call)
    tokens_prompt:     int           = 0   # input tokens this agent consumed
    tokens_completion: int           = 0   # output tokens this agent generated

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion


# ─────────────────────────────────────────────
# Per-Category Metrics
# ─────────────────────────────────────────────

class ConsensusMetrics(BaseModel):
    """
    Metrics specific to consensus-mode cases.

    Novel metrics (absent from existing benchmarks):
      quorum_reached        - did >= alpha agents agree?
      consensus_confidence  - confidence of the winning answer
      disagreement_score    - fraction of agents NOT in the majority (0 = perfect)
      outlier_detected      - was a divergent agent correctly identified?
      early_stop_triggered  - was early termination used?
      quorum_efficiency     - 1 - (rounds_used / max_rounds)  [higher = better]
      stability_horizon_met - did stability counter reach beta?
    """
    quorum_reached:        bool              = False
    quorum_size:           int               = 0
    consensus_confidence:  float             = 0.0
    disagreement_score:    float             = 1.0
    rounds_used:           int               = 0
    max_rounds_allowed:    int               = 5
    early_stop_triggered:  bool              = False
    quorum_efficiency:     float             = 0.0
    outlier_detected:      bool              = False
    stability_horizon_met: bool              = False
    weighted_votes:        Dict[str, float]  = Field(default_factory=dict)
    # Token consumption per consensus run
    tokens_prompt_total:     int = 0   # sum of all agents' input tokens
    tokens_completion_total: int = 0   # sum of all agents' output tokens
    tokens_per_round:        List[int] = Field(default_factory=list)  # tokens by round
    token_efficiency:        float = 0.0  # correct_answer / tokens_total (0 if no LLM)


class CollaborationMetrics(BaseModel):
    """Metrics for collaboration-mode cases."""
    task_completed:          bool  = False
    subtasks_total:          int   = 0
    subtasks_completed:      int   = 0
    subtask_completion_rate: float = 0.0
    coordination_score:      float = 0.0
    redundancy_rate:         float = 0.0
    milestone_hit_rate:      float = 0.0


class RiskMetrics(BaseModel):
    """
    Metrics for the VAN (Verification Agent Network) risk pipeline.

    Novel metrics:
      pre_screen_triggered  - deterministic fast path fired (no LLM needed)
      challenge_appropriate - when challenged, was it the right call?
      validator_agreement   - fraction of validators aligned with final decision
    """
    expected_decision:        ExpectedDecision
    actual_decision:          Optional[str]        = None
    expected_risk_level:      str                  = "low"
    actual_risk_level:        Optional[str]        = None

    decision_correct:         bool                 = False
    risk_level_correct:       bool                 = False

    pre_screen_triggered:     bool                 = False
    challenge_appropriate:    Optional[bool]       = None
    validator_agreement:      float                = 0.0

    difficulty_level:         str                  = "simple"
    participating_validators: List[str]            = Field(default_factory=list)
    validator_results:        List[Dict[str, Any]] = Field(default_factory=list)
    # Token consumption (per validator pipeline run)
    tokens_prompt_total:     int   = 0
    tokens_completion_total: int   = 0
    tokens_saved_by_prescreen: int = 0  # tokens NOT spent because pre-screen fired


# ─────────────────────────────────────────────
# Benchmark Case
# ─────────────────────────────────────────────

class BenchmarkCase(BaseModel):
    """A single benchmark test case."""
    case_id:     str = Field(default_factory=lambda: f"case-{uuid.uuid4().hex[:8]}")
    name:        str
    description: str
    category:    BenchmarkCategory
    difficulty:  Difficulty         = Difficulty.MEDIUM
    tags:        List[str]          = Field(default_factory=list)

    # Consensus / Collaboration inputs
    task:                     Optional[str]         = None
    expected_answer:          Optional[str]         = None
    answer_variants:          List[str]             = Field(default_factory=list)
    num_agents:               int                   = 3
    max_rounds:               int                   = 5
    quorum_threshold:         float                 = 0.5
    agent_capability_weights: List[float]           = Field(default_factory=list)

    # Collaboration subtasks
    subtasks:   List[str] = Field(default_factory=list)
    milestones: List[str] = Field(default_factory=list)

    # Risk assessment input
    risk_payload:        Optional[Dict[str, Any]]  = None
    expected_decision:   Optional[ExpectedDecision] = None
    expected_risk_level: Optional[str]             = None

    # Adversarial flags
    has_outlier_agent: bool          = False
    outlier_agent_idx: Optional[int] = None
    inject_noise:      bool          = False

    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Benchmark Result
# ─────────────────────────────────────────────

class BenchmarkResult(BaseModel):
    """Result of running a single BenchmarkCase."""
    result_id:    str = Field(default_factory=lambda: f"res-{uuid.uuid4().hex[:8]}")
    case_id:      str
    case_name:    str
    category:     BenchmarkCategory
    difficulty:   Difficulty

    outcome:      ConsensusOutcome = ConsensusOutcome.ERROR
    correct:      bool             = False
    final_answer: Optional[str]    = None
    latency_s:    float            = 0.0

    # Token consumption — the key efficiency metric for LLM-based systems
    tokens_prompt:     int = 0   # total input tokens across all agents/rounds
    tokens_completion: int = 0   # total output tokens across all agents/rounds
    tokens_saved:      int = 0   # tokens saved by early-stop or pre-screen

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    agent_snapshots:       List[AgentSnapshot]          = Field(default_factory=list)
    consensus_metrics:     Optional[ConsensusMetrics]   = None
    collaboration_metrics: Optional[CollaborationMetrics] = None
    risk_metrics:          Optional[RiskMetrics]         = None

    raw_output: Dict[str, Any] = Field(default_factory=dict)
    error:      Optional[str]  = None
    timestamp:  datetime       = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# Suite
# ─────────────────────────────────────────────

class BenchmarkSuite(BaseModel):
    """A named collection of benchmark cases."""
    suite_id:    str = Field(default_factory=lambda: f"suite-{uuid.uuid4().hex[:8]}")
    name:        str
    description: str
    version:     str                  = "0.1.0"
    cases:       List[BenchmarkCase]  = Field(default_factory=list)
    metadata:    Dict[str, Any]       = Field(default_factory=dict)

    def filter_by_category(self, category: BenchmarkCategory) -> List[BenchmarkCase]:
        return [c for c in self.cases if c.category == category]

    def filter_by_difficulty(self, difficulty: Difficulty) -> List[BenchmarkCase]:
        return [c for c in self.cases if c.difficulty == difficulty]

    def filter_by_tag(self, tag: str) -> List[BenchmarkCase]:
        return [c for c in self.cases if tag in c.tags]


class BenchmarkSuiteResult(BaseModel):
    """Aggregated results from running a full BenchmarkSuite."""
    suite_id:   str
    suite_name: str
    run_id:     str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")

    total_cases: int = 0
    passed:      int = 0
    failed:      int = 0
    errored:     int = 0

    # Global
    accuracy:       float = 0.0
    mean_latency_s: float = 0.0
    p50_latency_s:  float = 0.0
    p95_latency_s:  float = 0.0

    # Token consumption — aggregate across all cases
    total_tokens_prompt:     int   = 0
    total_tokens_completion: int   = 0
    total_tokens:            int   = 0
    total_tokens_saved:      int   = 0   # saved by early-stop + pre-screen
    mean_tokens_per_case:    float = 0.0
    p50_tokens_per_case:     float = 0.0
    p95_tokens_per_case:     float = 0.0
    token_efficiency_rate:   float = 0.0  # total_tokens_saved / (total_tokens + saved)
    mean_tokens_per_correct: float = 0.0  # avg tokens for cases that passed

    # Consensus
    consensus_rate:         float = 0.0
    mean_consensus_conf:    float = 0.0
    mean_disagreement:      float = 0.0
    early_stop_rate:        float = 0.0
    mean_quorum_efficiency: float = 0.0
    outlier_detection_rate: float = 0.0

    # Collaboration
    mean_subtask_completion: float = 0.0
    mean_milestone_hit:      float = 0.0
    mean_coordination:       float = 0.0

    # Risk
    risk_accuracy:            float = 0.0
    risk_level_accuracy:      float = 0.0
    risk_f1_approve:          float = 0.0
    risk_f1_reject:           float = 0.0
    mean_validator_agreement: float = 0.0
    pre_screen_rate:          float = 0.0

    # Per-result list
    results:   List[BenchmarkResult] = Field(default_factory=list)
    timestamp: datetime              = Field(default_factory=datetime.utcnow)
    metadata:  Dict[str, Any]        = Field(default_factory=dict)
