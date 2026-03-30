"""""
AegeanBench — The first open benchmark for Multi-Agent Consensus systems.

Evaluates:
  1. Consensus Quality      — does the agent committee converge correctly?
  2. Collaboration Quality  — do agents divide and solve complex tasks?
  3. Risk Assessment        — does the VAN pipeline make correct decisions?

Metrics:
  - consensus_rate         : fraction of cases where consensus is reached
  - consensus_confidence   : mean confidence of the final consensus answer
  - disagreement_score     : normalized disagreement across agents
  - quorum_efficiency      : rounds used vs. max_rounds allowed
  - early_stop_rate        : fraction of cases terminated early
  - outlier_detection_rate : fraction of outlier agents correctly identified
  - collaboration_accuracy : task completion accuracy in collaboration mode
  - risk_accuracy          : VAN pipeline decision correctness
  - risk_f1                : F1 score on risk-level classification
  - latency_p50/p95        : median and 95th-percentile execution latency
"""

__version__ = "0.1.0"
__author__  = "AegeanBench Contributors"
""
