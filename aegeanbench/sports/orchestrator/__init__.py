"""Pipeline orchestration: train, predict, bet, evaluate, persist."""

from aegeanbench.sports.orchestrator.evaluator import (
    BettingEvaluation,
    PerMatchEvaluation,
    RunnerEvaluation,
    evaluate_betting,
    evaluate_runner,
    evaluate_runners,
)
from aegeanbench.sports.orchestrator.persistence import (
    DEFAULT_RUNS_DIR,
    RunManifest,
    list_runs,
    load_run,
    save_evaluation,
    save_run,
)
from aegeanbench.sports.orchestrator.pipeline import (
    PipelineConfig,
    PipelineRunResult,
    WorldCupPipeline,
)

__all__ = [
    "WorldCupPipeline",
    "PipelineConfig",
    "PipelineRunResult",
    "evaluate_runner",
    "evaluate_runners",
    "evaluate_betting",
    "RunnerEvaluation",
    "PerMatchEvaluation",
    "BettingEvaluation",
    "save_run",
    "save_evaluation",
    "load_run",
    "list_runs",
    "RunManifest",
    "DEFAULT_RUNS_DIR",
]
