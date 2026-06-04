"""
WorldCupPipeline: top-level orchestrator that wires Days 1-4 together.

Responsibilities:
    1. Pull fixtures from the SportsDataGateway.
    2. Train any predictors that need training (Elo, Dixon-Coles) on
       historical data.
    3. For each fixture, build a MatchContext and run every configured
       predictor in parallel, collecting Predictions per runner.
    4. Identify value bets across all matches for a chosen runner.
    5. Build a betting portfolio.
    6. (Optional) Persist the whole run for later evaluation / reporting.
    7. (Optional, called separately later) Evaluate predictions and the
       portfolio against realised results.

The pipeline is deliberately stateless across runs - state lives in:
    - The trained predictor parameters (in the predictor instances)
    - The persisted run directory on disk

This makes the pipeline safe to invoke from a cron job, a CLI command,
or an HTTP endpoint without worrying about thread safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from aegeanbench.sports.betting import (
    BettingPortfolio,
    PortfolioConfig,
    build_portfolio,
    find_value_bets,
)
from aegeanbench.sports.gateway import MatchContext, SportsDataGateway
from aegeanbench.sports.models import Match, Prediction
from aegeanbench.sports.orchestrator.evaluator import (
    BettingEvaluation,
    RunnerEvaluation,
    evaluate_betting,
    evaluate_runners,
)
from aegeanbench.sports.orchestrator.persistence import (
    save_evaluation,
    save_run,
)
from aegeanbench.sports.predictors import (
    DixonColesPredictor,
    EloPredictor,
    Predictor,
)

logger = logging.getLogger(__name__)


# ----------------------------- config -----------------------------


@dataclass
class PipelineConfig:
    """Configuration for a WorldCupPipeline run."""
    competition: str = "FIFA World Cup 2026"
    # Predictor IDs to train (must implement fit()). Others are stateless.
    train_history_size: int = 500
    # Which runner's predictions drive value-bet construction?
    betting_runner_id: Optional[str] = "aegean"
    # Portfolio risk configuration; None disables betting.
    portfolio_config: Optional[PortfolioConfig] = None
    # Whether to persist the run to disk.
    persist: bool = True
    persist_label: Optional[str] = None
    runs_dir: Optional[Path] = None


# ----------------------------- result types -----------------------------


@dataclass
class PipelineRunResult:
    """Output of one pipeline.run() invocation."""
    run_dir: Optional[Path]
    matches: List[Match]
    predictions_by_runner: Dict[str, List[Prediction]] = field(default_factory=dict)
    portfolio: Optional[BettingPortfolio] = None
    config: Optional[PipelineConfig] = None

    @property
    def runner_ids(self) -> List[str]:
        return list(self.predictions_by_runner.keys())

    def predictions_for(self, runner_id: str) -> List[Prediction]:
        return list(self.predictions_by_runner.get(runner_id, []))


# ----------------------------- pipeline -----------------------------


class WorldCupPipeline:
    """
    Compose a gateway and a list of predictors into a runnable pipeline.

    Example:
        gw = SportsDataGateway(mock=True)
        predictors = [EloPredictor(), DixonColesPredictor(),
                      LLMPredictor(MockLLMClient('gpt-5'), 'gpt-5', 'GPT-5'),
                      AegeanPredictor(mock=True)]
        pipeline = WorldCupPipeline(gw, predictors)
        result = pipeline.run()
        print(result.predictions_by_runner)
    """

    def __init__(
        self,
        gateway: SportsDataGateway,
        predictors: Sequence[Predictor],
        config: Optional[PipelineConfig] = None,
    ):
        if not predictors:
            raise ValueError("WorldCupPipeline requires at least one predictor")
        self.gateway = gateway
        self.predictors = list(predictors)
        self.config = config or PipelineConfig()
        self._trained = False

    # ---------- public API ----------

    def train(self, history: Optional[Iterable[Match]] = None) -> None:
        """
        Train all predictors that need historical data.

        Stateless predictors (LLMs, Aegean black-box) have a no-op fit()
        so calling train() unconditionally is safe.
        """
        hist = list(history) if history is not None else self.gateway.load_training_history(
            num_matches=self.config.train_history_size
        )
        for predictor in self.predictors:
            try:
                predictor.fit(hist)
            except Exception as e:
                logger.warning(
                    "predictor %s fit() failed: %s",
                    getattr(predictor, "runner_id", predictor.__class__.__name__),
                    e,
                )
        self._trained = True

    def run(
        self,
        matches: Optional[Sequence[Match]] = None,
        train: bool = True,
    ) -> PipelineRunResult:
        """
        Execute the full pipeline.

        Args:
            matches: explicit list of fixtures. Defaults to gateway.list_fixtures().
            train: when True (default) call train() first if not yet trained.

        Returns:
            PipelineRunResult with predictions, portfolio, and run_dir if persisted.
        """
        if train and not self._trained:
            self.train()

        fixtures = list(matches) if matches is not None else self.gateway.list_fixtures(
            self.config.competition
        )
        logger.info("Pipeline running over %d fixtures", len(fixtures))

        # Build MatchContext once per match; all predictors share it.
        contexts: List[MatchContext] = [self.gateway.build_context(m) for m in fixtures]

        predictions_by_runner: Dict[str, List[Prediction]] = {
            getattr(p, "runner_id", p.__class__.__name__): [] for p in self.predictors
        }
        for ctx in contexts:
            for predictor in self.predictors:
                runner_id = getattr(predictor, "runner_id", predictor.__class__.__name__)
                try:
                    pred = predictor.predict(ctx)
                    predictions_by_runner[runner_id].append(pred)
                except Exception as e:
                    logger.warning(
                        "predictor %s failed on match %s: %s",
                        runner_id, ctx.match.match_id, e,
                    )
                    # Skip this prediction but keep going for other matches

        portfolio = self._build_portfolio(predictions_by_runner, contexts)

        run_dir: Optional[Path] = None
        if self.config.persist:
            run_dir = save_run(
                predictions_by_runner=predictions_by_runner,
                portfolio=portfolio,
                competition=self.config.competition,
                label=self.config.persist_label,
                config={
                    "betting_runner_id": self.config.betting_runner_id,
                    "train_history_size": self.config.train_history_size,
                    "n_predictors": len(self.predictors),
                    "n_fixtures": len(fixtures),
                },
                runs_dir=self.config.runs_dir,
            )

        return PipelineRunResult(
            run_dir=run_dir,
            matches=fixtures,
            predictions_by_runner=predictions_by_runner,
            portfolio=portfolio,
            config=self.config,
        )

    def evaluate(
        self,
        run_result: PipelineRunResult,
        matches_with_ground_truth: Optional[Sequence[Match]] = None,
    ) -> Dict[str, object]:
        """
        Score a previous run against ground truth.

        Args:
            run_result: output of self.run()
            matches_with_ground_truth: if provided, used directly; otherwise
                we use the matches in run_result.matches (their result field
                must have been populated since the run).

        Returns:
            Dict with keys 'runner_evaluations' and 'betting_evaluation'.
            Also writes evaluation.json into the run_dir if it was persisted.
        """
        matches = list(
            matches_with_ground_truth
            if matches_with_ground_truth is not None
            else run_result.matches
        )

        runner_evals: Dict[str, RunnerEvaluation] = evaluate_runners(
            run_result.predictions_by_runner, matches
        )

        bet_eval: Optional[BettingEvaluation] = None
        if run_result.portfolio is not None:
            bet_eval = evaluate_betting(run_result.portfolio.bets, matches)

        if run_result.run_dir is not None:
            save_evaluation(run_result.run_dir, runner_evals, bet_eval)

        return {
            "runner_evaluations": runner_evals,
            "betting_evaluation": bet_eval,
        }

    # ---------- internals ----------

    def _build_portfolio(
        self,
        predictions_by_runner: Dict[str, List[Prediction]],
        contexts: List[MatchContext],
    ) -> Optional[BettingPortfolio]:
        if self.config.portfolio_config is None:
            return None
        runner_id = self.config.betting_runner_id
        if not runner_id or runner_id not in predictions_by_runner:
            logger.info("No betting runner configured or runner not found; skipping portfolio")
            return None

        preds = {p.match_id: p for p in predictions_by_runner[runner_id]}
        candidates = []
        for ctx in contexts:
            pred = preds.get(ctx.match.match_id)
            if pred is None:
                continue
            candidates.extend(
                find_value_bets(
                    pred, ctx.match,
                    min_edge=self.config.portfolio_config.min_edge,
                    kelly_fraction_pct=self.config.portfolio_config.kelly_fraction,
                )
            )
        return build_portfolio(candidates, self.config.portfolio_config)
