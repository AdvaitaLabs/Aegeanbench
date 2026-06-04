"""
Prediction evaluator: scores predictions against realised match results.

Implements the metrics the benchmark dashboards rely on:

  - brier_score    average over matches of sum((p_i - actual_i)^2), one for
                   each of the 3 outcomes. Lower is better, 0.0 is perfect,
                   0.25 is a uniform (1/3, 1/3, 1/3) guess on a 3-way market.
  - log_loss       -sum(actual_i * log(p_i)) averaged over matches; penalises
                   confidently-wrong predictions much more than Brier.
  - hit_rate       fraction of matches where the argmax outcome matched.
  - roi            realised return on staked capital, given a portfolio of
                   sized bets and the actual results.

All metrics tolerate predictions for matches that haven't completed yet
(result=None) by simply skipping them and reporting how many were skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from aegeanbench.sports.betting import SizedBet
from aegeanbench.sports.models import Match, MatchOutcome, Prediction


# ----------------------------- helpers -----------------------------


def _outcome_one_hot(outcome: MatchOutcome) -> Dict[MatchOutcome, float]:
    """Convert a realised outcome to a one-hot probability distribution."""
    return {o: 1.0 if o == outcome else 0.0 for o in MatchOutcome}


def _pred_dist(p: Prediction) -> Dict[MatchOutcome, float]:
    return {
        MatchOutcome.HOME_WIN: p.p_home_win,
        MatchOutcome.DRAW: p.p_draw,
        MatchOutcome.AWAY_WIN: p.p_away_win,
    }


def _brier(p: Prediction, actual: MatchOutcome) -> float:
    """Brier score for one prediction vs one realised outcome."""
    actual_oh = _outcome_one_hot(actual)
    pred = _pred_dist(p)
    return sum((pred[o] - actual_oh[o]) ** 2 for o in MatchOutcome)


def _log_loss(p: Prediction, actual: MatchOutcome, eps: float = 1e-12) -> float:
    """Cross-entropy loss for one prediction vs one realised outcome."""
    pred = _pred_dist(p)
    pred_actual = max(eps, min(1.0 - eps, pred[actual]))
    return -math.log(pred_actual)


# ----------------------------- result types -----------------------------


@dataclass
class PerMatchEvaluation:
    """Evaluation of one prediction against one realised result."""
    match_id: str
    runner_id: str
    brier_score: float
    log_loss: float
    correct: bool                # argmax of prediction matched realised outcome
    predicted_outcome: MatchOutcome
    actual_outcome: MatchOutcome
    p_predicted: float           # probability the model assigned to its argmax

    def to_dict(self) -> Dict:
        return {
            "match_id": self.match_id,
            "runner_id": self.runner_id,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "correct": self.correct,
            "predicted_outcome": self.predicted_outcome.value,
            "actual_outcome": self.actual_outcome.value,
            "p_predicted": self.p_predicted,
        }


@dataclass
class RunnerEvaluation:
    """Aggregate metrics for one runner across many matches."""
    runner_id: str
    n_evaluated: int
    n_skipped_no_result: int = 0
    mean_brier: float = 0.0
    mean_log_loss: float = 0.0
    hit_rate: float = 0.0
    mean_confidence: float = 0.0
    per_match: List[PerMatchEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "runner_id": self.runner_id,
            "n_evaluated": self.n_evaluated,
            "n_skipped_no_result": self.n_skipped_no_result,
            "mean_brier": self.mean_brier,
            "mean_log_loss": self.mean_log_loss,
            "hit_rate": self.hit_rate,
            "mean_confidence": self.mean_confidence,
            "per_match": [e.to_dict() for e in self.per_match],
        }


@dataclass
class BettingEvaluation:
    """Realised returns for a portfolio of bets."""
    n_bets: int
    n_won: int
    n_lost: int
    n_skipped_no_result: int
    total_stake: float
    total_returned: float        # amount returned to bettor (stake + profit on wins)
    net_profit: float            # total_returned - total_stake
    roi: float                   # net_profit / total_stake

    def to_dict(self) -> Dict:
        return {
            "n_bets": self.n_bets,
            "n_won": self.n_won,
            "n_lost": self.n_lost,
            "n_skipped_no_result": self.n_skipped_no_result,
            "total_stake": self.total_stake,
            "total_returned": self.total_returned,
            "net_profit": self.net_profit,
            "roi": self.roi,
        }


# ----------------------------- evaluator -----------------------------


def evaluate_runner(
    predictions: Iterable[Prediction],
    matches: Iterable[Match],
) -> RunnerEvaluation:
    """
    Score one runner's predictions against the realised results in matches.

    Matches without result populated are skipped (not yet played).
    Returns the aggregated RunnerEvaluation; predictions and matches need
    not be in the same order, we join on match_id.
    """
    match_index: Dict[str, Match] = {m.match_id: m for m in matches}
    per_match: List[PerMatchEvaluation] = []
    skipped = 0
    runner_id: Optional[str] = None
    confidences: List[float] = []

    for p in predictions:
        runner_id = runner_id or p.runner_id
        m = match_index.get(p.match_id)
        if m is None or m.result is None or m.result.outcome is None:
            skipped += 1
            continue
        actual = m.result.outcome
        brier = _brier(p, actual)
        ll = _log_loss(p, actual)
        predicted = p.predicted_outcome
        per_match.append(
            PerMatchEvaluation(
                match_id=p.match_id,
                runner_id=p.runner_id,
                brier_score=brier,
                log_loss=ll,
                correct=(predicted == actual),
                predicted_outcome=predicted,
                actual_outcome=actual,
                p_predicted=p.predicted_probability,
            )
        )
        if p.confidence is not None:
            confidences.append(p.confidence)

    if not per_match:
        return RunnerEvaluation(
            runner_id=runner_id or "unknown",
            n_evaluated=0,
            n_skipped_no_result=skipped,
        )

    n = len(per_match)
    return RunnerEvaluation(
        runner_id=runner_id or "unknown",
        n_evaluated=n,
        n_skipped_no_result=skipped,
        mean_brier=sum(e.brier_score for e in per_match) / n,
        mean_log_loss=sum(e.log_loss for e in per_match) / n,
        hit_rate=sum(1 for e in per_match if e.correct) / n,
        mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
        per_match=per_match,
    )


def evaluate_runners(
    predictions_by_runner: Dict[str, List[Prediction]],
    matches: Iterable[Match],
) -> Dict[str, RunnerEvaluation]:
    """Convenience: evaluate many runners and return a dict by runner_id."""
    matches_list = list(matches)
    return {
        runner_id: evaluate_runner(preds, matches_list)
        for runner_id, preds in predictions_by_runner.items()
    }


# ----------------------------- betting evaluation -----------------------------


def evaluate_betting(
    bets: Iterable[SizedBet],
    matches: Iterable[Match],
) -> BettingEvaluation:
    """
    Compute realised ROI for a portfolio against actual match results.

    Settlement rule:
        win  -> stake * decimal_odds returned (gross)
        loss -> 0 returned
        no result -> bet skipped, stake refunded conceptually

    Returns BettingEvaluation with absolute numbers and ROI.
    """
    match_index: Dict[str, Match] = {m.match_id: m for m in matches}
    n_won = 0
    n_lost = 0
    skipped = 0
    total_stake = 0.0
    total_returned = 0.0
    bets_list = list(bets)

    for bet in bets_list:
        m = match_index.get(bet.candidate.match_id)
        if m is None or m.result is None or m.result.outcome is None:
            skipped += 1
            continue
        total_stake += bet.stake_amount
        if bet.candidate.outcome == m.result.outcome:
            n_won += 1
            total_returned += bet.stake_amount * bet.candidate.decimal_odds
        else:
            n_lost += 1
            # 0 returned

    net = total_returned - total_stake
    roi = (net / total_stake) if total_stake > 0 else 0.0
    return BettingEvaluation(
        n_bets=n_won + n_lost,
        n_won=n_won,
        n_lost=n_lost,
        n_skipped_no_result=skipped,
        total_stake=total_stake,
        total_returned=total_returned,
        net_profit=net,
        roi=roi,
    )
