"""
Value-bet identification for football matches.

A value bet exists when the model's probability is meaningfully higher than
the market's implied probability:

    edge = p_model * decimal_odds - 1
    edge > 0  ->  positive expected value
    edge > min_edge  ->  worth acting on (covers parsing noise + costs)

This module wraps a (Prediction, Match) pair into zero or more
ValueBetCandidate objects, one per outcome (home_win / draw / away_win)
that clears the edge threshold.

Bookmaker selection: by default we use the consensus odds across all
bookmakers (margin-removed). Callers can opt into "best-line" mode that
picks the single bookmaker with the highest odds on each outcome to
maximize expected return at the cost of operational complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from aegeanbench.sports.betting.kelly import KellyResult, kelly_fraction
from aegeanbench.sports.models import Match, MatchOutcome, Odds, Prediction


OUTCOME_LABELS: Dict[MatchOutcome, str] = {
    MatchOutcome.HOME_WIN: "home_win",
    MatchOutcome.DRAW: "draw",
    MatchOutcome.AWAY_WIN: "away_win",
}


@dataclass
class ValueBetCandidate:
    """A single candidate value bet on one outcome of one match."""
    match_id: str
    runner_id: str
    outcome: MatchOutcome
    p_model: float                # Model's probability for this outcome
    p_market: float               # Market consensus implied probability
    decimal_odds: float           # The odds we'd actually take
    edge: float                   # p_model * decimal_odds - 1
    bookmaker: str = "consensus"  # Source of decimal_odds
    kelly: Optional[KellyResult] = None   # Filled by attach_kelly()
    confidence: Optional[float] = None    # Optional model confidence
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def outcome_label(self) -> str:
        return OUTCOME_LABELS[self.outcome]

    def to_dict(self) -> Dict:
        return {
            "match_id": self.match_id,
            "runner_id": self.runner_id,
            "outcome": self.outcome.value,
            "p_model": self.p_model,
            "p_market": self.p_market,
            "decimal_odds": self.decimal_odds,
            "edge": self.edge,
            "bookmaker": self.bookmaker,
            "kelly": (
                {
                    "full_kelly": self.kelly.full_kelly,
                    "fractional_kelly": self.kelly.fractional_kelly,
                    "growth_rate": self.kelly.growth_rate,
                } if self.kelly else None
            ),
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


def _pick_decimal_odds(
    match: Match,
    outcome: MatchOutcome,
    mode: str,
) -> Tuple[float, float, str]:
    """
    Choose decimal odds and the corresponding implied market probability
    for the given outcome.

    Args:
        match: Match with .odds populated.
        outcome: HOME_WIN, DRAW, or AWAY_WIN.
        mode: "consensus" (default, average across books) or "best"
              (highest single-bookmaker odds; maximises potential payout).

    Returns:
        (decimal_odds, market_implied_probability, bookmaker_label)
    """
    if not match.odds:
        return (0.0, 0.0, "none")

    outcome_attr = {
        MatchOutcome.HOME_WIN: "home_win",
        MatchOutcome.DRAW: "draw",
        MatchOutcome.AWAY_WIN: "away_win",
    }[outcome]

    if mode == "best":
        # Pick the single bookmaker with the highest odds for this outcome
        best = max(match.odds, key=lambda o: getattr(o, outcome_attr))
        decimal_odds = float(getattr(best, outcome_attr))
        p_market = best.fair_probs[outcome_attr]
        return (decimal_odds, p_market, best.bookmaker)

    # Default: consensus fair probability -> implied decimal odds
    consensus = match.consensus_odds
    if not consensus:
        return (0.0, 0.0, "none")
    p_market = consensus[outcome_attr]
    if p_market <= 0:
        return (0.0, 0.0, "consensus")
    # Implied "fair" decimal odds = 1 / fair_prob. This is the no-margin odds.
    # We use the AVERAGE bookmaker decimal odds as the actually-bettable price.
    avg_decimal = sum(getattr(o, outcome_attr) for o in match.odds) / len(match.odds)
    return (float(avg_decimal), float(p_market), "consensus")


def find_value_bets(
    prediction: Prediction,
    match: Match,
    min_edge: float = 0.05,
    odds_mode: str = "consensus",
    kelly_fraction_pct: float = 0.25,
    kelly_cap: float = 0.20,
) -> List[ValueBetCandidate]:
    """
    Identify value bets across all three outcomes of a single match.

    Args:
        prediction: Model's predicted probability distribution.
        match: Match with odds populated.
        min_edge: Minimum edge to qualify as a value bet (default 5%).
            Lower thresholds capture more bets but include more noise.
        odds_mode: "consensus" (avg across books) or "best" (highest line).
        kelly_fraction_pct: Fractional Kelly multiplier passed through.
        kelly_cap: Hard cap on stake fraction.

    Returns:
        List of ValueBetCandidate sorted by edge descending. Empty if
        no outcome clears min_edge.
    """
    if not match.odds:
        return []

    candidates: List[ValueBetCandidate] = []
    probs = {
        MatchOutcome.HOME_WIN: prediction.p_home_win,
        MatchOutcome.DRAW: prediction.p_draw,
        MatchOutcome.AWAY_WIN: prediction.p_away_win,
    }

    for outcome, p_model in probs.items():
        decimal_odds, p_market, bookmaker = _pick_decimal_odds(match, outcome, odds_mode)
        if decimal_odds <= 1.0:
            continue
        edge = p_model * decimal_odds - 1.0
        if edge < min_edge:
            continue
        kr = kelly_fraction(
            p_win=p_model,
            decimal_odds=decimal_odds,
            fraction=kelly_fraction_pct,
            cap=kelly_cap,
        )
        candidates.append(
            ValueBetCandidate(
                match_id=match.match_id,
                runner_id=prediction.runner_id,
                outcome=outcome,
                p_model=p_model,
                p_market=p_market,
                decimal_odds=decimal_odds,
                edge=edge,
                bookmaker=bookmaker,
                kelly=kr,
                confidence=prediction.confidence,
            )
        )

    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates


def best_value_bet(
    prediction: Prediction,
    match: Match,
    min_edge: float = 0.05,
    **kwargs,
) -> Optional[ValueBetCandidate]:
    """Return only the single best value bet for the match, or None."""
    candidates = find_value_bets(prediction, match, min_edge=min_edge, **kwargs)
    return candidates[0] if candidates else None
