"""Betting layer: Kelly sizing, value-bet identification, portfolio construction."""

from aegeanbench.sports.betting.kelly import (
    KellyResult,
    kelly_fraction,
    stake_amount,
)
from aegeanbench.sports.betting.portfolio import (
    BettingPortfolio,
    PortfolioConfig,
    SizedBet,
    build_portfolio,
)
from aegeanbench.sports.betting.value import (
    OUTCOME_LABELS,
    ValueBetCandidate,
    best_value_bet,
    find_value_bets,
)

__all__ = [
    "KellyResult",
    "kelly_fraction",
    "stake_amount",
    "ValueBetCandidate",
    "find_value_bets",
    "best_value_bet",
    "OUTCOME_LABELS",
    "PortfolioConfig",
    "SizedBet",
    "BettingPortfolio",
    "build_portfolio",
]
