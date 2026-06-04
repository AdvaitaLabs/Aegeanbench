"""
Sports prediction benchmark module.

Designed for World Cup 2026 sprint but generalizable to any tournament.
Mock-first architecture: works fully offline today, swaps to real APIs
when keys land.
"""

from aegeanbench.sports.gateway import MatchContext, SportsDataGateway
from aegeanbench.sports.models import (
    CompetitionStage,
    Match,
    MatchOutcome,
    MatchResult,
    Odds,
    Player,
    Prediction,
    Team,
)

__all__ = [
    "CompetitionStage",
    "Match",
    "MatchContext",
    "MatchOutcome",
    "MatchResult",
    "Odds",
    "Player",
    "Prediction",
    "SportsDataGateway",
    "Team",
]
