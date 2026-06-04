"""
SportsDataGateway: composes multiple adapters into a single facade.

Runners and predictors talk to the gateway, not individual adapters.
The gateway decides which adapter to call for each field and stitches
the results into a complete Match object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from aegeanbench.sports.cache import FileCache, get_default_cache
from aegeanbench.sports.models import Match, Team
from aegeanbench.sports.sources import (
    FBrefAdapter,
    FetchPolicy,
    FootballDataAdapter,
    KaggleHistoryLoader,
    SoccersAPIAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:
    """
    Everything a predictor needs to predict one match.

    Built by SportsDataGateway.build_context(). Designed so the same
    object can be passed to Dixon-Coles, Elo, LLM runners, and aegean.
    """
    match: Match
    home_history: List[Match]
    away_history: List[Match]
    h2h: List[Match]
    home_xg_profile: Dict[str, float]
    away_xg_profile: Dict[str, float]

    def summary(self) -> str:
        """Compact one-line summary for logs."""
        return (
            f"{self.match.home_team.fifa_code} vs {self.match.away_team.fifa_code} "
            f"@ {self.match.kickoff_at.date()} | "
            f"xG: {self.home_xg_profile.get('xg_for', '?'):.2f}-"
            f"{self.away_xg_profile.get('xg_for', '?'):.2f} | "
            f"h2h: {len(self.h2h)} matches"
        )


class SportsDataGateway:
    """
    Facade over all sports data adapters.

    Construction:
        gw = SportsDataGateway()                # default: mock everywhere
        gw = SportsDataGateway(mock=False)      # try real APIs (need keys)
    """

    def __init__(
        self,
        mock: bool = True,
        cache: Optional[FileCache] = None,
    ):
        self.mock = mock
        self.cache = cache or get_default_cache()
        # Initialize adapters. Each falls back to mock when no key configured.
        self.football_data = FootballDataAdapter(mock_by_default=mock)
        self.soccersapi = SoccersAPIAdapter(mock_by_default=mock)
        self.fbref = FBrefAdapter(mock_by_default=mock)
        self.history_loader = KaggleHistoryLoader(mock_by_default=mock)

    # -------- top-level API --------

    def list_teams(self, competition: str = "FIFA World Cup 2026") -> List[Team]:
        """All teams in the tournament."""
        return self.football_data.fetch_teams(competition)

    def list_fixtures(
        self, competition: str = "FIFA World Cup 2026"
    ) -> List[Match]:
        """All matches in the tournament."""
        return self.football_data.fetch_fixtures(competition)

    def build_context(self, match: Match, history_size: int = 10) -> MatchContext:
        """
        Build a MatchContext by pulling odds, lineups, xG, and history.

        This is the heavy call - cached aggressively because every runner
        for every match will need it.
        """
        cache_key = ("gateway", "build_context", match.match_id, history_size)
        cached = self.cache.get(*cache_key)
        if cached is not None:
            # Cache hit: not yet reconstructing dataclasses from dict — just
            # rebuild from scratch for now. v2: add deserialization.
            logger.debug("gateway cache hit for %s but rebuilding (deser TBD)", match.match_id)

        # Enrich match with odds + lineups
        if not match.odds:
            match.odds = self.soccersapi.fetch_odds(match.match_id)
        if not match.home_lineup:
            match.home_lineup = self.soccersapi.fetch_lineup(match.match_id, match.home_team.fifa_code)
        if not match.away_lineup:
            match.away_lineup = self.soccersapi.fetch_lineup(match.match_id, match.away_team.fifa_code)
        if not match.h2h_last5:
            h2h = self.soccersapi.fetch_h2h(match.home_team.fifa_code, match.away_team.fifa_code)
            match.h2h_last5 = [m.to_dict() for m in h2h]
        else:
            h2h = []

        # Pull team histories
        home_history = self.football_data.fetch_team_history(
            match.home_team.fifa_code, last_n=history_size
        )
        away_history = self.football_data.fetch_team_history(
            match.away_team.fifa_code, last_n=history_size
        )

        # xG profiles
        home_xg = self.fbref.fetch_xg_profile(match.home_team.fifa_code)
        away_xg = self.fbref.fetch_xg_profile(match.away_team.fifa_code)

        ctx = MatchContext(
            match=match,
            home_history=home_history,
            away_history=away_history,
            h2h=h2h,
            home_xg_profile=home_xg,
            away_xg_profile=away_xg,
        )

        return ctx

    def load_training_history(self, num_matches: int = 500) -> List[Match]:
        """For Dixon-Coles / Elo training: pull many historical matches."""
        return self.history_loader.load(num_matches=num_matches)
