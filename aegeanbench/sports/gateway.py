"""
SportsDataGateway: composes multiple adapters into a single facade.

Runners and predictors talk to the gateway, not individual adapters.
The gateway decides which adapter to call for each field and stitches
the results into a complete Match object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aegeanbench.sports.cache import FileCache, get_default_cache
from aegeanbench.sports.models import Match, Team
from aegeanbench.sports.sources import (
    FBrefAdapter,
    FetchPolicy,
    FootballDataAdapter,
    KaggleHistoryLoader,
    SoccersAPIAdapter,
)
from aegeanbench.sports.sources.openweather import (
    HOST_CITIES,
    OpenWeatherAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchContext:  # noqa: D101
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
    # Optional pre-match weather snapshot from OpenWeatherMap. None when
    # no API key is configured or the venue isn't known.
    weather: Optional[Dict[str, Any]] = None
    # Optional chat-room snippet forwarded by the front-end so the
    # ChatAgent / consensus prompt can see what the audience is saying.
    # Carried on the context (not on Match.h2h_last5 - that would
    # corrupt real head-to-head data).
    chat_summary: Optional[str] = None
    # Optional aggregate H2H stats from football-data (all-time
    # encounters between the two sides). Shape:
    #   {"num_matches": 1, "total_goals": 2,
    #    "home": {"wins": 0, "draws": 1, "losses": 0, ...},
    #    "away": {"wins": 0, "draws": 1, "losses": 0, ...}}
    h2h_aggregate: Optional[Dict[str, Any]] = None

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
        self.weather = OpenWeatherAdapter(mock=mock)

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

        # Enrich match with odds + real squads via football-data.
        if not match.odds:
            match.odds = self.soccersapi.fetch_odds(match.match_id)

        # Real 26-man squads from football-data (replaces soccersapi
        # mock lineup which only returns "Player 1..11" placeholders).
        if not match.home_lineup:
            match.home_lineup = self.football_data.fetch_squad(match.home_team.name or match.home_team.fifa_code)
        if not match.away_lineup:
            match.away_lineup = self.football_data.fetch_squad(match.away_team.name or match.away_team.fifa_code)

        # H2H from football-data's free aggregate endpoint. soccersapi's
        # h2h needs a paid plan; on the Soccer Odds plan we just hit mock.
        if not match.h2h_last5:
            h2h = self.soccersapi.fetch_h2h(
                match.home_team.fifa_code,
                match.away_team.fifa_code,
                match_id=match.match_id,
            )
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

        # Weather (optional). Look up the host city from venue text; fall
        # back to None when the venue doesn't match our known WC city list.
        weather = self._fetch_weather_for_match(match)

        # Real H2H aggregate from football-data (free tier returns this
        # even when the detailed list is gated). Best-effort, never raises.
        h2h_aggregate = None
        try:
            kickoff_iso = match.kickoff_at.date().isoformat() if match.kickoff_at else None
            fd_match_id = self.football_data.resolve_match_id(
                match.home_team.name or match.home_team.fifa_code,
                match.away_team.name or match.away_team.fifa_code,
                date_iso=kickoff_iso,
            )
            if fd_match_id:
                h2h_aggregate = self.football_data.fetch_h2h_aggregate(fd_match_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("h2h aggregate fetch failed: %s", exc)

        ctx = MatchContext(
            match=match,
            home_history=home_history,
            away_history=away_history,
            h2h=h2h,
            home_xg_profile=home_xg,
            away_xg_profile=away_xg,
            weather=weather,
            h2h_aggregate=h2h_aggregate,
        )

        return ctx

    def _fetch_weather_for_match(self, match: Match) -> Optional[Dict[str, Any]]:
        """
        Resolve venue text to a known city, then call OpenWeather.
        Returns None when the venue doesn't include a recognised city.
        """
        venue = (match.venue or "").lower()
        if not venue:
            return None
        for city in HOST_CITIES.keys():
            if city.lower() in venue:
                try:
                    return self.weather.fetch_for_match(city, match.kickoff_at)
                except Exception as e:
                    logger.warning("weather fetch failed for %s: %s", city, e)
                    return None
        return None

    def load_training_history(self, num_matches: int = 500) -> List[Match]:
        """For Dixon-Coles / Elo training: pull many historical matches."""
        return self.history_loader.load(num_matches=num_matches)
