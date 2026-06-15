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
    # In-play snapshot from soccersapi /livescores when the match is
    # currently live or finished. None when the match hasn't started.
    # Shape:
    #   {"status": "inplay"|"ht"|"ft", "minute": 67,
    #    "home_goals": 1, "away_goals": 0,
    #    "recent_events": [{"minute": 38, "kind": "goal",
    #                       "team": "MEX", "player": "Vela"}, ...]}
    live_state: Optional[Dict[str, Any]] = None

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

        # Live state: ONLY relevant if the match is in-play or just
        # ended. soccersapi only returns it when the match is among
        # the current livescores set, so this returns None during
        # pre-match silence — that's the right behaviour.
        live_state = self._fetch_live_state_for_match(match)

        ctx = MatchContext(
            match=match,
            home_history=home_history,
            away_history=away_history,
            h2h=h2h,
            home_xg_profile=home_xg,
            away_xg_profile=away_xg,
            weather=weather,
            h2h_aggregate=h2h_aggregate,
            live_state=live_state,
        )

        return ctx

    # World Cup 2026 venue -> host city map. football-data returns the
    # stadium name ("Azteca", "MetLife Stadium") not the city, so we
    # need this lookup before calling OpenWeather. Substring match is
    # used so "Estadio Azteca" still resolves to Mexico City.
    _VENUE_TO_CITY: Dict[str, str] = {
        # Mexico
        "azteca":          "Mexico City",
        "estadio azteca":  "Mexico City",
        "bbva":            "Monterrey",
        "estadio bbva":    "Monterrey",
        "akron":           "Guadalajara",
        "estadio akron":   "Guadalajara",
        # USA
        "at&t stadium":    "Dallas",
        "at&t":            "Dallas",
        "sofi":            "Los Angeles",
        "sofi stadium":    "Los Angeles",
        "metlife":         "New York",
        "metlife stadium": "New York",
        "mercedes-benz":   "Atlanta",
        "lincoln financial": "Philadelphia",
        "hard rock":       "Miami",
        "nrg":             "Houston",
        "nrg stadium":     "Houston",
        "levi's":          "San Francisco",
        "levis":           "San Francisco",
        "arrowhead":       "Kansas City",
        "gillette":        "Boston",
        "lumen":           "Seattle",
        "lumen field":     "Seattle",
        # Canada
        "bmo":             "Toronto",
        "bmo field":       "Toronto",
        "bc place":        "Vancouver",
    }

    def _resolve_venue_city(self, venue_text: str) -> Optional[str]:
        v = (venue_text or "").lower().strip()
        if not v:
            return None
        # Try venue-name keys first (most specific)
        for key, city in self._VENUE_TO_CITY.items():
            if key in v:
                return city
        # Fall back to the older direct city-name match
        for city in HOST_CITIES.keys():
            if city.lower() in v:
                return city
        return None

    def _fetch_live_state_for_match(self, match: Match) -> Optional[Dict[str, Any]]:
        """
        Pull current live score + recent events.

        Resolution order:
          1. soccersapi t=live + match_events (has full event stream
             when it's caught up, but its `status` flag often lags 15-
             30 min behind real life on the Soccer 100 plan).
          2. football-data /matches/{fd_id} (no per-goal events, but
             status / minute / score update much faster — verified
             2026-06-15 reading IN_PLAY 57' / 2-1 while soccersapi
             still said Notstarted on the same match).

        Returns None when both sources have nothing (truly pre-match).
        """
        # ---- 1. soccersapi primary ----
        sa_state = self._fetch_live_state_soccersapi(match)
        if sa_state is not None:
            return sa_state

        # ---- 2. football-data fallback ----
        try:
            kickoff_iso = (match.kickoff_at.date().isoformat()
                           if match.kickoff_at else None)
            fd_match_id = self.football_data.resolve_match_id(
                match.home_team.name or match.home_team.fifa_code,
                match.away_team.name or match.away_team.fifa_code,
                date_iso=kickoff_iso,
            )
            if fd_match_id is None:
                return None
            fd_state = self.football_data.fetch_live_state(fd_match_id)
            if fd_state is None:
                return None
            # Only surface as "live state" when the match is actually
            # in-play or finished — scheduled matches are not what
            # we want to inject into the prompt.
            if fd_state.get("status") in ("inplay", "ht", "ft"):
                logger.info(
                    "live state for %s sourced from football-data (sa lag)",
                    match.match_id,
                )
                return fd_state
            return None
        except Exception as e:
            logger.warning(
                "football-data live state fallback failed for %s: %s",
                match.match_id, e,
            )
            return None

    def _fetch_live_state_soccersapi(self, match: Match) -> Optional[Dict[str, Any]]:
        """Primary live-state path: soccersapi t=live + match_events."""
        try:
            from aegeanbench.sports.sources.soccersapi_live import SoccersAPILiveClient
            client = SoccersAPILiveClient()
            live_state = None
            for m in client.fetch_live_matches():
                if str(m.match_id) == str(match.match_id):
                    live_state = m
                    break
            if live_state is None:
                return None

            events = client.fetch_match_events(match.match_id) or []
            recent_events = []
            for ev in events[-8:]:
                recent_events.append({
                    "minute": ev.minute,
                    "kind": ev.kind.value,
                    "team": ev.team_fifa_code,
                    "player": ev.player_name,
                    "detail": ev.detail,
                })

            return {
                "status": live_state.status,
                "minute": live_state.minute,
                "home_goals": live_state.home_goals,
                "away_goals": live_state.away_goals,
                "recent_events": recent_events,
                "source": "soccersapi",
            }
        except Exception as e:
            logger.warning("soccersapi live state failed for %s: %s", match.match_id, e)
            return None

    def _fetch_weather_for_match(self, match: Match) -> Optional[Dict[str, Any]]:
        """
        Resolve venue text to a known city, then call OpenWeather.
        Returns None when the venue doesn't include a recognised city.
        """
        city = self._resolve_venue_city(match.venue or "")
        if not city:
            return None
        try:
            return self.weather.fetch_for_match(city, match.kickoff_at)
        except Exception as e:
            logger.warning("weather fetch failed for %s: %s", city, e)
            return None

    def load_training_history(self, num_matches: int = 500) -> List[Match]:
        """For Dixon-Coles / Elo training: pull many historical matches."""
        return self.history_loader.load(num_matches=num_matches)
