"""
Adapter for SoccersAPI (soccersapi.com).

Provides odds, lineups, h2h, and injuries. Real API integration is a stub
until key lands; mock returns plausible numbers shaped after real data.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

from aegeanbench.sports.models import (
    CompetitionStage,
    Match,
    MatchResult,
    Odds,
    Player,
    Team,
)
from aegeanbench.sports.sources.base import FetchPolicy, SourceAdapter

logger = logging.getLogger(__name__)

API_BASE = "https://api.soccersapi.com/v2.2"


# ---------- mock data ----------

# Plausible odds shaped after typical World Cup opening match prices.
# Margin ~ 5-7% (typical for top-tier bookmakers).
def _mock_odds(home_strong: bool = True) -> List[Odds]:
    now = datetime.now()
    if home_strong:
        return [
            Odds("Pinnacle", now, home_win=1.85, draw=3.60, away_win=4.20,
                 over_2_5=1.95, under_2_5=1.85, btts_yes=1.80, btts_no=2.00),
            Odds("Bet365", now, home_win=1.90, draw=3.50, away_win=4.00,
                 over_2_5=2.00, under_2_5=1.80),
            Odds("William Hill", now, home_win=1.88, draw=3.55, away_win=4.10),
        ]
    return [
        Odds("Pinnacle", now, home_win=2.50, draw=3.20, away_win=2.90),
        Odds("Bet365", now, home_win=2.55, draw=3.15, away_win=2.85),
        Odds("William Hill", now, home_win=2.45, draw=3.25, away_win=2.95),
    ]


def _mock_player(player_id: str, name: str, fifa_code: str, position: str, **extra) -> Player:
    defaults = dict(
        age=28,
        club="Top Club FC",
        matches_played_for_team=50,
        goals_for_team=10,
    )
    defaults.update(extra)
    return Player(
        player_id=player_id,
        name=name,
        team_fifa_code=fifa_code,
        position=position,
        **defaults,
    )


def _mock_lineup(team_fifa_code: str) -> List[Player]:
    """Return 11 mock starters. Star players hard-coded for top sides."""
    stars = {
        "ARG": [("messi", "Lionel Messi", "FW", {"goals_for_team": 109, "age": 38})],
        "BRA": [("vini-jr", "Vinicius Junior", "FW", {"goals_for_team": 12, "age": 25})],
        "FRA": [("mbappe", "Kylian Mbappe", "FW", {"goals_for_team": 50, "age": 27})],
        "POR": [("ronaldo", "Cristiano Ronaldo", "FW", {"goals_for_team": 130, "age": 41})],
        "ENG": [("bellingham", "Jude Bellingham", "MF", {"goals_for_team": 8, "age": 22})],
        "GER": [("musiala", "Jamal Musiala", "MF", {"goals_for_team": 6, "age": 22})],
        "ESP": [("rodri", "Rodri Hernandez", "MF", {"goals_for_team": 4, "age": 29})],
        "NED": [("van-dijk", "Virgil van Dijk", "DF", {"goals_for_team": 9, "age": 34})],
    }
    star_specs = stars.get(team_fifa_code, [])
    players: List[Player] = []
    for pid, name, pos, extra in star_specs:
        players.append(_mock_player(f"{team_fifa_code}-{pid}", name, team_fifa_code, pos, **extra))
    # Fill to 11
    while len(players) < 11:
        idx = len(players) + 1
        pos = "GK" if idx == 1 else ("DF" if idx <= 5 else ("MF" if idx <= 8 else "FW"))
        players.append(
            _mock_player(f"{team_fifa_code}-p{idx}", f"Player {idx}", team_fifa_code, pos)
        )
    return players


def _mock_h2h(home_fifa: str, away_fifa: str, last_n: int) -> List[Match]:
    base = datetime(2024, 1, 1)
    matches = []
    for i in range(min(last_n, 5)):
        # Alternate winners with one draw for variety
        if i == 2:
            r = MatchResult(home_goals=1, away_goals=1)
        elif i % 2 == 0:
            r = MatchResult(home_goals=2, away_goals=1)
        else:
            r = MatchResult(home_goals=0, away_goals=2)
        matches.append(
            Match(
                match_id=f"H2H-{home_fifa}-{away_fifa}-{i}",
                competition="International Friendly",
                stage=CompetitionStage.FRIENDLY,
                kickoff_at=base - timedelta(days=i * 180),
                home_team=Team(home_fifa, home_fifa),
                away_team=Team(away_fifa, away_fifa),
                result=r,
            )
        )
    return matches


# ---------- adapter ----------


class SoccersAPIAdapter(SourceAdapter):
    name = "soccersapi"
    has_odds = True
    has_lineups = True

    # SoccersAPI auth pattern: ?user=USER&token=TOKEN
    # The `user` is the account username (from the SoccersAPI dashboard).
    # We accept either constructor injection or env-var configuration.

    def __init__(
        self,
        api_key: Optional[str] = None,
        user: Optional[str] = None,
        mock_by_default: Optional[bool] = None,
    ):
        key = api_key or os.getenv("AEGEANBENCH_SOCCERSAPI_KEY")
        self.user = user or os.getenv("AEGEANBENCH_SOCCERSAPI_USER", "")
        default_mock = mock_by_default if mock_by_default is not None else (key is None)
        super().__init__(api_key=key, mock_by_default=default_mock)

    def _real_fetch_odds(self, match_id: str, policy: FetchPolicy) -> List[Odds]:
        """
        Pull pre-match odds from SoccersAPI.

        Endpoint:
            GET /v2.2/fixtures/?t=match_odds&id=<fixture_id>

        Response shape (verified 2026-06-09 against live API):
            { "data": [
                {
                  "id": 1,
                  "name": "1X2, Full Time Result",
                  "bookmakers": [
                    { "id": 2, "name": "Bet365",
                      "odds": { "data": { "home": "1.500", "draw": "4.000",
                                          "away": "5.500", ... } } },
                    ...
                  ]
                },
                { "id": 3, "name": "Asian Handicap", ... },
                { "id": 2, "name": "Over/Under, Goal Line", ... }
              ] }

        We only consume the 1X2 (Full Time Result) market for our betting
        layer. Asian Handicap and Over/Under are ignored in V1.
        """
        import requests
        try:
            r = requests.get(
                f"{API_BASE}/fixtures/",
                params={
                    "user": self.user,
                    "token": self.api_key,
                    "t": "match_odds",
                    "id": match_id,
                },
                timeout=policy.timeout_seconds,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            logger.warning("soccersapi match_odds fetch failed for %s: %s", match_id, e)
            return []

        markets = payload.get("data") or []
        if not isinstance(markets, list):
            markets = [markets]

        # Find the 1X2 / Full Time Result market
        ftr = None
        for market in markets:
            name = (market.get("name") or "").lower()
            if "1x2" in name or "full time" in name or "match winner" in name:
                ftr = market
                break
        if ftr is None:
            logger.debug("soccersapi: no 1X2 market for fixture %s", match_id)
            return []

        out: List[Odds] = []
        for bk in ftr.get("bookmakers", []):
            bk_name = bk.get("name") or f"bookmaker_{bk.get('id')}"
            data = ((bk.get("odds") or {}).get("data")) or {}
            try:
                home = float(data.get("home") or 0)
                draw = float(data.get("draw") or 0)
                away = float(data.get("away") or 0)
            except (TypeError, ValueError):
                continue
            if home <= 1.0 or draw <= 1.0 or away <= 1.0:
                continue
            out.append(
                Odds(
                    bookmaker=bk_name,
                    timestamp=datetime.now(),
                    home_win=home,
                    draw=draw,
                    away_win=away,
                )
            )
        return out

    def _real_fetch_lineup(
        self, match_id: str, team_fifa_code: str, policy: FetchPolicy
    ) -> List[Player]:
        """
        Pull real starting XI from SoccersAPI.

        Endpoint:
            GET /v2.2/fixtures/?t=match_lineups&id=<fixture_id>

        Response shape (verified live):
            { "data": {
                "localteam": { "id": .., "lineup": [
                    {"player_id": "...", "player_name": "...",
                     "team_id": .., "number": .., "position": "...",
                     "type": "lineup"|"bench"},
                    ...
                ]},
                "visitorteam": { ... }
              } }
        """
        import requests
        try:
            r = requests.get(
                f"{API_BASE}/fixtures/",
                params={
                    "user": self.user,
                    "token": self.api_key,
                    "t": "match_lineups",
                    "id": match_id,
                },
                timeout=policy.timeout_seconds,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            logger.warning("soccersapi match_lineups fetch failed for %s: %s", match_id, e)
            return _mock_lineup(team_fifa_code)

        data = payload.get("data") or {}
        if isinstance(data, list):
            data = data[0] if data else {}

        # Figure out which side the caller asked for. We try the team's
        # name / short_code first since that's what the gateway has;
        # fall back to localteam when the heuristic fails.
        wanted = (team_fifa_code or "").upper()
        local = data.get("localteam") or {}
        visitor = data.get("visitorteam") or {}

        def _team_matches(team_block: Dict, code: str) -> bool:
            name = (team_block.get("name") or "").upper()
            short = (team_block.get("short_code") or "").upper()
            iso = (team_block.get("country_iso") or "").upper()
            return code in {short, iso} or code in name

        if _team_matches(local, wanted):
            chosen = local
        elif _team_matches(visitor, wanted):
            chosen = visitor
        else:
            chosen = local  # best guess

        lineup_rows = chosen.get("lineup") or []
        out: List[Player] = []
        for row in lineup_rows:
            # Skip bench rows when explicitly tagged
            if str(row.get("type") or "").lower() == "bench":
                continue
            out.append(
                Player(
                    player_id=str(row.get("player_id") or row.get("id") or ""),
                    name=row.get("player_name") or row.get("name") or "Unknown",
                    team_fifa_code=team_fifa_code,
                    position=str(row.get("position") or "?").upper()[:2],
                    age=int(row["age"]) if str(row.get("age") or "").isdigit() else None,
                    club=row.get("team") or row.get("club"),
                    matches_played_for_team=int(row.get("appearances") or 0),
                    goals_for_team=int(row.get("goals") or 0),
                )
            )

        if not out:
            logger.debug("soccersapi lineup empty for %s/%s, using mock", match_id, team_fifa_code)
            return _mock_lineup(team_fifa_code)
        return out

    def _real_fetch_h2h(
        self,
        home_fifa: str,
        away_fifa: str,
        last_n: int,
        policy: FetchPolicy,
        match_id: Optional[str] = None,
    ) -> List[Match]:
        """
        Pull real head-to-head history from SoccersAPI.

        Endpoint:
            GET /v2.2/fixtures/?t=match_h2h&id=<fixture_id>

        Response shape (verified live):
            { "data": [
                {"id": ..., "match_start": "2024-09-06 19:00:00",
                 "league_name": "...",
                 "teams": {
                   "home": {"name": "...", "short_code": "..."},
                   "away": {"name": "...", "short_code": "..."}
                 },
                 "scores": {"ft_score": "2-1", "ht_score": "1-0"}
                },
                ...
              ] }

        Requires a fixture context to call. If the caller didn't pass
        match_id we fall back to mock so we never break the predict path.
        """
        if not match_id:
            logger.debug("soccersapi h2h needs match_id; using mock")
            return _mock_h2h(home_fifa, away_fifa, last_n)

        import requests
        try:
            r = requests.get(
                f"{API_BASE}/fixtures/",
                params={
                    "user": self.user,
                    "token": self.api_key,
                    "t": "match_h2h",
                    "id": match_id,
                },
                timeout=policy.timeout_seconds,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            logger.warning("soccersapi match_h2h fetch failed for %s: %s", match_id, e)
            return _mock_h2h(home_fifa, away_fifa, last_n)

        rows = payload.get("data") or []
        if not isinstance(rows, list):
            rows = [rows]

        out: List[Match] = []
        for row in rows[:last_n]:
            try:
                teams = row.get("teams") or {}
                home = teams.get("home") or {}
                away = teams.get("away") or {}
                scores = row.get("scores") or {}
                ft = str(scores.get("ft_score") or "").strip()
                ht = str(scores.get("ht_score") or "").strip()
                if "-" not in ft:
                    continue
                hg, ag = ft.split("-", 1)
                kickoff = datetime.now()
                ts = row.get("match_start") or row.get("date")
                if ts:
                    try:
                        kickoff = datetime.fromisoformat(str(ts).replace(" ", "T"))
                    except ValueError:
                        pass
                home_code = (home.get("short_code") or home.get("country_iso") or home.get("name") or "")[:3].upper()
                away_code = (away.get("short_code") or away.get("country_iso") or away.get("name") or "")[:3].upper()
                result = MatchResult(
                    home_goals=int(hg.strip()),
                    away_goals=int(ag.strip()),
                )
                if "-" in ht:
                    try:
                        hh, ah = ht.split("-", 1)
                        result.ht_home_goals = int(hh.strip())
                        result.ht_away_goals = int(ah.strip())
                    except ValueError:
                        pass
                out.append(
                    Match(
                        match_id=f"H2H-{row.get('id', '?')}",
                        competition=row.get("league_name") or "International",
                        stage=CompetitionStage.FRIENDLY,
                        kickoff_at=kickoff,
                        home_team=Team(home_code, home.get("name") or home_code),
                        away_team=Team(away_code, away.get("name") or away_code),
                        result=result,
                    )
                )
            except Exception as parse_err:
                logger.debug("h2h row parse failed: %s", parse_err)
                continue

        if not out:
            return _mock_h2h(home_fifa, away_fifa, last_n)
        return out

    def fetch_odds(self, match_id: str, policy: Optional[FetchPolicy] = None) -> List[Odds]:
        policy = self._resolve_policy(policy)
        if policy.mock:
            # Heuristic: matches with Argentina/Brazil/France as home → home strong
            home_strong = any(code in match_id for code in ("ARG", "BRA", "FRA", "ENG"))
            return _mock_odds(home_strong=home_strong)
        return self._real_fetch_odds(match_id, policy)

    def fetch_lineup(
        self, match_id: str, team_fifa_code: str, policy: Optional[FetchPolicy] = None
    ) -> List[Player]:
        policy = self._resolve_policy(policy)
        if policy.mock:
            return _mock_lineup(team_fifa_code)
        return self._real_fetch_lineup(match_id, team_fifa_code, policy)

    def fetch_h2h(
        self,
        home_fifa: str,
        away_fifa: str,
        last_n: int = 5,
        policy: Optional[FetchPolicy] = None,
        match_id: Optional[str] = None,
    ) -> List[Match]:
        """
        Real H2H needs a fixture id (soccersapi's match_h2h endpoint).
        match_id is optional only because the base class signature
        doesn't have it; pass it when you have it.
        """
        policy = self._resolve_policy(policy)
        if policy.mock:
            return _mock_h2h(home_fifa, away_fifa, last_n)
        return self._real_fetch_h2h(home_fifa, away_fifa, last_n, policy, match_id=match_id)

    # (legacy stubs removed - see live implementations above)
