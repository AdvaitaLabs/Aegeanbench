"""
Adapter for football-data.org TIER ONE.

Sprint mode: returns mock data so we can develop without waiting for the
API key. When AEGEANBENCH_FOOTBALL_DATA_KEY is set and policy.mock=False,
falls through to the real API.

Mock dataset covers 8 representative World Cup 2026 teams (groups A and B
in the canonical FIFA draw template). Extend WORLDCUP_2026_TEAMS_MOCK when
the real 32-team list is final.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional

from aegeanbench.sports.models import (
    CompetitionStage,
    Match,
    MatchOutcome,
    MatchResult,
    Odds,
    Team,
)
from aegeanbench.sports.sources.base import FetchPolicy, SourceAdapter

logger = logging.getLogger(__name__)

API_BASE = "https://api.football-data.org/v4"


# ---------- mock data ----------

WORLDCUP_2026_TEAMS_MOCK: List[Team] = [
    Team("BRA", "Brazil", "巴西", fifa_rank=5, elo_rating=1981.0, group="A"),
    Team("ARG", "Argentina", "阿根廷", fifa_rank=1, elo_rating=2114.0, group="A"),
    Team("FRA", "France", "法国", fifa_rank=2, elo_rating=2058.0, group="B"),
    Team("GER", "Germany", "德国", fifa_rank=10, elo_rating=1918.0, group="B"),
    Team("ESP", "Spain", "西班牙", fifa_rank=8, elo_rating=1955.0, group="C"),
    Team("ENG", "England", "英格兰", fifa_rank=4, elo_rating=2026.0, group="C"),
    Team("POR", "Portugal", "葡萄牙", fifa_rank=6, elo_rating=1975.0, group="D"),
    Team("NED", "Netherlands", "荷兰", fifa_rank=7, elo_rating=1962.0, group="D"),
]


def _mock_match(
    match_id: str,
    home: Team,
    away: Team,
    kickoff: datetime,
    stage: CompetitionStage = CompetitionStage.GROUP,
    result: Optional[MatchResult] = None,
) -> Match:
    return Match(
        match_id=match_id,
        competition="FIFA World Cup 2026",
        stage=stage,
        kickoff_at=kickoff,
        home_team=home,
        away_team=away,
        venue="MetLife Stadium, New Jersey",
        result=result,
    )


# A canonical mock fixture list (8 group-stage matches across 4 groups)
def _mock_fixtures() -> List[Match]:
    base = datetime(2026, 6, 12, 18, 0)
    teams = WORLDCUP_2026_TEAMS_MOCK
    fixtures = [
        # Group A
        _mock_match("WC2026-A1", teams[0], teams[1], base),
        _mock_match("WC2026-A2", teams[1], teams[0], base + timedelta(days=5)),
        # Group B
        _mock_match("WC2026-B1", teams[2], teams[3], base + timedelta(days=1)),
        _mock_match("WC2026-B2", teams[3], teams[2], base + timedelta(days=6)),
        # Group C
        _mock_match("WC2026-C1", teams[4], teams[5], base + timedelta(days=2)),
        _mock_match("WC2026-C2", teams[5], teams[4], base + timedelta(days=7)),
        # Group D
        _mock_match("WC2026-D1", teams[6], teams[7], base + timedelta(days=3)),
        _mock_match("WC2026-D2", teams[7], teams[6], base + timedelta(days=8)),
    ]
    return fixtures


def _mock_team_history(fifa_code: str, last_n: int) -> List[Match]:
    """Generate plausible recent results for a team."""
    teams_by_code = {t.fifa_code: t for t in WORLDCUP_2026_TEAMS_MOCK}
    if fifa_code not in teams_by_code:
        return []
    target = teams_by_code[fifa_code]
    opponents = [t for t in WORLDCUP_2026_TEAMS_MOCK if t.fifa_code != fifa_code]

    history = []
    base = datetime(2026, 5, 1)
    for i in range(min(last_n, len(opponents))):
        opp = opponents[i]
        # Make Argentina and France win slightly more often (rank-based)
        target_strong = target.fifa_rank is not None and target.fifa_rank <= 5
        result = MatchResult(
            home_goals=2 if target_strong else 1,
            away_goals=1,
        )
        history.append(
            Match(
                match_id=f"FRIENDLY-{fifa_code}-{i}",
                competition="International Friendly",
                stage=CompetitionStage.FRIENDLY,
                kickoff_at=base - timedelta(days=i * 14),
                home_team=target,
                away_team=opp,
                result=result,
            )
        )
    return history


# ---------- adapter ----------


class FootballDataAdapter(SourceAdapter):
    name = "football_data"
    has_fixtures = True
    has_historical = True

    def __init__(self, api_key: Optional[str] = None, mock_by_default: Optional[bool] = None):
        key = api_key or os.getenv("AEGEANBENCH_FOOTBALL_DATA_KEY")
        # Default to mock when no key is configured
        default_mock = mock_by_default if mock_by_default is not None else (key is None)
        super().__init__(api_key=key, mock_by_default=default_mock)

    def fetch_teams(self, competition: str, policy: Optional[FetchPolicy] = None) -> List[Team]:
        policy = self._resolve_policy(policy)
        if policy.mock:
            return list(WORLDCUP_2026_TEAMS_MOCK)
        return self._real_fetch_teams(competition, policy)

    def fetch_fixtures(
        self,
        competition: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        policy: Optional[FetchPolicy] = None,
    ) -> List[Match]:
        policy = self._resolve_policy(policy)
        if policy.mock:
            matches = _mock_fixtures()
            if from_date:
                matches = [m for m in matches if m.kickoff_at >= from_date]
            if to_date:
                matches = [m for m in matches if m.kickoff_at <= to_date]
            return matches
        return self._real_fetch_fixtures(competition, from_date, to_date, policy)

    def fetch_team_history(
        self,
        fifa_code: str,
        last_n: int = 10,
        policy: Optional[FetchPolicy] = None,
    ) -> List[Match]:
        policy = self._resolve_policy(policy)
        if policy.mock:
            return _mock_team_history(fifa_code, last_n)
        return self._real_fetch_team_history(fifa_code, last_n, policy)

    # ---------- real API (live HTTP path) ----------
    #
    # football-data.org reference:
    #   GET /v4/competitions/{code}/teams    free tier covers WC, EC, ...
    #   GET /v4/competitions/{code}/matches  paginated, supports date filters
    # Auth: X-Auth-Token header, free key = 10 req/min
    # Competition codes: WC for World Cup, EC for European Championship.

    COMPETITION_CODES = {
        "FIFA World Cup 2026": "WC",
        "FIFA World Cup": "WC",
        "WC": "WC",
        "UEFA Euro 2024": "EC",
        "EC": "EC",
    }

    def _competition_code(self, competition: str) -> str:
        return self.COMPETITION_CODES.get(competition, competition.upper())

    def _headers(self) -> dict:
        return {"X-Auth-Token": self.api_key} if self.api_key else {}

    def _real_fetch_teams(self, competition: str, policy: FetchPolicy) -> List[Team]:
        import requests
        code = self._competition_code(competition)
        url = f"{API_BASE}/competitions/{code}/teams"
        try:
            resp = requests.get(url, headers=self._headers(), timeout=policy.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("football_data fetch_teams failed (%s); falling back to mock", e)
            return list(WORLDCUP_2026_TEAMS_MOCK)

        teams: List[Team] = []
        for raw in data.get("teams", []):
            fifa_code = (raw.get("tla") or raw.get("shortName") or "").upper()
            name = raw.get("name") or fifa_code
            if not fifa_code:
                continue
            teams.append(Team(fifa_code=fifa_code, name=name))
        if not teams:
            logger.warning("football_data returned 0 teams; using mock")
            return list(WORLDCUP_2026_TEAMS_MOCK)
        return teams

    def _real_fetch_fixtures(
        self,
        competition: str,
        from_date: Optional[datetime],
        to_date: Optional[datetime],
        policy: FetchPolicy,
    ) -> List[Match]:
        import requests
        code = self._competition_code(competition)
        url = f"{API_BASE}/competitions/{code}/matches"
        params: dict = {}
        if from_date:
            params["dateFrom"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["dateTo"] = to_date.strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params,
                timeout=policy.timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("football_data fetch_fixtures failed (%s); falling back to mock", e)
            return _mock_fixtures()

        return self._parse_fixtures(data.get("matches", []), competition)

    def _real_fetch_team_history(
        self, fifa_code: str, last_n: int, policy: FetchPolicy
    ) -> List[Match]:
        """Free tier: /v4/teams/{id}/matches is restricted; degrade to mock."""
        logger.info("football_data team history requires paid tier; using mock for %s", fifa_code)
        return _mock_team_history(fifa_code, last_n)

    @staticmethod
    def _parse_fixtures(rows: list, competition: str) -> List[Match]:
        """Convert raw football-data /matches rows into our Match dataclass."""
        out: List[Match] = []
        for r in rows:
            home_raw = r.get("homeTeam") or {}
            away_raw = r.get("awayTeam") or {}
            home_code = (home_raw.get("tla") or home_raw.get("shortName") or "").upper()
            away_code = (away_raw.get("tla") or away_raw.get("shortName") or "").upper()
            if not home_code or not away_code:
                continue
            kickoff_str = r.get("utcDate") or r.get("kickoff")
            try:
                kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                kickoff = datetime.utcnow()

            stage_raw = (r.get("stage") or "").upper()
            stage_map = {
                "GROUP_STAGE": CompetitionStage.GROUP,
                "LAST_16": CompetitionStage.ROUND_OF_16,
                "QUARTER_FINALS": CompetitionStage.QUARTER_FINAL,
                "SEMI_FINALS": CompetitionStage.SEMI_FINAL,
                "THIRD_PLACE": CompetitionStage.THIRD_PLACE,
                "FINAL": CompetitionStage.FINAL,
            }
            stage = stage_map.get(stage_raw, CompetitionStage.GROUP)

            result = None
            score = (r.get("score") or {}).get("fullTime") or {}
            home_goals = score.get("home")
            away_goals = score.get("away")
            if home_goals is not None and away_goals is not None:
                result = MatchResult(home_goals=int(home_goals), away_goals=int(away_goals))

            out.append(
                Match(
                    match_id=str(r.get("id") or f"{home_code}-{away_code}-{kickoff:%Y%m%d}"),
                    competition=competition,
                    stage=stage,
                    kickoff_at=kickoff,
                    home_team=Team(home_code, home_raw.get("name", home_code)),
                    away_team=Team(away_code, away_raw.get("name", away_code)),
                    venue=r.get("venue"),
                    result=result,
                )
            )
        return out
