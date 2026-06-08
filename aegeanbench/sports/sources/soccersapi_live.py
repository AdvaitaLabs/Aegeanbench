"""
Live-scores client for SoccersAPI.

Subscription used:
    - "足球 API 世界杯" (World Cup package): /livescores/, /fixtures/, /lineups/
    - "足球 API 赔率"   (odds package):       /odds/

Auth follows the SoccersAPI documented pattern:
    ?user=<USER>&token=<TOKEN>

Set both env vars at startup:
    AEGEANBENCH_SOCCERSAPI_USER
    AEGEANBENCH_SOCCERSAPI_KEY    (the "token")

All endpoints fall back to a mock fixture / event list when no key is
configured so the rest of the pipeline keeps working offline.

The shape of LiveMatchState below is intentionally minimal - just the
fields a downstream scheduler needs to decide whether to re-trigger
consensus or push a WebSocket event.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


SOCCERSAPI_BASE = "https://api.soccersapi.com/v2.2"
POLL_INTERVAL_S = 30.0


# ----------------------------- event types -----------------------------


class LiveEventKind(str, Enum):
    """The five event categories we react to in real time."""
    GOAL = "goal"
    OWN_GOAL = "own_goal"
    PENALTY = "penalty"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    KICK_OFF = "kick_off"


@dataclass
class LiveEvent:
    """A single in-match event from SoccersAPI."""
    match_id: str
    event_id: str
    kind: LiveEventKind
    minute: int
    team_fifa_code: Optional[str] = None    # which team triggered it
    player_name: Optional[str] = None
    detail: str = ""                         # extra description if any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_high_priority(self) -> bool:
        """Events that should immediately re-trigger consensus."""
        return self.kind in {
            LiveEventKind.GOAL,
            LiveEventKind.OWN_GOAL,
            LiveEventKind.PENALTY,
            LiveEventKind.RED_CARD,
            LiveEventKind.HALF_TIME,
            LiveEventKind.FULL_TIME,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "event_id": self.event_id,
            "kind": self.kind.value,
            "minute": self.minute,
            "team": self.team_fifa_code,
            "player": self.player_name,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LiveMatchState:
    """Current state of a match. Snapshot returned by polling."""
    match_id: str
    status: str                          # "scheduled" / "in_play" / "ht" / "ft" / "postponed"
    minute: int = 0
    home_goals: int = 0
    away_goals: int = 0
    home_team: str = ""
    away_team: str = ""
    last_event_id: Optional[str] = None  # to dedupe across polls
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_finished(self) -> bool:
        return self.status in ("ft", "after_extra_time", "after_penalties")

    @property
    def is_live(self) -> bool:
        return self.status in ("in_play", "first_half", "second_half", "ht", "extra_time")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "status": self.status,
            "minute": self.minute,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "fetched_at": self.fetched_at.isoformat(),
        }


# ----------------------------- client -----------------------------


class SoccersAPILiveClient:
    """
    Thin client around SoccersAPI's livescores + match endpoints.

    Polling pattern (recommended):
        client = SoccersAPILiveClient()
        while True:
            for state in client.fetch_live_matches(tournament="world-cup"):
                ...react...
            sleep(POLL_INTERVAL_S)
    """

    def __init__(
        self,
        user: Optional[str] = None,
        token: Optional[str] = None,
        mock: Optional[bool] = None,
        timeout: float = 6.0,
    ):
        # SoccersAPI auth: ?user=<USER>&token=<TOKEN>. Some packages allow
        # token-only authentication; we keep both fields optional so the
        # adapter still works if only the token is configured.
        self.user = user or os.getenv("AEGEANBENCH_SOCCERSAPI_USER", "")
        self.token = token or os.getenv("AEGEANBENCH_SOCCERSAPI_KEY", "")
        if mock is None:
            # Only the token is strictly required - if we have it we go live.
            mock = not self.token
        self.mock = mock
        self.timeout = timeout
        # Track last-seen event_id per match so we can de-duplicate
        self._last_event_seen: Dict[str, str] = {}

    # ----------------- live scores -----------------

    def fetch_live_matches(
        self,
        tournament: str = "world-cup",
    ) -> List[LiveMatchState]:
        """Return current state of every live match in the tournament."""
        if self.mock:
            return self._mock_live_matches()
        try:
            return self._real_live_matches(tournament)
        except Exception as e:
            logger.warning("soccersapi live fetch failed (%s); returning empty", e)
            return []

    def fetch_match_events(
        self,
        match_id: str,
        since_event_id: Optional[str] = None,
    ) -> List[LiveEvent]:
        """
        Return events for one match. Optionally filter to events newer
        than the last-seen ID for that match (caller can pass either
        explicit since_event_id or rely on this client's internal cache).
        """
        if self.mock:
            return self._mock_events(match_id)
        try:
            events = self._real_events(match_id)
        except Exception as e:
            logger.warning("soccersapi events fetch failed for %s (%s)", match_id, e)
            return []

        # Dedupe via last-seen tracker
        last_seen = since_event_id or self._last_event_seen.get(match_id)
        fresh = self._dedupe_events(events, last_seen)
        if fresh:
            self._last_event_seen[match_id] = fresh[-1].event_id
        return fresh

    # ----------------- real HTTP path -----------------

    def _auth_params(self) -> Dict[str, str]:
        return {"user": self.user or "", "token": self.token or ""}

    def _real_live_matches(self, tournament: str) -> List[LiveMatchState]:
        import requests

        params = self._auth_params()
        params["t"] = "livescores"
        # SoccersAPI lets you filter by league_id; the World Cup package
        # has a fixed numeric ID we pass through.
        if tournament == "world-cup":
            params["league_id"] = os.getenv("SOCCERSAPI_WORLD_CUP_LEAGUE_ID", "")
        url = f"{SOCCERSAPI_BASE}/livescores/"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data") or data.get("livescores") or []
        return [self._parse_match_row(r) for r in rows if r]

    def _real_events(self, match_id: str) -> List[LiveEvent]:
        import requests

        params = self._auth_params()
        params["t"] = "matches"
        params["id"] = match_id
        url = f"{SOCCERSAPI_BASE}/matches/"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        body = data.get("data") or {}
        if isinstance(body, list) and body:
            body = body[0]
        events_raw = body.get("events") or []
        return [self._parse_event_row(match_id, r) for r in events_raw if r]

    @staticmethod
    def _parse_match_row(row: Dict[str, Any]) -> LiveMatchState:
        home = row.get("home") or row.get("home_team") or {}
        away = row.get("away") or row.get("away_team") or {}
        score = row.get("score") or {}
        return LiveMatchState(
            match_id=str(row.get("id") or row.get("match_id") or ""),
            status=str(row.get("status") or "scheduled").lower(),
            minute=int(row.get("minute") or 0),
            home_goals=int(score.get("home", row.get("home_score", 0)) or 0),
            away_goals=int(score.get("away", row.get("away_score", 0)) or 0),
            home_team=str(home.get("name") or home.get("short_code") or ""),
            away_team=str(away.get("name") or away.get("short_code") or ""),
        )

    @staticmethod
    def _parse_event_row(match_id: str, row: Dict[str, Any]) -> LiveEvent:
        kind_map = {
            "goal": LiveEventKind.GOAL,
            "own_goal": LiveEventKind.OWN_GOAL,
            "penalty": LiveEventKind.PENALTY,
            "yellowcard": LiveEventKind.YELLOW_CARD,
            "redcard": LiveEventKind.RED_CARD,
            "substitution": LiveEventKind.SUBSTITUTION,
            "ht": LiveEventKind.HALF_TIME,
            "ft": LiveEventKind.FULL_TIME,
            "ko": LiveEventKind.KICK_OFF,
        }
        raw_kind = str(row.get("type") or row.get("event") or "").lower().replace("-", "_").replace(" ", "_")
        kind = kind_map.get(raw_kind, LiveEventKind.GOAL)
        return LiveEvent(
            match_id=match_id,
            event_id=str(row.get("id") or row.get("event_id") or ""),
            kind=kind,
            minute=int(row.get("minute") or 0),
            team_fifa_code=row.get("team_code") or row.get("team_short") or None,
            player_name=row.get("player") or row.get("player_name") or None,
            detail=str(row.get("info") or row.get("detail") or ""),
        )

    @staticmethod
    def _dedupe_events(events: List[LiveEvent], last_seen: Optional[str]) -> List[LiveEvent]:
        if not last_seen:
            return events
        # Keep only events strictly after last_seen (assuming server returns
        # in chronological order)
        out: List[LiveEvent] = []
        seen = False
        for ev in events:
            if seen:
                out.append(ev)
            elif ev.event_id == last_seen:
                seen = True
        # If we never spotted last_seen, return everything (safer than dropping)
        return out if seen else events

    # ----------------- mock data -----------------

    @staticmethod
    def _mock_live_matches() -> List[LiveMatchState]:
        return [
            LiveMatchState(
                match_id="WC2026-A1",
                status="in_play",
                minute=23,
                home_goals=1, away_goals=0,
                home_team="Brazil", away_team="Argentina",
            ),
        ]

    @staticmethod
    def _mock_events(match_id: str) -> List[LiveEvent]:
        return [
            LiveEvent(
                match_id=match_id, event_id="ev_1",
                kind=LiveEventKind.KICK_OFF, minute=0,
            ),
            LiveEvent(
                match_id=match_id, event_id="ev_2",
                kind=LiveEventKind.GOAL, minute=23,
                team_fifa_code="BRA", player_name="Vinicius Junior",
                detail="header",
            ),
        ]
