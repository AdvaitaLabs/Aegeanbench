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
        return self.status in (
            "ft", "finished", "after_extra_time", "after_penalties",
            "ended",
        )

    @property
    def is_live(self) -> bool:
        return self.status in (
            "in_play", "inplay", "first_half", "second_half",
            "ht", "extra_time", "half_time",
        )

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
        """
        Return current state of every live match in the tournament.

        Best-effort: when SoccersAPI returns an "endpoint not in your plan"
        message, or the World Cup league_id filter is unsupported, we
        return an empty list rather than raising so the rest of the
        pipeline keeps running with football-data as primary source.
        """
        if self.mock:
            return self._mock_live_matches()
        try:
            matches = self._real_live_matches(tournament)
            # Plan-coverage check: a non-empty meta.msg means the call
            # technically succeeded but the data was clipped. We still
            # return whatever rows are present (often 0).
            if not matches:
                logger.info(
                    "soccersapi /livescores returned 0 matches for tournament=%s "
                    "(likely plan coverage; falling back to other sources)",
                    tournament,
                )
            return matches
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
        """
        Pull currently-live matches.

        Verified endpoint: GET /v2.2/livescores/?t=live
        We drop the league_id filter intentionally — empirically the
        filter returns 0 even when matches are clearly in play (verified
        2026-06-12 with Korea vs Czech). Without the filter the call
        returns all in-play matches; we filter to WC client-side by
        checking row.league.id == 377.
        """
        import requests

        params = self._auth_params()
        params["t"] = "live"
        url = f"{SOCCERSAPI_BASE}/livescores/"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data") or []
        if not isinstance(rows, list):
            rows = [rows]

        wc_only = (tournament == "world-cup")
        wc_league_id = int(os.getenv("SOCCERSAPI_WORLD_CUP_LEAGUE_ID", "377") or 377)
        out: List[LiveMatchState] = []
        for r in rows:
            if not r:
                continue
            if wc_only:
                lid = ((r.get("league") or {}).get("id"))
                try:
                    if lid is not None and int(lid) != wc_league_id:
                        continue
                except (TypeError, ValueError):
                    pass
            out.append(self._parse_match_row(r))
        return out

    def _real_events(self, match_id: str) -> List[LiveEvent]:
        """
        Pull all in-match events for one fixture. Verified endpoint:
            GET /v2.2/fixtures/?t=match_events&id=<fixture_id>

        Returns events including: goals, shots, cards, substitutions,
        period markers. We map the rich type vocabulary down to our
        LiveEventKind enum where possible.
        """
        import requests

        params = self._auth_params()
        params["t"] = "match_events"
        params["id"] = match_id
        url = f"{SOCCERSAPI_BASE}/fixtures/"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data") or []
        if not isinstance(rows, list):
            rows = [rows]
        parsed: List[LiveEvent] = []
        for r in rows:
            if not r:
                continue
            ev = self._parse_event_row(match_id, r)
            if ev is not None:
                parsed.append(ev)
        return parsed

    @staticmethod
    def _parse_match_row(row: Dict[str, Any]) -> LiveMatchState:
        """
        Parse a row from /livescores. Real shape (verified 2026-06-12):
            {
              "id": 2517287,
              "status": 4, "status_name": "Inplay",
              "time": {"minute": 84, "datetime": "..."},
              "teams": {
                "home": {"id": 772, "name": "Korea Republic",
                         "short_code": "KOR"},
                "away": {"id": 798, "name": "Czechia",
                         "short_code": "CZE"}
              },
              "scores": {"home_score": "2", "away_score": "1",
                         "ht_score": "1-0", "ft_score": null}
            }
        """
        teams = row.get("teams") or {}
        home = teams.get("home") or row.get("home") or {}
        away = teams.get("away") or row.get("away") or {}
        scores = row.get("scores") or row.get("score") or {}
        time_obj = row.get("time") or {}

        try:
            minute = int(time_obj.get("minute") or row.get("minute") or 0)
        except (TypeError, ValueError):
            minute = 0

        def _g(side_key: str) -> int:
            v = scores.get(f"{side_key}_score")
            if v is None:
                v = scores.get(side_key)
            try:
                return int(v) if v is not None else 0
            except (TypeError, ValueError):
                return 0

        # status_name is the human label ("Inplay", "Finished", "HT", ...)
        # — much more useful than the numeric "status" code.
        status = (
            str(row.get("status_name") or row.get("status") or "scheduled")
            .lower().replace(" ", "_")
        )

        return LiveMatchState(
            match_id=str(row.get("id") or row.get("match_id") or ""),
            status=status,
            minute=minute,
            home_goals=_g("home"),
            away_goals=_g("away"),
            home_team=str(home.get("name") or home.get("short_code") or ""),
            away_team=str(away.get("name") or away.get("short_code") or ""),
        )

    @staticmethod
    def _parse_event_row(match_id: str, row: Dict[str, Any]) -> Optional["LiveEvent"]:
        """
        Map SoccersAPI's verbose event type vocabulary down to our enum.

        Real types seen in the API include: goal, own_goal, penalty,
        yellow_card, red_card, substitution, shot_on_target,
        shot_off_target, corner, foul, offside, ... We only surface
        the events that meaningfully affect the score / consensus.
        Anything else returns None so the caller can skip it.
        """
        kind_map = {
            "goal": LiveEventKind.GOAL,
            "own_goal": LiveEventKind.OWN_GOAL,
            "penalty": LiveEventKind.PENALTY,
            "penalty_goal": LiveEventKind.GOAL,
            "penalty_missed": LiveEventKind.PENALTY,
            "yellow_card": LiveEventKind.YELLOW_CARD,
            "yellowcard": LiveEventKind.YELLOW_CARD,
            "red_card": LiveEventKind.RED_CARD,
            "redcard": LiveEventKind.RED_CARD,
            "yellowred_card": LiveEventKind.RED_CARD,
            "substitution": LiveEventKind.SUBSTITUTION,
            "ht": LiveEventKind.HALF_TIME,
            "ft": LiveEventKind.FULL_TIME,
            "ko": LiveEventKind.KICK_OFF,
            "kick_off": LiveEventKind.KICK_OFF,
        }
        raw_kind = str(row.get("type") or row.get("event") or "").lower().replace("-", "_").replace(" ", "_")
        kind = kind_map.get(raw_kind)
        if kind is None:
            # shot_on_target / shot_off_target / corner / foul etc.
            # - high-resolution stats we currently don't act on
            return None
        # SoccersAPI events don't have a stable event_id; synthesise one
        # from match_id + minute + raw_kind so de-dup still works.
        event_id = str(
            row.get("id")
            or row.get("event_id")
            or f"{match_id}-{row.get('minute', 0)}-{raw_kind}"
        )
        try:
            minute = int(row.get("minute") or 0)
        except (TypeError, ValueError):
            minute = 0
        return LiveEvent(
            match_id=match_id,
            event_id=event_id,
            kind=kind,
            minute=minute,
            team_fifa_code=row.get("team_code") or row.get("team_short") or str(row.get("team_id") or "") or None,
            player_name=row.get("player_name") or row.get("player") or None,
            detail=str(row.get("info") or row.get("reason") or row.get("detail") or ""),
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
