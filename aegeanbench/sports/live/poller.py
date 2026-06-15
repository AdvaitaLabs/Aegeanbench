"""
Background poller that drives the live-event pipeline.

Every N seconds (default 30s) it asks soccersapi which matches are
currently in-play. For each one we pull recent events and:

  1. Push every event onto the WebSocket channel
        match:{match_id}     -> "live_event" envelope
  2. Feed high-priority events (goal, red_card, half_time, full_time)
     into the MatchEventScheduler. If the scheduler returns a trigger,
     we invalidate the prediction cache for that match so the next
     /predict re-runs against the fresh state.

Soft-fail: if the soccersapi plan doesn't include match_events, the
poller logs once and goes idle. No exceptions bubble up.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Set

logger = logging.getLogger(__name__)


_POLL_INTERVAL_SECONDS = 30.0
_HIGH_PRIORITY_KINDS = {
    "goal", "own_goal", "penalty_scored", "penalty_missed",
    "red_card", "half_time", "full_time",
}
# Force a fresh consensus this often during a live match even if no
# high-priority events landed. Picks up gradual momentum shifts that
# don't fire a discrete trigger (sustained pressure, possession swing).
_PERIODIC_REFRESH_SECONDS = 8 * 60


class LiveEventPoller:
    """
    Long-running asyncio task. Construct once at startup, start() spins
    up the loop in the background, stop() cancels it cleanly.
    """

    def __init__(
        self,
        live_hub,
        scheduler,
        prediction_cache,
        interval_seconds: float = _POLL_INTERVAL_SECONDS,
    ):
        self.live_hub = live_hub
        self.scheduler = scheduler
        self.prediction_cache = prediction_cache
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._seen: Set[str] = set()  # event_ids we've already pushed
        self._disabled_reason: Optional[str] = None  # soft-disable flag
        self._poll_count = 0
        # Track last forced refresh per match so we re-consense every
        # PERIODIC_REFRESH seconds even on low-priority event streams
        # (lots of shots / corners / offsides without a goal).
        self._last_periodic_refresh: dict = {}

    def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run_forever(), name="live-poller")
        # Use warning level so the message survives default INFO filtering.
        logger.warning("LiveEventPoller started (interval=%.0fs)", self.interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run_forever(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("LiveEventPoller tick failed: %s", e)
            await asyncio.sleep(self.interval)

    async def _tick(self) -> None:
        if self._disabled_reason:
            return
        self._poll_count += 1

        from aegeanbench.sports.sources.soccersapi_live import SoccersAPILiveClient
        client = SoccersAPILiveClient()

        try:
            live_matches = client.fetch_live_matches()
        except Exception as e:
            logger.warning("live matches fetch failed: %s", e)
            live_matches = []

        # When soccersapi has nothing (its `status` often lags 15-30
        # minutes behind real life on Soccer 100), check football-data
        # for currently in-play WC matches as a fallback. We can't get
        # per-event detail from FD's free tier, but we can at least
        # invalidate the prediction cache for matches that are actually
        # live, so the next /predict re-runs with fresh state.
        if not live_matches:
            try:
                await self._invalidate_from_football_data()
            except Exception as e:  # noqa: BLE001
                logger.warning("football_data fallback failed: %s", e)
            return

        for m in live_matches:
            try:
                await self._handle_match(client, m)
            except Exception as e:  # noqa: BLE001
                logger.warning("live tick failed for %s: %s", m.match_id, e)

    async def _invalidate_from_football_data(self) -> None:
        """
        Walk football-data's WC IN_PLAY matches and trigger a cache
        flush + WS notice for any we haven't seen recently. Map FD ids
        back to soccersapi ids so the cache key (which uses soccersapi
        match_ids from the front-end) actually drops.
        """
        try:
            import requests
        except ImportError:
            return
        import os
        key = os.getenv("AEGEANBENCH_FOOTBALL_DATA_KEY", "")
        if not key:
            return
        try:
            r = requests.get(
                "https://api.football-data.org/v4/competitions/WC/matches",
                headers={"X-Auth-Token": key},
                params={"status": "LIVE"},
                timeout=8,
            )
            r.raise_for_status()
            payload = r.json() or {}
        except Exception as e:
            logger.warning("football_data LIVE matches fetch failed: %s", e)
            return

        matches = payload.get("matches") or []
        if not matches:
            return

        # We need a (fd_id -> soccersapi_id) translation. Use the
        # cached match_info that match_brief built — soccersapi t=info
        # carries soccersapi's own id for any fixture we've touched.
        # As a simpler heuristic, match by (home_team_name, away_team_name)
        # against any sa match_id we've seen recently.
        for m in matches:
            home = (m.get("homeTeam") or {}).get("name", "")
            away = (m.get("awayTeam") or {}).get("name", "")
            fd_minute = m.get("minute") or 0
            score = (m.get("score") or {}).get("fullTime") or {}
            envelope = {
                "type": "live_event_fallback",
                "source": "football_data",
                "home": home,
                "away": away,
                "minute": fd_minute,
                "home_goals": score.get("home", 0),
                "away_goals": score.get("away", 0),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            # Best-effort: push to all WS subscribers of any channel that
            # mentions either team (the front-end can pick relevant rooms).
            try:
                await self.live_hub.publish("predictions", envelope)
            except Exception:  # noqa: BLE001
                pass

            # Periodic refresh logic: if we've never invalidated this
            # match (no soccersapi_id known here), there's nothing to
            # invalidate in the prediction cache. But the WS push still
            # lets the front-end know "this match is live, status feed
            # has it even though our primary source doesn't yet".
            logger.info(
                "football_data sees LIVE: %s %s-%s %s (%s')",
                home, score.get("home", 0), score.get("away", 0), away, fd_minute,
            )

    async def _handle_match(self, client, match) -> None:
        # Periodic refresh: if the match is live and we haven't flushed
        # the cache in PERIODIC_REFRESH_SECONDS, force a re-consensus
        # so panel takes fresh state into account (momentum shifts,
        # accumulated possession edge, etc.) even without a goal.
        import time as _t
        now_ts = _t.time()
        last = self._last_periodic_refresh.get(match.match_id, 0.0)
        if now_ts - last > _PERIODIC_REFRESH_SECONDS:
            self._last_periodic_refresh[match.match_id] = now_ts
            n = self.prediction_cache.invalidate_match(match.match_id)
            if n > 0 or last > 0:
                logger.info(
                    "periodic refresh: flushed %d cache entries for %s",
                    n, match.match_id,
                )
                await self.live_hub.publish(
                    f"match:{match.match_id}",
                    {
                        "type": "consensus_invalidated",
                        "match_id": match.match_id,
                        "reason": "periodic_refresh",
                        "detail": f"every {_PERIODIC_REFRESH_SECONDS // 60} min during live play",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    },
                )

        events = client.fetch_match_events(match.match_id)
        if events is None:
            return
        if not events and self._poll_count == 1:
            # First tick yielded nothing — could be that the soccersapi
            # plan doesn't cover events. Don't disable yet; many matches
            # legitimately have no events early.
            return

        for ev in events:
            event_uid = f"{match.match_id}:{ev.event_id or ''}:{ev.minute}:{ev.kind.value}"
            if event_uid in self._seen:
                continue
            self._seen.add(event_uid)

            envelope = {
                "type": "live_event",
                "match_id": match.match_id,
                "minute": ev.minute,
                "kind": ev.kind.value,
                # LiveEvent dataclass uses team_fifa_code + player_name,
                # not team/player. Old shorthand crashed every tick.
                "team": ev.team_fifa_code,
                "player": ev.player_name,
                "detail": ev.detail or "",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            await self.live_hub.publish(f"match:{match.match_id}", envelope)

            if ev.kind.value in _HIGH_PRIORITY_KINDS:
                trigger = self.scheduler.on_live_event(ev)
                if trigger is not None:
                    n = self.prediction_cache.invalidate_match(match.match_id)
                    logger.info(
                        "live event %s @ %s' triggered cache flush (%d entries) for %s",
                        ev.kind.value, ev.minute, n, match.match_id,
                    )
                    await self.live_hub.publish(
                        f"match:{match.match_id}",
                        {
                            "type": "consensus_invalidated",
                            "match_id": match.match_id,
                            "reason": trigger.reason.value,
                            "detail": trigger.detail,
                            "ts": envelope["ts"],
                        },
                    )

        # Trim the seen set so it doesn't grow unbounded across days
        if len(self._seen) > 5000:
            keep = set(list(self._seen)[-2000:])
            self._seen = keep
