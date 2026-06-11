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
            return

        if not live_matches:
            return

        for m in live_matches:
            try:
                await self._handle_match(client, m)
            except Exception as e:  # noqa: BLE001
                logger.warning("live tick failed for %s: %s", m.match_id, e)

    async def _handle_match(self, client, match) -> None:
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
                "team": ev.team,
                "player": ev.player,
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
