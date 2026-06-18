"""
Arena background warmer.

Walks upcoming fixtures once a minute and, at the T-24h and T-1h pre-match
checkpoints, computes the FULL arena comparison (aegean-consensus + every
benchmark) and caches it — so /api/v1/arena/upcoming shows real
predictions instead of "pending", instantly and without re-spending
tokens per request.

Checkpoint rationale:
  * T-24h: populate the upcoming list a day ahead.
  * T-1h : refresh once lineups are out — this is when aegean-consensus
           (data-grounded) gains the most. Benchmarks barely change in a
           day, so re-running them here is mild waste; a future
           optimisation can refresh only aegean at T-1h.

Uses its OWN MatchEventScheduler so its once-per-checkpoint bookkeeping
does not collide with the PrematchWarmer's. Disable with
ARENA_WARMER_DISABLED=1.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_TICK_INTERVAL_SECONDS = 60.0
_COMPETITION = "FIFA World Cup 2026"


class ArenaWarmer:
    def __init__(
        self,
        arena_service,
        scheduler,
        lang: str = "en",
        interval_seconds: float = _TICK_INTERVAL_SECONDS,
    ):
        self.svc = arena_service
        self.scheduler = scheduler
        self.lang = lang
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run_forever(), name="arena-warmer")
        logger.warning("ArenaWarmer started (interval=%.0fs)", self.interval)

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
                logger.warning("ArenaWarmer tick failed: %s", e)
            await asyncio.sleep(self.interval)

    async def _tick(self) -> None:
        gw = self.svc.gateway
        if gw is None:
            return
        try:
            fixtures = await asyncio.to_thread(gw.list_fixtures, _COMPETITION)
        except Exception as e:  # noqa: BLE001
            logger.warning("ArenaWarmer fixture fetch failed: %s", e)
            return

        now = datetime.now(timezone.utc)
        for match in fixtures or []:
            kickoff = getattr(match, "kickoff_at", None)
            if kickoff is None:
                continue
            trigger = self.scheduler.on_clock_tick(match.match_id, kickoff, now)
            if trigger is None:
                continue
            await self._warm(match, trigger)

    async def _warm(self, match, trigger) -> None:
        kickoff = getattr(match, "kickoff_at", None)
        kickoff_iso = kickoff.isoformat() if kickoff is not None else None
        home = getattr(getattr(match, "home_team", None), "name", None)
        away = getattr(getattr(match, "away_team", None), "name", None)
        # Heavy (N models incl. the consensus) — run off the event loop.
        try:
            payload = await asyncio.to_thread(
                self.svc.compute_match,
                match_id=match.match_id, home_team=home, away_team=away,
                kickoff_iso=kickoff_iso, venue=getattr(match, "venue", None),
                lang=self.lang, use_cache=False,
            )
            n_ok = sum(1 for m in payload.get("models", []) if m.get("status") == "ok")
            logger.info("arena warm %s (%s): %d models ok", match.match_id, trigger.detail, n_ok)
        except Exception as e:  # noqa: BLE001
            logger.warning("arena warm failed for %s: %s", match.match_id, e)
