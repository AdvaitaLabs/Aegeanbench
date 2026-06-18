"""
Tournament-forecast warmer: regenerate every model's full forecast
  * once a day, and
  * whenever a match-day completes (new real results appear).

Both triggers are detected from the shared real state: we track the
number of completed (actual) matches and the timestamp of the last
regeneration. A change in completed count => a match-day finished =>
regenerate; otherwise regenerate once the daily interval elapses.

Disable with TOURNAMENT_WARMER_DISABLED=1.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_CHECK_INTERVAL_SECONDS = 30 * 60.0     # check every 30 min
_DAILY_SECONDS = 24 * 3600.0


class TournamentWarmer:
    def __init__(self, tournament_service, lang: str = "en",
                 interval_seconds: float = _CHECK_INTERVAL_SECONDS):
        self.svc = tournament_service
        self.lang = lang
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._last_regen_ts: Optional[float] = None
        self._last_completed: int = -1

    def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run_forever(), name="tournament-warmer")
        logger.warning("TournamentWarmer started (interval=%.0fs)", self.interval)

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
                logger.warning("TournamentWarmer tick failed: %s", e)
            await asyncio.sleep(self.interval)

    @staticmethod
    def _now() -> float:
        return datetime.now(timezone.utc).timestamp()

    async def _tick(self) -> None:
        state = await asyncio.to_thread(self.svc._real_state)
        completed = sum(
            1 for g in state.get("groups", []) for m in g.get("matches", []) if m.get("actual")
        )
        now = self._now()
        daily_due = self._last_regen_ts is None or (now - self._last_regen_ts) >= _DAILY_SECONDS
        matchday_done = completed != self._last_completed and self._last_completed >= 0

        if not (daily_due or matchday_done):
            return
        reason = "match-day completed" if matchday_done else "daily"
        logger.warning("TournamentWarmer regenerating all models (%s, %d completed matches)",
                       reason, completed)
        await asyncio.to_thread(self.svc.compute_all, self.lang, False)
        self._last_regen_ts = now
        self._last_completed = completed
