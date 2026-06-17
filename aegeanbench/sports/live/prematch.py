"""
Feature A: background pre-match consensus warmer.

A user opening a match panel shortly before kickoff should not wait the
~80s a fresh consensus costs. This long-running task walks the upcoming
fixtures once a minute and asks the MatchEventScheduler whether any has
crossed a pre-match checkpoint (T-24h / T-1h by default). When one does,
it pre-computes the consensus and stores it with a long cache TTL so the
next /predict for that match is an instant cache hit.

It shares the MatchEventScheduler with the live poller so the throttle
and "fire each checkpoint once" bookkeeping is consistent across both.

Soft-fail: any error in a tick is logged and swallowed; the loop keeps
running. Disabled entirely via PREMATCH_WARMER_DISABLED=1.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


_TICK_INTERVAL_SECONDS = 60.0
# Warmed predictions must survive from the checkpoint until kickoff. The
# T-24h snapshot needs to outlive the whole pre-match day; the newer T-1h
# refresh is stored afterwards so get_latest_for_match returns it for the
# final hour. 26h covers the longest (T-24h) gap with margin.
_WARM_TTL_SECONDS = 26 * 3600
_COMPETITION = "FIFA World Cup 2026"


class PrematchWarmer:
    """Long-running asyncio task; same start()/stop() shape as the poller."""

    def __init__(
        self,
        gateway,
        scheduler,
        prediction_cache,
        agent_ids: Optional[List[str]] = None,
        lang: str = "en",
        interval_seconds: float = _TICK_INTERVAL_SECONDS,
        warm_ttl_seconds: float = _WARM_TTL_SECONDS,
    ):
        self.gateway = gateway
        self.scheduler = scheduler
        self.prediction_cache = prediction_cache
        self.agent_ids = agent_ids
        self.lang = lang
        self.interval = interval_seconds
        self.warm_ttl = warm_ttl_seconds
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None:
            return
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run_forever(), name="prematch-warmer")
        logger.warning("PrematchWarmer started (interval=%.0fs)", self.interval)

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
                logger.warning("PrematchWarmer tick failed: %s", e)
            await asyncio.sleep(self.interval)

    async def _tick(self) -> None:
        try:
            fixtures = await asyncio.to_thread(
                self.gateway.list_fixtures, _COMPETITION
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("PrematchWarmer fixture fetch failed: %s", e)
            return

        now = datetime.now(timezone.utc)
        for match in fixtures or []:
            kickoff = getattr(match, "kickoff_at", None)
            if kickoff is None:
                continue
            trigger = self.scheduler.on_clock_tick(match.match_id, kickoff, now)
            if trigger is not None:
                await self._warm(match, trigger)

    async def _warm(self, match, trigger) -> None:
        agent_ids = self.agent_ids
        if not agent_ids:
            from aegeanbench.sports.predictors.aegean import DEFAULT_AGENT_TYPES
            agent_ids = list(DEFAULT_AGENT_TYPES)

        kickoff = getattr(match, "kickoff_at", None)
        kickoff_iso = kickoff.isoformat() if kickoff is not None else None
        home = getattr(getattr(match, "home_team", None), "name", None)
        away = getattr(getattr(match, "away_team", None), "name", None)

        from aegeanbench.sports.reporter import prediction_service as svc
        payload = await svc.run_and_cache(
            gateway=self.gateway,
            cache=self.prediction_cache,
            match_id=match.match_id,
            agent_ids=agent_ids,
            lang=self.lang,
            home_team=home,
            away_team=away,
            kickoff_iso=kickoff_iso,
            venue=getattr(match, "venue", None),
            cache_ttl=self.warm_ttl,
            source="prematch_warm",
        )
        mocked = payload is None or (payload.get("_meta") or {}).get("is_mock")
        logger.info(
            "prematch warm %s for %s (%s)",
            "skipped (mock/unresolved)" if mocked else "cached",
            match.match_id, trigger.detail,
        )
