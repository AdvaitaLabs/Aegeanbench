"""
In-process pub/sub fan-out for WebSocket push.

Sprint scope: a single process serves all WebSocket clients. The hub
holds an asyncio.Queue per subscriber per channel and publish() does
non-blocking fan-out. Slow clients are dropped (queue full).

Scales to a few thousand connections per process. For larger fan-out
we would swap this for Redis pub/sub or NATS - but that is V3 work,
and our public API stays the same so it's a backend swap.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


class LiveHub:
    """
    Per-channel subscriber list with non-blocking publish.

    Channels are arbitrary strings. Convention used by the server:
        "predictions"           global prediction updates
        "match:<match_id>"      per-match score + re-evaluation
    """

    def __init__(self, queue_max_size: int = 64):
        self._channels: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._queue_max_size = queue_max_size
        self._lock = asyncio.Lock()

    async def subscribe(self, channel: str) -> asyncio.Queue:
        """Register a new subscriber for the channel and return its queue."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_max_size)
        async with self._lock:
            self._channels[channel].add(queue)
        return queue

    async def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        """Remove the subscriber. Safe to call even if already gone."""
        async with self._lock:
            subs = self._channels.get(channel)
            if subs and queue in subs:
                subs.remove(queue)
                if not subs:
                    del self._channels[channel]

    async def publish(self, channel: str, payload: Any) -> int:
        """
        Fan out a payload to every subscriber on the channel.

        Slow subscribers whose queue is full are silently dropped from the
        delivery (their queue remains so they can reconnect logically),
        following the "newest matters more than oldest" policy that suits
        score/prediction streams.

        Returns:
            number of subscribers the payload was delivered to.
        """
        delivered = 0
        async with self._lock:
            subs = list(self._channels.get(channel, ()))
        for queue in subs:
            try:
                queue.put_nowait(payload)
                delivered += 1
            except asyncio.QueueFull:
                # Drop the oldest message and try once more
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                    delivered += 1
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    logger.warning("dropping payload for full queue on channel %s", channel)
        return delivered

    def subscriber_count(self, channel: str = None) -> int:
        """Count subscribers, optionally filtered to one channel."""
        if channel is not None:
            return len(self._channels.get(channel, ()))
        return sum(len(subs) for subs in self._channels.values())

    def channels(self) -> List[str]:
        """Return active channel names."""
        return list(self._channels.keys())
