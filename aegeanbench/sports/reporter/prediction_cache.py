"""
In-process prediction cache for /api/v1/predict.

A single /predict call runs 8 agents × 2-3 rounds × Praka and takes
20-30 s. The cache short-circuits identical requests within a TTL
window so the front-end (or multiple chat users on the same table)
doesn't pay that latency repeatedly.

Cache key: (match_id, agent_ids_sorted_tuple, lang, chat_signature)
  - chat_signature is a hash of the last 10 chat lines so a fresh
    burst of group-chat opinions invalidates the cache naturally.

A live-event hook (see invalidate_match) lets the scheduler flush all
entries for a match when a goal/red card lands, so the next /predict
re-runs against fresh state.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_TTL_SECONDS = 60.0          # Cached predictions stay fresh for 1 minute
_MAX_ENTRIES = 256           # LRU eviction beyond this


class PredictionCache:
    """Thread-safe TTL + LRU cache keyed by request fingerprint."""

    def __init__(self, ttl_seconds: float = _TTL_SECONDS, max_entries: int = _MAX_ENTRIES):
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, Tuple[float, Dict[str, Any]]]" = OrderedDict()
        self._match_keys: Dict[str, set] = {}   # match_id -> set of keys
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _chat_signature(chat_messages: Optional[List[Any]]) -> str:
        if not chat_messages:
            return "no-chat"
        # Hash only the last 10 lines' text - that's what the predictor
        # actually feeds into the prompt.
        texts = []
        for m in chat_messages[-10:]:
            t = m.text if hasattr(m, "text") else (m.get("text") if isinstance(m, dict) else str(m))
            texts.append(str(t))
        digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
        return digest

    def make_key(
        self,
        match_id: str,
        agent_ids: List[str],
        lang: str,
        chat_messages: Optional[List[Any]] = None,
    ) -> str:
        agents_part = ",".join(sorted(agent_ids))
        chat_part = self._chat_signature(chat_messages)
        return f"{match_id}|{agents_part}|{lang}|{chat_part}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            ts, payload = entry
            if time.time() - ts > self.ttl:
                self._store.pop(key, None)
                self.misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self.hits += 1
            return payload

    def put(self, key: str, match_id: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = (time.time(), payload)
            self._store.move_to_end(key)
            self._match_keys.setdefault(match_id, set()).add(key)
            # LRU eviction
            while len(self._store) > self.max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                # Drop from match index too
                for mid, keys in list(self._match_keys.items()):
                    keys.discard(evicted_key)
                    if not keys:
                        self._match_keys.pop(mid, None)

    def invalidate_match(self, match_id: str) -> int:
        """
        Drop every cached prediction for a match. Called by the live-event
        scheduler when a goal/red card lands so the next /predict for
        this match re-runs against fresh state. Returns count dropped.
        """
        with self._lock:
            keys = self._match_keys.pop(match_id, set())
            for key in keys:
                self._store.pop(key, None)
            if keys:
                logger.info("invalidated %d cached predictions for %s",
                            len(keys), match_id)
            return len(keys)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._store),
                "matches_tracked": len(self._match_keys),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / max(self.hits + self.misses, 1), 3),
                "ttl_seconds": self.ttl,
                "max_entries": self.max_entries,
            }
