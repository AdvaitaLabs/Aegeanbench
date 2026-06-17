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
        # Each entry is (stored_at, expires_at, payload). expires_at lets
        # pre-warmed pre-match predictions live far longer than the 60s
        # default while live refreshes keep a short shelf life.
        self._store: "OrderedDict[str, Tuple[float, float, Dict[str, Any]]]" = OrderedDict()
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
            _stored_at, expires_at, _lang, payload = entry
            if time.time() > expires_at:
                self._store.pop(key, None)
                self.misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self.hits += 1
            return payload

    def put(
        self,
        key: str,
        match_id: str,
        payload: Dict[str, Any],
        ttl: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> None:
        """
        Store a prediction. ttl overrides the default shelf life for this
        entry only — pre-match warm writes pass a long ttl (e.g. an hour)
        so they survive until kickoff, live refreshes pass a short one.

        lang is recorded so get_latest_for_match can reuse the right
        language's prediction (a zh viewer must not be served en text).
        """
        now = time.time()
        effective_ttl = self.ttl if ttl is None else ttl
        with self._lock:
            self._store[key] = (now, now + effective_ttl, lang, payload)
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

    def get_latest_for_match(
        self,
        match_id: str,
        lang: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the most-recently-stored, non-expired prediction for a
        match. When lang is given, only entries computed in that language
        are considered.

        Used by:
          - the live path, so every viewer reuses the single prediction
            the background ticker computed instead of each running their
            own consensus;
          - the chat debounce, so a fresh burst of chatter serves the
            last real prediction instead of recomputing — refreshes are
            driven by the schedulers (pre-match checkpoints / live ticker),
            not by user chat.

        max_age_seconds, when set, additionally requires the entry to be
        younger than that (independent of its TTL).
        """
        now = time.time()
        with self._lock:
            keys = self._match_keys.get(match_id, set())
            best: Optional[Tuple[float, Dict[str, Any]]] = None
            for key in list(keys):
                entry = self._store.get(key)
                if entry is None:
                    continue
                stored_at, expires_at, entry_lang, payload = entry
                if now > expires_at:
                    continue
                if lang is not None and entry_lang is not None and entry_lang != lang:
                    continue
                if max_age_seconds is not None and now - stored_at > max_age_seconds:
                    continue
                if best is None or stored_at > best[0]:
                    best = (stored_at, payload)
            return best[1] if best else None

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
