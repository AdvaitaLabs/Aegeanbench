"""
Local filesystem cache for sports data sources.

Uses JSON for portability (parquet requires pyarrow; we keep deps minimal
for the sprint and can swap to parquet in v2 if size becomes an issue).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".aegeanbench" / "sports_cache"


class FileCache:
    """
    Simple keyed cache backed by JSON files on disk.

    Key derivation: stable hash of (adapter_name, method, *args, **kwargs).
    Value: JSON-serialized payload + metadata (timestamp, ttl).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        default_ttl: timedelta = timedelta(hours=24),
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl

    @staticmethod
    def _stable_key(*parts: Any) -> str:
        """Hash all parts into a stable filename-safe key."""
        joined = "|".join(repr(p) for p in parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, *parts: Any, ttl: Optional[timedelta] = None) -> Optional[Any]:
        """
        Read from cache. Returns None on miss or expired.

        Args:
            *parts: key components (adapter, method, args...)
            ttl: override default TTL
        """
        key = self._stable_key(*parts)
        path = self._path(key)
        if not path.exists():
            return None

        try:
            with path.open() as f:
                envelope = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("cache read failed for %s: %s", key, e)
            return None

        ts = datetime.fromisoformat(envelope["cached_at"])
        effective_ttl = ttl or self.default_ttl
        if datetime.now() - ts > effective_ttl:
            return None  # expired

        return envelope["payload"]

    def set(self, payload: Any, *parts: Any) -> None:
        """Write to cache."""
        key = self._stable_key(*parts)
        path = self._path(key)
        envelope = {
            "cached_at": datetime.now().isoformat(),
            "key_parts": [repr(p) for p in parts],
            "payload": payload,
        }
        # Atomic write: write to tmp then rename
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            json.dump(envelope, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)

    def get_or_compute(
        self,
        compute_fn: Callable[[], Any],
        *parts: Any,
        ttl: Optional[timedelta] = None,
    ) -> Any:
        """
        Cache-aside pattern: return cached value or call compute_fn() and cache the result.

        Args:
            compute_fn: zero-arg callable that returns the value to cache
            *parts: key components
            ttl: override default TTL
        """
        cached = self.get(*parts, ttl=ttl)
        if cached is not None:
            logger.debug("cache hit: %s", parts[:2])
            return cached
        logger.debug("cache miss: %s", parts[:2])
        value = compute_fn()
        self.set(value, *parts)
        return value

    def clear(self) -> int:
        """Delete all cache entries. Returns count deleted."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count


# Singleton accessor for convenience
_default_cache: Optional[FileCache] = None


def get_default_cache() -> FileCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = FileCache()
    return _default_cache
