"""
Runtime-editable global prompt store.

One product-tunable string that gets appended to every agent's system
prompt across /answer, /predict, and /divination. Stored as a JSON file
so it persists across restarts; reads are cached in-process for 30s.

History: every POST keeps the previous 5 versions so a bad edit can be
rolled back without engineer help.

File layout (~/.aegeanbench/runtime_prompt.json):
    {
        "current": {
            "prompt": "...",
            "updated_at": "2026-06-10T08:00:00Z",
            "updated_by": "product@advaita.xyz"
        },
        "history": [
            {"prompt": "...", "updated_at": "...", "updated_by": "..."},
            ...
        ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_DEFAULT_DIR = Path(os.path.expanduser("~/.aegeanbench/prompts"))
_DEFAULT_PATH = _DEFAULT_DIR / "runtime_prompt.json"
_MAX_HISTORY = 5
_CACHE_TTL = 30.0

_lock = threading.Lock()
_cache: Dict[str, Any] = {"loaded_at": 0.0, "data": None}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path() -> Path:
    return Path(os.getenv("AEGEANBENCH_RUNTIME_PROMPT_PATH", str(_DEFAULT_PATH)))


def _empty() -> Dict[str, Any]:
    return {"current": None, "history": []}


def _load_from_disk() -> Dict[str, Any]:
    p = _path()
    if not p.exists():
        return _empty()
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f) or _empty()
    except Exception as e:
        logger.warning("runtime_prompt store read failed (%s); using empty", e)
        return _empty()


def _save_to_disk(data: Dict[str, Any]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def _get_data(force_reload: bool = False) -> Dict[str, Any]:
    with _lock:
        now = time.time()
        if (not force_reload
                and _cache["data"] is not None
                and now - _cache["loaded_at"] < _CACHE_TTL):
            return _cache["data"]
        data = _load_from_disk()
        _cache["data"] = data
        _cache["loaded_at"] = now
        return data


# ----------------------------- public API -----------------------------


def get_current_prompt() -> str:
    """Return the active global prompt addendum, '' when none configured."""
    data = _get_data()
    cur = (data or {}).get("current")
    return (cur or {}).get("prompt", "") if cur else ""


def get_full_state() -> Dict[str, Any]:
    """Return current + history for the admin GET endpoint."""
    return _get_data(force_reload=True)


def set_prompt(prompt: str, updated_by: Optional[str] = None) -> Dict[str, Any]:
    """
    Replace the current prompt; the previous one is pushed onto history.
    Returns the new state.
    """
    with _lock:
        data = _load_from_disk()
        if data.get("current"):
            history: List[Dict[str, Any]] = data.get("history") or []
            history.insert(0, data["current"])
            data["history"] = history[:_MAX_HISTORY]
        data["current"] = {
            "prompt": prompt,
            "updated_at": _now_iso(),
            "updated_by": updated_by or "unknown",
        }
        _save_to_disk(data)
        _cache["data"] = data
        _cache["loaded_at"] = time.time()
        return data


def rollback_to_previous() -> Dict[str, Any]:
    """Restore the most-recent history entry as current. No-op if empty."""
    with _lock:
        data = _load_from_disk()
        history: List[Dict[str, Any]] = data.get("history") or []
        if not history:
            return data
        latest = history.pop(0)
        # The current one drops into history (so rollback is undoable)
        if data.get("current"):
            history.insert(0, data["current"])
            history = history[:_MAX_HISTORY]
        data["current"] = latest
        data["history"] = history
        _save_to_disk(data)
        _cache["data"] = data
        _cache["loaded_at"] = time.time()
        return data


def invalidate_cache() -> None:
    """Force the next get_current_prompt() to re-read from disk."""
    with _lock:
        _cache["data"] = None
        _cache["loaded_at"] = 0.0
