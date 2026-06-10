"""
YAML-backed prompt template loader.

All user-facing LLM prompts (predict / qa / divination) live in
templates.yaml so the product team can tweak wording without touching
Python code. Edits take effect on the next `docker compose restart
aegeanbench`.

Usage:
    from aegeanbench.sports.prompts.loader import get_template
    text = get_template("predict.system_en")
    rendered = get_template("qa.user_template").format(
        context_section="...",
        chat_section="",
        who="alice: ",
        question="who wins?",
    )

Falls back to safe defaults if templates.yaml is missing or malformed
so the service stays up.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


_TEMPLATES_PATH = Path(__file__).parent / "templates.yaml"
_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_loaded = False


def _load() -> Dict[str, Any]:
    """Load and cache the YAML once per process."""
    global _loaded
    if _loaded:
        return _cache
    with _lock:
        if _loaded:
            return _cache
        try:
            import yaml
            with _TEMPLATES_PATH.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _cache.update(data)
            logger.info("Loaded prompt templates from %s", _TEMPLATES_PATH)
        except FileNotFoundError:
            logger.error("templates.yaml not found at %s; using empty fallback", _TEMPLATES_PATH)
        except Exception as e:
            logger.error("failed to parse templates.yaml: %s", e)
        _loaded = True
    return _cache


def get_template(dotted_key: str, default: str = "") -> str:
    """
    Return a template by dotted path, e.g. "predict.system_en".

    Returns `default` when the path is missing so callers can supply a
    hard-coded fallback during edits.
    """
    data = _load()
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(part)
        if cur is None:
            return default
    return str(cur)


def reload() -> None:
    """Force a reload — useful in tests after editing templates.yaml."""
    global _loaded
    with _lock:
        _cache.clear()
        _loaded = False
