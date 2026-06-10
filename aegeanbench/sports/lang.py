"""
Tiny language detector for routing replies between Chinese and English.

The product rule is simple: if the user's question / chat messages
contain CJK characters, the agent should answer in Chinese; otherwise
English. We don't try to handle other languages — only zh vs en.

Used by /answer, /predict, /divination to inject a language directive
into the system prompt.
"""

from __future__ import annotations

from typing import Iterable, Optional


# CJK Unified Ideographs (CJK / extension-A) cover modern Chinese.
def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def detect_lang(text: Optional[str]) -> str:
    """Return 'zh' if the text contains CJK chars, else 'en'."""
    if not text:
        return "en"
    return "zh" if _has_cjk(text) else "en"


def detect_from_signals(
    primary: Optional[str] = None,
    secondary: Iterable[str] = (),
) -> str:
    """
    Decide the response language from a primary signal (the user's own
    question) plus optional secondary signals (recent chat messages).
    Primary wins; secondary is only consulted when primary is empty.
    """
    if primary:
        return detect_lang(primary)
    for s in secondary:
        if s:
            return detect_lang(s)
    return "en"


def lang_directive(lang: str) -> str:
    """Single-line directive to drop into a system prompt."""
    if lang == "zh":
        return (
            "Answer in Simplified Chinese (简体中文). Keep tone professional. "
            "Do not mix English sentences unless quoting team / bookmaker names."
        )
    return (
        "Answer in English. Keep tone professional. Do not insert Chinese "
        "characters unless quoting a proper noun verbatim."
    )
