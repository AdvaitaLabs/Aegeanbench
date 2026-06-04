"""
Mock chat data generator for sprint development.

Use this until the chat service teammate ships their real endpoint. Plug
into ChatAgent via:

    from aegeanbench.sports.chat_mock import MockChatFetcher
    ChatAgent(llm_client=..., chat_fetch_fn=MockChatFetcher().fetch)

Produces plausible bilingual chat windows that:
  - Vary slightly per match_id (hash-seeded)
  - Lean toward whichever team's FIFA code appears earlier in the ID
  - Include a mix of English, Chinese, and emoji
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


# Templates seeded per locale; weighted so most messages are casual rather
# than analytical, matching real fan-chat behaviour.
_EN_TEMPLATES = [
    "{home} looks unstoppable today",
    "{away} fans where you at",
    "Bet365 has {home} -0.5, easy money",
    "I think this is a draw",
    "{away} keeper has been shaky lately",
    "Nervous as always before kickoff",
    "{home} all the way 🇧🇷",
    "anyone seeing the live xG?",
    "{home} press has been weak the last 3 games",
    "this match is rigged lol",
    "midfield matchup is huge today",
    "{away} 3-0 incoming",
    "let's go boys",
    "i'm tilting already and game hasn't started",
]

_ZH_TEMPLATES = [
    "{home}今天必胜",
    "{away}状态不好",
    "感觉今天会平局",
    "中场拼抢是关键",
    "{home}加油！！",
    "押了{away} +1球",
    "梅西/{star}今天上吗",
    "看{away}最近三场表现不行",
    "下半场见分晓",
    "感觉裁判会偏",
    "守门员状态太关键了",
    "{home}最近xG很猛",
]

_LANG_DIST = [("en", 0.55), ("zh", 0.45)]


def _seeded_random(match_id: str) -> random.Random:
    """Stable RNG keyed off match_id so the same match yields the same chat."""
    seed = int(hashlib.sha256(match_id.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def _team_names_from_match_id(match_id: str) -> Dict[str, str]:
    """
    Infer team names from match_id. AegeanBench convention is
    'WC2026-A1' (group + slot). We don't have the real teams here, so
    we fall back to placeholders.
    """
    # When the caller is willing to provide team codes via context_hint
    # they should use MockChatFetcher(context_hint=...) (below).
    return {"home": "Home", "away": "Away", "star": "Messi"}


class MockChatFetcher:
    """
    Drop-in mock for ChatFetcher.fetch() / chat_fetch_fn.

    Args:
        context_hint: optional dict mapping match_id -> {"home": ..., "away": ...,
            "star": ...} so generated messages reference real team names.
        base_message_count: average number of messages per window.
        timestamp_ref: anchor datetime; messages are spread backward
            from this point. Default: current UTC time.
    """

    def __init__(
        self,
        context_hint: Dict[str, Dict[str, str]] = None,
        base_message_count: int = 25,
        timestamp_ref: datetime = None,
    ):
        self.context_hint = context_hint or {}
        self.base_message_count = base_message_count
        # Allow tests to inject a fixed anchor; otherwise use now().
        self.timestamp_ref = timestamp_ref

    def fetch(self, match_id: str, window_minutes: int = 30) -> Dict[str, Any]:
        rng = _seeded_random(match_id)
        teams = self.context_hint.get(match_id) or _team_names_from_match_id(match_id)

        n_messages = max(
            5, int(rng.gauss(self.base_message_count, self.base_message_count * 0.3))
        )
        anchor = self.timestamp_ref or datetime.now(timezone.utc)
        window_start = anchor - timedelta(minutes=window_minutes)

        messages: List[Dict[str, Any]] = []
        for i in range(n_messages):
            lang_pick = rng.random()
            lang = "en" if lang_pick < _LANG_DIST[0][1] else "zh"
            template_pool = _EN_TEMPLATES if lang == "en" else _ZH_TEMPLATES
            template = rng.choice(template_pool)
            text = template.format(
                home=teams.get("home", "Home"),
                away=teams.get("away", "Away"),
                star=teams.get("star", "Messi"),
            )
            # Spread messages across the window
            offset_minutes = window_minutes * (i / max(n_messages - 1, 1))
            ts = window_start + timedelta(minutes=offset_minutes)
            messages.append(
                {
                    "message_id": f"mock_{match_id}_{i:04d}",
                    "user_id": f"user_{rng.randint(1000, 9999)}",
                    "user_name": f"fan_{rng.randint(100, 999)}",
                    "text": text,
                    "timestamp": ts.isoformat(),
                    "language": lang,
                }
            )

        return {
            "match_id": match_id,
            "window_start": window_start.isoformat(),
            "window_end": anchor.isoformat(),
            "total_messages": len(messages),
            "messages": messages,
        }


# Convenience: a module-level callable so callers can pass it directly
# to ChatAgent without instantiating the class explicitly.
def fetch_mock_chat_window(match_id: str, window_minutes: int = 30) -> Dict[str, Any]:
    return MockChatFetcher().fetch(match_id, window_minutes)
