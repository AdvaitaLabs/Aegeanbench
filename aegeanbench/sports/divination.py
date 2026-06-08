"""
User-initiated divination handler.

Powers POST /api/v1/divination: the user clicks "tarot" or "周易" in the
UI, picks their own cards / hexagram, and we return a reading. Separate
from the OccultAgent / BaziAgent that auto-generate every consensus round.

Two divination types:
    tarot   user picks 3 cards by index from the 78-card deck
    iching  user picks 1 hexagram from the 64-hexagram set

The reading is rendered through an LLM call so the prose feels mystical
and personalised. When no LLM is configured we fall back to a structured
template-based reading.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ----------------------------- card / hexagram data -----------------------------

# We re-export the tarot and hexagram tables from aegean-consensus so the
# divination module here is self-contained for testing. In production the
# data lives in aegean-consensus and AegeanBench loads it at startup.

# 78-card tarot deck - identical schema to aegean-consensus' occult_data.
# Duplicated here so AegeanBench can run divinations even when aegean-
# consensus is unreachable (mock mode).
TAROT_DECK: List[Dict[str, str]] = [
    {"name": "The Fool",            "keyword": "new beginning, unexpected outcome"},
    {"name": "The Magician",        "keyword": "skill, talent breakthrough"},
    {"name": "The High Priestess",  "keyword": "intuition, hidden tactical depth"},
    {"name": "The Empress",         "keyword": "abundance of chances created"},
    {"name": "The Emperor",         "keyword": "discipline, structured defence"},
    {"name": "The Hierophant",      "keyword": "tradition, conservative approach"},
    {"name": "The Lovers",          "keyword": "harmony in midfield partnership"},
    {"name": "The Chariot",         "keyword": "momentum, willpower advantage"},
    {"name": "Strength",            "keyword": "stamina, late-game resilience"},
    {"name": "The Hermit",          "keyword": "lone striker decides it"},
    {"name": "Wheel of Fortune",    "keyword": "luck swing, deflections matter"},
    {"name": "Justice",             "keyword": "VAR / refereeing in spotlight"},
    {"name": "The Hanged Man",      "keyword": "pause, unexpected tactical shift"},
    {"name": "Death",               "keyword": "end of an era, reset"},
    {"name": "Temperance",          "keyword": "balanced game, draws likely"},
    {"name": "The Devil",           "keyword": "indiscipline, red card risk"},
    {"name": "The Tower",           "keyword": "sudden collapse, big upset"},
    {"name": "The Star",            "keyword": "hope, rising young player shines"},
    {"name": "The Moon",            "keyword": "uncertainty, illusions in the press"},
    {"name": "The Sun",             "keyword": "clarity, dominant performance"},
    {"name": "Judgement",           "keyword": "decisive moment near full time"},
    {"name": "The World",           "keyword": "completion, tournament-defining"},
]
# Pad minor arcana
for suit, motif in (
    ("Cups",       "emotion, team chemistry"),
    ("Pentacles",  "physical condition, fitness"),
    ("Swords",     "intellect, tactical clarity"),
    ("Wands",      "energy, attacking momentum"),
):
    for rank in (
        "Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
        "Page", "Knight", "Queen", "King",
    ):
        TAROT_DECK.append(
            {"name": f"{rank} of {suit}", "keyword": f"{rank.lower()} energy in {motif}"}
        )

assert len(TAROT_DECK) == 78


# 64-hexagram I Ching set. Same list as aegean-consensus/bazi_data.HEXAGRAMS
HEXAGRAMS: List[Dict[str, str]] = [
    {"id": "01", "name": "乾",   "keyword": "creative force, dominant attack"},
    {"id": "02", "name": "坤",   "keyword": "receptive, defensive posture"},
    {"id": "03", "name": "屯",   "keyword": "difficulty at the start, slow open"},
    {"id": "04", "name": "蒙",   "keyword": "youthful inexperience, mistakes likely"},
    {"id": "05", "name": "需",   "keyword": "waiting, patient build-up"},
    {"id": "06", "name": "讼",   "keyword": "conflict, refereeing controversy"},
    {"id": "07", "name": "师",   "keyword": "the army, disciplined collective"},
    {"id": "08", "name": "比",   "keyword": "holding together, midfield bond"},
    {"id": "09", "name": "小畜", "keyword": "small accumulation, edge in margins"},
    {"id": "10", "name": "履",   "keyword": "treading carefully, away advantage"},
    {"id": "11", "name": "泰",   "keyword": "peace, balanced flow"},
    {"id": "12", "name": "否",   "keyword": "stagnation, scoreless first half"},
    {"id": "13", "name": "同人", "keyword": "fellowship, team chemistry"},
    {"id": "14", "name": "大有", "keyword": "great possessions, possession dominance"},
    {"id": "15", "name": "谦",   "keyword": "modesty, underdog overperforms"},
    {"id": "16", "name": "豫",   "keyword": "enthusiasm, momentum swings"},
    {"id": "17", "name": "随",   "keyword": "following, reactive tactics"},
    {"id": "18", "name": "蛊",   "keyword": "decay, slow start needs reset"},
    {"id": "19", "name": "临",   "keyword": "approach, decisive late attack"},
    {"id": "20", "name": "观",   "keyword": "observation, cautious opening"},
    {"id": "21", "name": "噬嗑", "keyword": "biting through, breakthrough"},
    {"id": "22", "name": "贲",   "keyword": "adornment, surface flair masks weakness"},
    {"id": "23", "name": "剥",   "keyword": "splitting apart, collapse"},
    {"id": "24", "name": "复",   "keyword": "return, comeback"},
    {"id": "25", "name": "无妄", "keyword": "innocence, no manipulation"},
    {"id": "26", "name": "大畜", "keyword": "great accumulation, stored power"},
    {"id": "27", "name": "颐",   "keyword": "nourishment, careful preparation"},
    {"id": "28", "name": "大过", "keyword": "great excess, over-extension"},
    {"id": "29", "name": "坎",   "keyword": "abysmal, repeated danger"},
    {"id": "30", "name": "离",   "keyword": "the clinging, brilliant attack"},
    {"id": "31", "name": "咸",   "keyword": "influence, mutual attraction"},
    {"id": "32", "name": "恒",   "keyword": "duration, sustained rhythm"},
    {"id": "33", "name": "遁",   "keyword": "retreat, parking the bus"},
    {"id": "34", "name": "大壮", "keyword": "great power, aggressive press"},
    {"id": "35", "name": "晋",   "keyword": "progress, climb up the table"},
    {"id": "36", "name": "明夷", "keyword": "darkening light, error in judgement"},
    {"id": "37", "name": "家人", "keyword": "the family, defensive solidity"},
    {"id": "38", "name": "睽",   "keyword": "opposition, formation clash"},
    {"id": "39", "name": "蹇",   "keyword": "obstruction, missed chances"},
    {"id": "40", "name": "解",   "keyword": "deliverance, tactical change works"},
    {"id": "41", "name": "损",   "keyword": "decrease, missing key player"},
    {"id": "42", "name": "益",   "keyword": "increase, returning star boosts squad"},
    {"id": "43", "name": "夬",   "keyword": "breakthrough, late winner"},
    {"id": "44", "name": "姤",   "keyword": "coming to meet, unexpected encounter"},
    {"id": "45", "name": "萃",   "keyword": "gathering together, crowd support"},
    {"id": "46", "name": "升",   "keyword": "pushing upward, climb in the second half"},
    {"id": "47", "name": "困",   "keyword": "oppression, low-scoring affair"},
    {"id": "48", "name": "井",   "keyword": "the well, deep squad depth"},
    {"id": "49", "name": "革",   "keyword": "revolution, complete tactical overhaul"},
    {"id": "50", "name": "鼎",   "keyword": "the cauldron, fortune favours"},
    {"id": "51", "name": "震",   "keyword": "thunder, sudden goal"},
    {"id": "52", "name": "艮",   "keyword": "keeping still, defensive lockdown"},
    {"id": "53", "name": "渐",   "keyword": "gradual progress, slow build"},
    {"id": "54", "name": "归妹", "keyword": "marrying maiden, mismatched roles"},
    {"id": "55", "name": "丰",   "keyword": "abundance, goalfest"},
    {"id": "56", "name": "旅",   "keyword": "the wanderer, neutral venue effect"},
    {"id": "57", "name": "巽",   "keyword": "the gentle, fluid passing"},
    {"id": "58", "name": "兑",   "keyword": "joyous, celebrating fans"},
    {"id": "59", "name": "涣",   "keyword": "dispersion, formation breakdown"},
    {"id": "60", "name": "节",   "keyword": "limitation, disciplined defence"},
    {"id": "61", "name": "中孚", "keyword": "inner truth, captain leads"},
    {"id": "62", "name": "小过", "keyword": "small excess, late drama"},
    {"id": "63", "name": "既济", "keyword": "after completion, lead held"},
    {"id": "64", "name": "未济", "keyword": "before completion, unfinished business"},
]
assert len(HEXAGRAMS) == 64


# ----------------------------- result types -----------------------------


@dataclass
class DivinationResult:
    """One reading. Returned as JSON to the caller."""
    type: str                            # "tarot" or "iching"
    match_id: str
    table_id: Optional[str] = None
    drawn_cards: List[Dict[str, str]] = field(default_factory=list)
    hexagram: Optional[Dict[str, str]] = None
    reading: str = ""                    # natural-language interpretation
    outcome_lean: Optional[Dict[str, float]] = None   # optional probability hint
    rationale: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "match_id": self.match_id,
            "table_id": self.table_id,
            "drawn_cards": self.drawn_cards,
            "hexagram": self.hexagram,
            "reading": self.reading,
            "outcome_lean": self.outcome_lean,
            "rationale": self.rationale,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
        }


# ----------------------------- core handler -----------------------------


_TAROT_POSITIONS = ("past", "present", "future")


# ----------------------------- public catalogs -----------------------------
#
# Two read-only endpoints expose the full deck and hexagram set so the
# frontend can render the pickers. Each entry uses a stable `index` field
# the frontend echoes back when the user makes a selection.


def get_tarot_catalog() -> Dict[str, Any]:
    """Full 78-card tarot deck for the frontend picker UI."""
    return {
        "_meta": {
            "endpoint": "GET /api/v1/divination/tarot/deck",
            "description": "All 78 cards. Frontend renders picker; user picks 3 indices.",
        },
        "total_cards": len(TAROT_DECK),
        "positions": list(_TAROT_POSITIONS),
        "cards": [
            {
                "index": i,
                "name": card["name"],
                "keyword": card["keyword"],
                "arcana": "major" if i < 22 else "minor",
            }
            for i, card in enumerate(TAROT_DECK)
        ],
    }


def get_iching_catalog() -> Dict[str, Any]:
    """Full 64-hexagram set for the frontend picker UI."""
    return {
        "_meta": {
            "endpoint": "GET /api/v1/divination/iching/hexagrams",
            "description": "All 64 hexagrams. Frontend renders picker; user picks 1 index.",
        },
        "total_hexagrams": len(HEXAGRAMS),
        "hexagrams": [
            {
                "index": i,
                "id": hex_["id"],
                "name": hex_["name"],
                "keyword": hex_["keyword"],
            }
            for i, hex_ in enumerate(HEXAGRAMS)
        ],
    }


def _normalise_indices(indices: List[int], deck_size: int, k: int) -> List[int]:
    """Clip / dedupe / pad incoming indices so we always end up with k of them."""
    seen: List[int] = []
    for raw in indices:
        try:
            idx = int(raw) % deck_size
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            seen.append(idx)
        if len(seen) == k:
            break
    # Pad with deterministic fallbacks if the user gave too few
    fallback = 0
    while len(seen) < k:
        if fallback not in seen:
            seen.append(fallback)
        fallback += 1
    return seen


def _render_tarot_prompt(
    home_team: str,
    away_team: str,
    drawn: List[Dict[str, str]],
) -> str:
    lines = [
        f"You are a tarot reader at a football match: {home_team} vs {away_team}.",
        "The user has drawn these three cards (past / present / future):",
    ]
    for card in drawn:
        lines.append(
            f"  - {card['position']}: {card['name']} ({card['keyword']})"
        )
    lines.extend([
        "",
        "Write a mystical-but-grounded reading in 3-4 sentences in English.",
        "Then estimate a probability distribution over the three outcomes.",
        "Keep probabilities reasonable (do not output 0.99 or 0).",
        "",
        "OUTPUT FORMAT (strict JSON):",
        '{ "reading": "...", "p_home_win": 0.45, "p_draw": 0.30, "p_away_win": 0.25 }',
    ])
    return "\n".join(lines)


def _render_iching_prompt(
    home_team: str,
    away_team: str,
    hexagram: Dict[str, str],
) -> str:
    return (
        f"You are an I Ching reader at a football match: {home_team} vs {away_team}.\n"
        f"The user has cast hexagram {hexagram.get('id')} ({hexagram.get('name')}) "
        f"meaning: {hexagram.get('keyword')}.\n\n"
        "Write a culturally-grounded but playful reading in 3-4 sentences "
        "in English.\nThen estimate a probability distribution over the "
        "three outcomes.\nKeep probabilities reasonable.\n\n"
        "OUTPUT FORMAT (strict JSON):\n"
        '{ "reading": "...", "p_home_win": 0.45, "p_draw": 0.30, "p_away_win": 0.25 }'
    )


def _parse_llm_json(raw: str) -> Dict[str, Any]:
    """Tolerant JSON extractor for LLM output."""
    import json
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def perform_divination(
    div_type: str,
    match_id: str,
    home_team: str = "Home",
    away_team: str = "Away",
    table_id: Optional[str] = None,
    card_indices: Optional[List[int]] = None,
    hexagram_index: Optional[int] = None,
    llm_call=None,
) -> DivinationResult:
    """
    Produce one divination reading.

    Args:
        div_type: 'tarot' or 'iching'
        match_id: which match the reading is for
        home_team / away_team: display names for the LLM prompt
        table_id: optional, echoed back for chat-room association
        card_indices: tarot only - user's 3 picks into TAROT_DECK
        hexagram_index: iching only - user's pick into HEXAGRAMS
        llm_call: optional async/sync callable (system, user) -> str
                  When None we fall back to a deterministic template
                  reading. This is what makes the endpoint work even when
                  LLM keys aren't configured yet.
    """
    import time
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    if div_type == "tarot":
        picks = _normalise_indices(card_indices or [], deck_size=len(TAROT_DECK), k=3)
        drawn = [
            {**TAROT_DECK[idx], "position": _TAROT_POSITIONS[i]}
            for i, idx in enumerate(picks)
        ]
        prompt = _render_tarot_prompt(home_team, away_team, drawn)
        result = DivinationResult(
            type="tarot",
            match_id=match_id,
            table_id=table_id,
            drawn_cards=drawn,
            hexagram=None,
            timestamp=timestamp,
        )
    elif div_type == "iching":
        idx = (hexagram_index or 0) % len(HEXAGRAMS)
        hexagram = dict(HEXAGRAMS[idx])
        prompt = _render_iching_prompt(home_team, away_team, hexagram)
        result = DivinationResult(
            type="iching",
            match_id=match_id,
            table_id=table_id,
            drawn_cards=[],
            hexagram=hexagram,
            timestamp=timestamp,
        )
    else:
        raise ValueError(f"unsupported divination type: {div_type!r}")

    # Call the LLM if available, else fall back to template
    reading_text = ""
    p_home, p_draw, p_away = 0.40, 0.30, 0.30
    tokens = 0

    if llm_call is None:
        reading_text = _fallback_reading(result)
    else:
        try:
            raw = llm_call(
                "You are a fortune-telling assistant for a football product.",
                prompt,
            )
            parsed = _parse_llm_json(raw)
            reading_text = str(parsed.get("reading") or "").strip() or _fallback_reading(result)
            p_home = float(parsed.get("p_home_win", p_home))
            p_draw = float(parsed.get("p_draw", p_draw))
            p_away = float(parsed.get("p_away_win", p_away))
        except Exception as e:
            logger.warning("divination LLM call failed: %s", e)
            reading_text = _fallback_reading(result)

    # Normalise the lean
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
    else:
        p_home, p_draw, p_away = 1/3, 1/3, 1/3

    result.reading = reading_text
    result.outcome_lean = {
        "p_home_win": round(p_home, 4),
        "p_draw": round(p_draw, 4),
        "p_away_win": round(p_away, 4),
    }
    result.rationale = (
        f"tarot: {', '.join(c['name'] for c in result.drawn_cards)}"
        if div_type == "tarot"
        else f"iching: {result.hexagram['name']} ({result.hexagram['id']})"
    )
    result.tokens_used = tokens
    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result


def _fallback_reading(result: DivinationResult) -> str:
    """Deterministic English template reading used when no LLM is configured."""
    if result.type == "tarot":
        cards = result.drawn_cards
        if not cards:
            return "No cards drawn; please pick three cards and try again."
        names = ", ".join(card["name"] for card in cards)
        return (
            f"You drew: {names}. Read together, the match opens under "
            f"the sign of {cards[0]['keyword']}, the current moment is "
            f"shaped by {cards[1]['keyword']}, and the endgame trends "
            f"toward {cards[2]['keyword']}. Keep a calm head and stake "
            f"with care."
        )
    if result.type == "iching" and result.hexagram:
        return (
            f"You have cast hexagram {result.hexagram['name']} "
            f"({result.hexagram['id']}): {result.hexagram['keyword']}. "
            f"This sign hints at the rhythm and momentum of the match; "
            f"observe before committing."
        )
    return "Divination could not be completed. Please try again."
