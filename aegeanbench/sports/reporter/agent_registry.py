"""
Public-facing agent metadata for the World Cup product front-end.

Exposed via GET /api/v1/agents. The shape is intentionally generous so
the UI has everything it needs (name, Chinese name, description, weight,
icon) without making additional calls.

This table is the canonical product-side definition. The actual agent
classes live in aegean-consensus/src/aegean/agents/sports/.
"""

from __future__ import annotations

from typing import Any, Dict, List


SPORTS_AGENTS_PUBLIC: List[Dict[str, Any]] = [
    {
        "id": "stats_specialist",
        "name": "Stats Analyst",
        "name_zh": "数据分析师",
        "weight": 0.90,
        "model": "claude-opus-4-7",
        "icon": "chart-line",
        "color": "#2563eb",
        "category": "expert",
        "is_default": False,
        "selectable": True,
        "description": (
            "Quantitative football analyst. Reads xG, PPDA, possession, "
            "and last 10 matches. Down-weights narrative-based reasoning."
        ),
        "description_zh": (
            "数据派。看 xG / PPDA / 控球率 / 近 10 场战绩，"
            "不靠故事推理。"
        ),
    },
    {
        "id": "player_specialist",
        "name": "Player Analyst",
        "name_zh": "球员专家",
        "weight": 0.85,
        "model": "gpt-5",
        "icon": "user",
        "color": "#16a34a",
        "category": "expert",
        "is_default": False,
        "selectable": True,
        "description": (
            "Squad and player-form analyst. Tracks star availability, "
            "injuries, suspensions, club form of key players."
        ),
        "description_zh": (
            "看球员。核心球员的状态、伤病、停赛、俱乐部表现。"
        ),
    },
    {
        "id": "strategy_specialist",
        "name": "Tactics Analyst",
        "name_zh": "战术专家",
        "weight": 0.80,
        "model": "claude-opus-4-7",
        "icon": "compass",
        "color": "#9333ea",
        "category": "expert",
        "is_default": False,
        "selectable": True,
        "description": (
            "Tactical analyst. Compares formations, pressing style, "
            "and head-to-head coaching history."
        ),
        "description_zh": (
            "看战术。阵型克制、教练风格、过往交手历史。"
        ),
    },
    {
        "id": "market_specialist",
        "name": "Market Reader",
        "name_zh": "市场专家",
        "weight": 0.75,
        "model": "deepseek-v3",
        "icon": "trending-up",
        "color": "#ea580c",
        "category": "expert",
        "is_default": False,
        "selectable": True,
        "description": (
            "Sharp betting market reader. Anchors on bookmaker consensus, "
            "watches for line moves and sharp money."
        ),
        "description_zh": (
            "看赔率。从博彩公司共识出发，留意异动和大资金。"
        ),
    },
    {
        "id": "news_specialist",
        "name": "News Analyst",
        "name_zh": "新闻分析师",
        "weight": 0.65,
        "model": "claude-haiku-4-5",
        "icon": "newspaper",
        "color": "#0891b2",
        "category": "expert",
        "is_default": False,
        "selectable": True,
        "description": (
            "Reads pre-match news, morale, locker-room drama, and "
            "late roster changes."
        ),
        "description_zh": (
            "看新闻。赛前舆情、更衣室、突发状况。"
        ),
    },
    {
        "id": "chat_specialist",
        "name": "Crowd Pulse",
        "name_zh": "群体情绪",
        "weight": 0.20,
        "model": "claude-haiku-4-5",
        "icon": "users",
        "color": "#db2777",
        "category": "crowd",
        "is_default": True,
        "selectable": False,
        "description": (
            "Aggregates user chat sentiment in this room. ALWAYS included "
            "automatically - users cannot toggle it off. Modest weight so "
            "the crowd cannot drown out the experts."
        ),
        "description_zh": (
            "听群里在说什么。每个聊天窗口自动加入，用户不能取消。"
            "权重低，避免被舆论牵着走。"
        ),
    },
    {
        "id": "occult_specialist",
        "name": "Tarot Reader",
        "name_zh": "塔罗大师",
        "weight": 0.10,
        "model": "claude-haiku-4-5",
        "icon": "moon",
        "color": "#7c3aed",
        "category": "occult",
        "is_default": False,
        "selectable": True,
        "description": (
            "Western fortune-telling. Draws a 3-card tarot spread, consults "
            "captain zodiac signs and date numerology. Near-zero weight."
        ),
        "description_zh": (
            "西方占卜。每场抽 3 张塔罗 + 队长星座 + 数字命理，"
            "权重极低不影响决策。"
        ),
    },
    {
        "id": "iching_specialist",
        "name": "I Ching Diviner",
        "name_zh": "周易大师",
        "weight": 0.10,
        "model": "claude-haiku-4-5",
        "icon": "yin-yang",
        "color": "#b45309",
        "category": "occult",
        "is_default": False,
        "selectable": True,
        "description": (
            "Chinese fortune-telling counterpart to the tarot reader. "
            "Casts an I Ching hexagram and its changing hexagram. "
            "Near-zero weight."
        ),
        "description_zh": (
            "东方玄学。每场起一个易经主卦 + 变卦。权重极低不影响决策。"
        ),
    },
]


def get_total_weight() -> float:
    return sum(a["weight"] for a in SPORTS_AGENTS_PUBLIC)


def get_max_weight_share() -> float:
    total = get_total_weight()
    return max(a["weight"] for a in SPORTS_AGENTS_PUBLIC) / total if total else 0.0


def get_default_agent_ids() -> List[str]:
    """Agent IDs that are auto-included in every table regardless of selection."""
    return [a["id"] for a in SPORTS_AGENTS_PUBLIC if a.get("is_default")]


def get_selectable_agent_ids() -> List[str]:
    """Agent IDs the user picks from. Default agents are excluded."""
    return [a["id"] for a in SPORTS_AGENTS_PUBLIC if a.get("selectable")]


def build_agents_endpoint() -> Dict[str, Any]:
    """
    Public payload for GET /api/v1/agents.

    Includes the weight-share sanity check (must stay < 50% per the
    paper's Refinement Validity guarantee) so the UI can surface it
    as a "neutrality badge".
    """
    return {
        "_meta": {
            "endpoint": "GET /api/v1/agents",
            "description": "The sports agents users can pick into a chat room",
        },
        "total_agents": len(SPORTS_AGENTS_PUBLIC),
        "total_weight": round(get_total_weight(), 4),
        "max_weight_share": round(get_max_weight_share(), 4),
        "safety_threshold": 0.50,
        "default_agent_ids": get_default_agent_ids(),
        "selectable_agent_ids": get_selectable_agent_ids(),
        "agents": SPORTS_AGENTS_PUBLIC,
    }
