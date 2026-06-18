"""
The arena prompt: asks any LLM to predict one match and return the full
rich schema the arena UI renders (win probs, score distribution, full
reasoning, analysis sections, lineup, event timeline, match stats).

Used by both sides:
  * aegean-consensus passes a populated `data_block` (our gateway data).
  * benchmark models pass an empty `data_block` — they predict from their
    own knowledge ("direct call").

Output is a single strict JSON object so it parses uniformly regardless
of which model produced it.
"""

from __future__ import annotations

from typing import Optional

# The exact JSON shape every model must return. Kept verbose on purpose:
# the more explicit the schema, the more consistent the cross-model output.
_SCHEMA = """{
  "p_home_win": 0.0-1.0,
  "p_draw": 0.0-1.0,
  "p_away_win": 0.0-1.0,            // the three MUST sum to ~1.0
  "confidence": 0.0-1.0,
  "predicted_score": "e.g. 1-2",   // most likely exact score
  "rationale": "full multi-paragraph reasoning",
  "score_distribution": [           // 8-16 most likely exact scores
    {"score": "0-1", "prob": 0.0-1.0}
  ],
  "sections": {
    "odds_market": "implied market / odds prior",
    "lineup_analysis": "squad depth & availability read",
    "tactical": "formations, pressing, transitions",
    "h2h_recent": "head-to-head + recent form",
    "player_matchups": "key individual duels",
    "injuries": "injuries / suspensions / fatigue",
    "upset_paths": "how the underdog or a draw could happen",
    "score_logic": "why the scoreline above"
  },
  "lineup": {
    "home": {"formation": "4-3-3", "goalkeeper": "name",
              "defenders": ["..."], "midfielders": ["..."],
              "forwards": ["..."], "subs": ["..."]},
    "away": {"formation": "4-3-3", "goalkeeper": "name",
              "defenders": ["..."], "midfielders": ["..."],
              "forwards": ["..."], "subs": ["..."]}
  },
  "event_timeline": [               // predicted key events, time-ordered
    {"minute": 1-90, "team": "home|away", "kind": "goal|yellow_card|red_card|substitution|penalty",
     "player": "name", "detail": "e.g. assist: name"}
  ],
  "match_stats": {                  // predicted full-time stats
    "home": {"possession": 0-100, "shots": int, "shots_on_target": int, "corners": int},
    "away": {"possession": 0-100, "shots": int, "shots_on_target": int, "corners": int}
  }
}"""


def build_arena_prompt(
    *,
    home_team: str,
    away_team: str,
    kickoff_iso: Optional[str],
    venue: Optional[str],
    stage: Optional[str],
    lang: str = "en",
    data_block: Optional[str] = None,
):
    """Return {"system": ..., "user": ...} for the arena prediction."""
    zh = (lang or "en").lower().startswith("zh")

    if zh:
        system = (
            "你是一名顶级足球赛事分析师。请预测这场比赛并**只输出一个严格的 JSON 对象**,"
            "不要任何额外文字、不要 markdown 代码围栏。三个胜负概率必须和约为 1.0。"
            "rationale 与各 section 用中文撰写,内容具体、有数据支撑。"
            "阵容/时间轴/数据统计如无确切信息,基于你的足球知识给出合理预测。\n\n"
            f"严格按此 JSON 结构输出(注释仅说明,不要写进结果):\n{_SCHEMA}"
        )
    else:
        system = (
            "You are an elite football match analyst. Predict this match and "
            "output ONLY a single strict JSON object — no extra prose, no markdown "
            "fences. The three outcome probabilities must sum to ~1.0. Write the "
            "rationale and each section in English, concrete and evidence-based. "
            "If exact lineup/timeline/stats are unknown, give a reasonable prediction "
            "from your football knowledge.\n\n"
            f"Return exactly this JSON shape (comments are explanatory only):\n{_SCHEMA}"
        )

    lines = [
        f"Match: {home_team} (home) vs {away_team} (away)",
        f"Kickoff: {kickoff_iso or 'TBD'}",
        f"Venue: {venue or 'neutral'}",
        f"Stage: {stage or 'FIFA World Cup 2026 group stage'}",
    ]
    if data_block:
        # aegean-consensus path: real data we pulled. Benchmarks omit this.
        lines.append("")
        lines.append("Reference data (use it; it is reliable):")
        lines.append(data_block)
    else:
        lines.append("")
        lines.append("No external data is provided — predict from your own knowledge.")

    return {"system": system, "user": "\n".join(lines)}
