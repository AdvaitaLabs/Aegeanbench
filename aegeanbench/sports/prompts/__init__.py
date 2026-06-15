"""
Unified prompt builder for LLM-based and Aegean-based predictors.

Every runner that goes through an LLM (GPT-5, Claude, DeepSeek, Aegean
consensus) sees the EXACT same prompt. This is the core fairness guarantee
of the benchmark: differences in performance reflect model capability,
not prompt engineering luck.

Structure:
  SYSTEM:  role + output format contract
  USER:    structured match context (markdown for readability)

Output contract:
  Strict JSON with keys: p_home_win, p_draw, p_away_win, confidence,
  rationale, key_factors. Probabilities must sum to 1.0.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from aegeanbench.sports.gateway import MatchContext
from aegeanbench.sports.models import Match


_FALLBACK_SYSTEM = """You are a professional football match analyst.
Output strict JSON with keys p_home_win, p_draw, p_away_win, confidence,
rationale, key_factors. Probabilities must sum to 1.0.
"""


def _system_prompt() -> str:
    """
    Pull the /predict system prompt from templates.yaml so the product
    team can tune wording without code changes. Falls back to a minimal
    safe default if templates.yaml is missing.
    """
    from aegeanbench.sports.prompts.loader import get_template
    return get_template("predict.system_en", default=_FALLBACK_SYSTEM)


# Read at import time so call-sites that imported SYSTEM_PROMPT keep
# working unchanged. build_full_prompt() re-reads via _system_prompt()
# every request so edits take effect on aegeanbench restart.
SYSTEM_PROMPT = _system_prompt()


def _fmt_xg_line(xg: dict) -> str:
    """Render an xG profile defensively (missing fields shown as ?)."""
    def _num(key: str, suffix: str = "", pct: bool = False) -> str:
        val = xg.get(key)
        if val is None:
            return "?"
        try:
            v = float(val)
            return f"{v*100:.0f}%" if pct else f"{v:{suffix}}"
        except (ValueError, TypeError):
            return "?"
    base = (
        f"xG_for {_num('xg_for', '.2f')}, "
        f"xG_against {_num('xg_against', '.2f')}, "
        f"possession {_num('possession', pct=True)}, "
        f"PPDA {_num('ppda', '.1f')}"
    )
    # When the profile carries real recent-form numbers (from
    # martj42 international results), append them so the LLM sees the
    # empirical record alongside the (still rank-derived) possession/PPDA.
    extra = []
    if xg.get("matches_played") is not None:
        extra.append(
            f"recent {xg.get('matches_played', '?')} games: "
            f"{xg.get('wins', 0)}W-{xg.get('draws', 0)}D-{xg.get('losses', 0)}L"
        )
    if xg.get("last5_streak"):
        extra.append(f"last5 streak: {xg['last5_streak']}")
    if xg.get("fifa_rank"):
        extra.append(f"FIFA rank #{xg['fifa_rank']}")
    if extra:
        base += " | " + ", ".join(extra)
    return base


def _format_recent_form(history) -> str:
    """Render a team's last N matches as 'W-L-D vs OPP (score)' lines."""
    if not history:
        return "(no recent matches available)"
    lines = []
    for m in history[:5]:
        if m.result is None:
            continue
        outcome_for_team = (
            "W" if m.result.outcome.value == "home_win"
            else "L" if m.result.outcome.value == "away_win"
            else "D"
        )
        score = f"{m.result.home_goals}-{m.result.away_goals}"
        opp = m.away_team.fifa_code if outcome_for_team != "L" else m.home_team.fifa_code
        date = m.kickoff_at.strftime("%Y-%m-%d")
        lines.append(f"  - {date}  {outcome_for_team}  vs {opp}  ({score})")
    return "\n".join(lines) if lines else "(no recent matches with results)"


def _format_h2h(h2h_matches) -> str:
    if not h2h_matches:
        return "(no head-to-head history)"
    lines = []
    for m in h2h_matches[:5]:
        if m.result is None:
            continue
        score = f"{m.result.home_goals}-{m.result.away_goals}"
        lines.append(
            f"  - {m.kickoff_at.strftime('%Y-%m-%d')}  "
            f"{m.home_team.fifa_code} {score} {m.away_team.fifa_code}"
        )
    return "\n".join(lines) if lines else "(no h2h results)"


def _format_lineup(players, max_n: int = 5) -> str:
    if not players:
        return "(lineup not available)"
    rows = []
    for p in players[:max_n]:
        injury = " [INJURED]" if p.is_injured else ""
        susp = " [SUSPENDED]" if p.is_suspended else ""
        rows.append(
            f"  - {p.position} {p.name} (age {p.age}, "
            f"{p.goals_for_team} intl goals){injury}{susp}"
        )
    return "\n".join(rows)


def _format_weather(w: dict) -> str:
    """One-liner weather summary safe against missing fields."""
    return (
        f"  - {w.get('city', '?')}: "
        f"{w.get('conditions', 'Clear')}, "
        f"{w.get('temperature_c', '?')}°C, "
        f"humidity {w.get('humidity_pct', '?')}%, "
        f"wind {w.get('wind_kph', '?')} kph"
        + (
            f", precip {w['precipitation_mm_h']} mm/h"
            if w.get('precipitation_mm_h', 0) > 0 else ""
        )
    )


def _format_odds(match: Match) -> str:
    if not match.odds:
        return "(no odds available)"
    lines = []
    for o in match.odds[:3]:
        fp = o.fair_probs
        lines.append(
            f"  - {o.bookmaker}: H {o.home_win} (impl {fp['home_win']:.1%}) | "
            f"D {o.draw} (impl {fp['draw']:.1%}) | "
            f"A {o.away_win} (impl {fp['away_win']:.1%}) | "
            f"margin {o.margin*100:.2f}%"
        )
    consensus = match.consensus_odds
    if consensus:
        lines.append(
            f"  - CONSENSUS fair probabilities: "
            f"H {consensus['home_win']:.1%} | "
            f"D {consensus['draw']:.1%} | "
            f"A {consensus['away_win']:.1%}"
        )
    return "\n".join(lines)


def build_user_prompt(ctx: MatchContext, focus: Optional[str] = None) -> str:
    """
    Build the user-side prompt for a given MatchContext.

    Args:
        ctx: the match context with all enrichment populated
        focus: optional role hint for specialized agents (e.g., "stats",
            "player", "tactics"). When set, an extra "ANALYTICAL FOCUS"
            section nudges the model toward that lens. Default None gives
            the neutral generalist prompt used for raw single-LLM runners.

    Returns:
        Markdown-formatted prompt string ready to send to the LLM.
    """
    m = ctx.match
    home = m.home_team
    away = m.away_team
    home_xg = ctx.home_xg_profile
    away_xg = ctx.away_xg_profile

    parts: List[str] = []
    parts.append(f"# Match: {home.name} (home) vs {away.name} (away)")
    parts.append(f"Competition: {m.competition}")
    parts.append(f"Stage: {m.stage.value}")
    parts.append(f"Kickoff: {m.kickoff_at.isoformat()}")
    if m.venue:
        parts.append(f"Venue: {m.venue}")
    parts.append("")

    parts.append("## Team Ratings")
    parts.append(
        f"- {home.fifa_code} ({home.name}): "
        f"FIFA rank {home.fifa_rank or 'N/A'}, Elo {home.elo_rating or 'N/A'}"
    )
    parts.append(
        f"- {away.fifa_code} ({away.name}): "
        f"FIFA rank {away.fifa_rank or 'N/A'}, Elo {away.elo_rating or 'N/A'}"
    )
    parts.append("")

    parts.append("## Advanced Stats (last ~10 internationals)")
    parts.append(f"- {home.fifa_code}: {_fmt_xg_line(home_xg)}")
    parts.append(f"- {away.fifa_code}: {_fmt_xg_line(away_xg)}")
    parts.append("")

    parts.append(f"## {home.name} Recent Form")
    parts.append(_format_recent_form(ctx.home_history))
    parts.append("")
    parts.append(f"## {away.name} Recent Form")
    parts.append(_format_recent_form(ctx.away_history))
    parts.append("")

    # ===== LIVE STATE (most critical when present) =====
    # Render BEFORE pre-match data so the model anchors on what's
    # actually happening, not historical xG. When a match is 1-0 at
    # 47', telling the model that AFTER showing pre-match favourites
    # leads to the wrong prediction.
    live = getattr(ctx, "live_state", None) or {}
    if live:
        parts.insert(0, "")  # blank line for readability
        parts.insert(0, _render_live_state_block(
            home_code=home.fifa_code,
            away_code=away.fifa_code,
            home_name=home.name or home.fifa_code,
            away_name=away.name or away.fifa_code,
            live=live,
        ))

    parts.append("## Head-to-Head (last 5)")
    parts.append(_format_h2h(ctx.h2h))
    parts.append("")

    agg = getattr(ctx, "h2h_aggregate", None)
    if agg and agg.get("num_matches", 0) > 0:
        h = agg.get("home") or {}
        a = agg.get("away") or {}
        parts.append("## Head-to-Head (all-time aggregate)")
        parts.append(
            f"- Total meetings: {agg.get('num_matches', 0)} "
            f"({agg.get('total_goals', 0)} total goals)"
        )
        parts.append(
            f"- {h.get('name', home.fifa_code)}: "
            f"{h.get('wins', 0)}W {h.get('draws', 0)}D {h.get('losses', 0)}L"
        )
        parts.append(
            f"- {a.get('name', away.fifa_code)}: "
            f"{a.get('wins', 0)}W {a.get('draws', 0)}D {a.get('losses', 0)}L"
        )
        parts.append("")

    parts.append(f"## {home.name} Key Players")
    parts.append(_format_lineup(m.home_lineup))
    parts.append("")
    parts.append(f"## {away.name} Key Players")
    parts.append(_format_lineup(m.away_lineup))
    parts.append("")

    parts.append("## Market Odds (from bookmakers)")
    parts.append(_format_odds(m))
    parts.append("")

    if getattr(ctx, "weather", None):
        parts.append("## Weather at Kickoff")
        parts.append(_format_weather(ctx.weather))
        parts.append("")

    if getattr(ctx, "chat_summary", None):
        parts.append("## Crowd / Chat Sentiment")
        parts.append(ctx.chat_summary)
        parts.append("")

    if focus:
        parts.append("## ANALYTICAL FOCUS")
        parts.append(_focus_hint(focus))
        parts.append("")

    parts.append("---")
    parts.append("Output your probability distribution as strict JSON now.")

    return "\n".join(parts)


def _focus_hint(focus: str) -> str:
    """Per-role hint appended for specialized agents."""
    hints: Dict[str, str] = {
        "stats": (
            "Focus on the xG and advanced stats. Recent form weighted more "
            "than historical reputation. Ignore narrative-based reasoning."
        ),
        "player": (
            "Focus on key player availability and form. A missing star "
            "striker should materially shift probabilities. Consider "
            "injuries and suspensions explicitly."
        ),
        "strategy": (
            "Focus on tactical fit: pressing intensity (PPDA), possession "
            "battles, and historical matchups between coaching styles. "
            "Reference head-to-head patterns."
        ),
        "market": (
            "Focus on what the bookmaker consensus implies. Be skeptical of "
            "deviating far from the market unless data clearly supports it. "
            "Note any large discrepancies between bookmakers."
        ),
        "news": (
            "Focus on intangibles: team morale, recent controversies, coach "
            "stability, and any narrative shifts implied by the recent form."
        ),
        "occult": (
            "You are playing the role of a sports astrologer. Make a "
            "prediction based on numerology, team color symbolism, and "
            "intuition. This is for entertainment - keep probabilities "
            "loosely calibrated but include playful reasoning."
        ),
    }
    return hints.get(focus, f"Focus on the '{focus}' lens of analysis.")


def _render_live_state_block(
    home_code: str,
    away_code: str,
    home_name: str,
    away_name: str,
    live: Dict,
) -> str:
    """
    Render the LIVE STATE block. Sits at the top of the user prompt so
    the model anchors on what's happening NOW, not pre-match form.
    """
    status = (live.get("status") or "").lower()
    minute = live.get("minute") or 0
    hg = int(live.get("home_goals") or 0)
    ag = int(live.get("away_goals") or 0)
    diff = hg - ag

    headline = (
        f"# ⚡ LIVE STATE — DO NOT IGNORE\n"
        f"Status: {status.upper()} · minute {minute}'\n"
        f"Current score: {home_name} {hg}-{ag} {away_name}\n"
    )

    if status == "ft" or status == "finished":
        # Match is over, prediction is moot but still useful for post-game
        lean = (
            f"FT — {home_name} won {hg}-{ag}" if diff > 0
            else f"FT — {away_name} won {ag}-{hg}" if diff < 0
            else f"FT — draw {hg}-{ag}"
        )
        headline += f"\n{lean}.\n"
    elif diff != 0:
        leading = home_name if diff > 0 else away_name
        trailing = away_name if diff > 0 else home_name
        time_remaining = max(0, 90 - int(minute or 0))
        headline += (
            f"\n{leading} is LEADING by {abs(diff)}. {trailing} needs to "
            f"come from behind with ~{time_remaining}' remaining.\n"
            f"Your probability MUST factor this in: the trailing side's "
            f"probability of winning the match should drop sharply once "
            f"down a goal, especially deep in the second half.\n"
        )
    else:
        headline += (
            f"\nStill tied. Match is in the balance with "
            f"{max(0, 90 - int(minute or 0))}' remaining.\n"
        )

    events = live.get("recent_events") or []
    if events:
        headline += "\nRecent events:\n"
        for ev in events[-6:]:
            headline += (
                f"  {ev.get('minute', '?')}' {ev.get('kind', '?')}"
                f" · {ev.get('team') or '?'}"
                f"{' · ' + ev['player'] if ev.get('player') else ''}\n"
            )
    return headline


def build_full_prompt(
    ctx: MatchContext,
    focus: Optional[str] = None,
    lang: str = "en",
) -> Dict[str, str]:
    """
    Return both system and user prompts as a dict.

    Reads the active system prompt from prompts/templates.yaml on every
    call so product-team edits take effect after `docker compose
    restart aegeanbench` without a redeploy.

    When `lang='zh'` an extra directive (also from YAML) is appended so
    the model writes its `rationale` in Simplified Chinese; the JSON
    schema and probability fields stay English so parsing is unaffected.
    """
    from aegeanbench.sports.prompts.loader import get_template
    system = get_template("predict.system_en", default=_FALLBACK_SYSTEM)
    if lang == "zh":
        suffix = get_template("predict.system_zh_suffix", default="")
        if suffix:
            system = system.rstrip() + "\n\n" + suffix
    return {
        "system": system,
        "user": build_user_prompt(ctx, focus=focus),
    }
