"""
Role-specific match brief builder for the @-mention QA endpoint.

When a user @s a specialist, we pull only what that role cares about
from the data sources and stitch it into a plain-text brief that gets
prepended to the LLM prompt. Each role grabs different fields:

    stats_specialist   - h2h last 5, FIFA codes
    player_specialist  - lineups (home + away)
    strategy_specialist- h2h + lineups (formation cues)
    market_specialist  - odds snapshot from soccersapi
    news_specialist    - weather snapshot + h2h
    chat_specialist    - nothing extra (chat heat lives in caller's context)
    occult_specialist  - nothing extra
    iching_specialist  - nothing extra

Failures are absorbed into "(unavailable)" rather than raised so a
flaky data source never blocks the @-mention reply.

Briefs are cached in-process for 60 seconds keyed by (match_id, role)
so a burst of @-mentions in the same room doesn't hammer the upstream
APIs.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ----------------------------- cache -----------------------------

_CACHE_TTL_SECONDS = 60.0
_cache: Dict[Tuple[str, str], Tuple[float, str]] = {}


def _cached(key: Tuple[str, str]) -> Optional[str]:
    hit = _cache.get(key)
    if hit is None:
        return None
    ts, value = hit
    if time.time() - ts > _CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return value


def _store(key: Tuple[str, str], value: str) -> None:
    _cache[key] = (time.time(), value)


# ----------------------------- adapter helpers -----------------------------


def _gateway():
    """
    Build a SportsDataGateway with real adapters (mock=False). If env keys
    are missing the adapters individually fall back to mock data — we
    accept that quietly.
    """
    from aegeanbench.sports.gateway import SportsDataGateway
    return SportsDataGateway(mock=False)


def _format_odds(odds_list) -> str:
    if not odds_list:
        return "  (no odds available)"
    # Top 3 by inverse margin (tightest book first)
    rows = []
    for o in odds_list[:5]:
        if not o.home_win or not o.draw or not o.away_win:
            continue
        margin = (1 / o.home_win) + (1 / o.draw) + (1 / o.away_win) - 1
        rows.append(
            f"  {o.bookmaker:<15} home {o.home_win:.2f}  draw {o.draw:.2f}  "
            f"away {o.away_win:.2f}  (margin {margin*100:.1f}%)"
        )
    return "\n".join(rows) if rows else "  (no usable odds rows)"


def _format_h2h(h2h_matches) -> str:
    if not h2h_matches:
        return "  (no head-to-head record on file)"
    rows = []
    for m in h2h_matches[:5]:
        r = m.result
        if r is None:
            continue
        rows.append(
            f"  {m.kickoff_at.date()}  "
            f"{m.home_team.fifa_code} {r.home_goals}-{r.away_goals} "
            f"{m.away_team.fifa_code}"
        )
    return "\n".join(rows) if rows else "  (no completed h2h matches)"


def _format_lineup(players, max_n: int = 8) -> str:
    if not players:
        return "  (lineup not posted yet)"
    rows = []
    for p in players[:max_n]:
        rows.append(f"  {p.position:<3} {p.name}  (age {p.age}, {p.goals_for_team} g for nation)")
    return "\n".join(rows)


def _format_weather(w: Optional[Dict[str, Any]]) -> str:
    if not w:
        return "  (weather not available)"
    return (
        f"  {w.get('summary', '?')}, "
        f"{w.get('temp_c', '?')}°C, "
        f"wind {w.get('wind_kph', '?')} km/h, "
        f"rain prob {w.get('rain_prob', '?')}%"
    )


# ----------------------------- per-role builders -----------------------------


def _brief_market(match_id: str, gw) -> str:
    try:
        odds = gw.soccersapi.fetch_odds(match_id)
    except Exception as e:
        logger.warning("odds fetch failed for %s: %s", match_id, e)
        return "Pre-match odds: (unavailable)"
    return "Pre-match 1X2 odds (top books):\n" + _format_odds(odds)


def _brief_player(match_id: str, home_fifa: str, away_fifa: str, gw) -> str:
    try:
        home = gw.soccersapi.fetch_lineup(match_id, home_fifa)
        away = gw.soccersapi.fetch_lineup(match_id, away_fifa)
    except Exception as e:
        logger.warning("lineup fetch failed for %s: %s", match_id, e)
        return "Lineups: (unavailable)"
    return (
        f"Home XI ({home_fifa}):\n" + _format_lineup(home) +
        f"\n\nAway XI ({away_fifa}):\n" + _format_lineup(away)
    )


def _brief_h2h(home_fifa: str, away_fifa: str, gw) -> str:
    try:
        h2h = gw.soccersapi.fetch_h2h(home_fifa, away_fifa, last_n=5)
    except Exception as e:
        logger.warning("h2h fetch failed: %s", e)
        return "Head-to-head: (unavailable)"
    return "Head-to-head (last 5):\n" + _format_h2h(h2h)


def _brief_weather(match_data: Dict[str, Any], gw) -> str:
    venue_city = match_data.get("venue_city")
    if not venue_city:
        return ""  # weather requires a city; skip silently
    try:
        w = gw.weather.fetch_forecast(venue_city)
    except Exception as e:
        logger.warning("weather fetch failed for %s: %s", venue_city, e)
        return "Weather: (unavailable)"
    return f"Match-day weather ({venue_city}):\n" + _format_weather(w)


# ----------------------------- top-level dispatch -----------------------------


_BRIEF_FNS = {
    "market_specialist":  ["odds"],
    "player_specialist":  ["lineup"],
    "strategy_specialist": ["h2h", "lineup"],
    "stats_specialist":   ["h2h"],
    "news_specialist":    ["weather", "h2h"],
}


def build_brief_for_role(
    role: str,
    match_id: Optional[str],
    match_data: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a plain-text brief tailored to the agent role. Returns "" when
    the role doesn't benefit from a brief (chat/occult/iching) or when
    nothing could be fetched.
    """
    if not match_id:
        return ""
    needs = _BRIEF_FNS.get(role)
    if not needs:
        return ""

    cache_key = (match_id, role)
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    md = match_data or {}
    home_team = md.get("home_team", "Home")
    away_team = md.get("away_team", "Away")
    home_fifa = md.get("home_fifa") or _fifa_guess(home_team)
    away_fifa = md.get("away_fifa") or _fifa_guess(away_team)

    gw = _gateway()
    sections = [f"Match {match_id}: {home_team} vs {away_team}"]
    if "odds" in needs:
        sections.append(_brief_market(match_id, gw))
    if "h2h" in needs:
        sections.append(_brief_h2h(home_fifa, away_fifa, gw))
    if "lineup" in needs:
        sections.append(_brief_player(match_id, home_fifa, away_fifa, gw))
    if "weather" in needs:
        wx = _brief_weather(md, gw)
        if wx:
            sections.append(wx)

    brief = "\n\n".join(s for s in sections if s)
    _store(cache_key, brief)
    return brief


# Tiny country->FIFA mapping for the most common names we'll see at the
# World Cup. Unknown names fall through to a 3-letter uppercase slice.
_FIFA_HINT = {
    "Mexico": "MEX", "South Africa": "RSA", "United States": "USA", "USA": "USA",
    "Argentina": "ARG", "Brazil": "BRA", "France": "FRA", "Germany": "GER",
    "Spain": "ESP", "England": "ENG", "Portugal": "POR", "Netherlands": "NED",
    "Italy": "ITA", "Belgium": "BEL", "Croatia": "CRO", "Japan": "JPN",
    "Korea Republic": "KOR", "South Korea": "KOR", "Morocco": "MAR",
    "Saudi Arabia": "KSA", "Canada": "CAN", "Australia": "AUS",
}


def _fifa_guess(country: str) -> str:
    if country in _FIFA_HINT:
        return _FIFA_HINT[country]
    return (country or "???")[:3].upper()
