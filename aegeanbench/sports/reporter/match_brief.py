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
from typing import Any, Dict, List, Optional, Tuple

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
    # Rank by tightest book first (lowest margin = sharpest price) and
    # only surface the top 3. The model doesn't need 18 lines of nearly-
    # identical numbers — that just dilutes attention and risks empty
    # responses on stricter Praka tiers.
    rows: List[Tuple[float, str]] = []
    for o in odds_list:
        if not o.home_win or not o.draw or not o.away_win:
            continue
        margin = (1 / o.home_win) + (1 / o.draw) + (1 / o.away_win) - 1
        rows.append((
            margin,
            f"  {o.bookmaker:<15} home {o.home_win:.2f}  draw {o.draw:.2f}  "
            f"away {o.away_win:.2f}  (margin {margin*100:.1f}%)"
        ))
    rows.sort(key=lambda r: r[0])
    return "\n".join(r[1] for r in rows[:3]) if rows else "  (no usable odds rows)"


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


def _format_lineup(players, per_position_cap: int = 3) -> str:
    """
    Render the squad balanced across positions so the model sees keepers,
    defenders, midfielders, AND forwards. football-data returns the
    squad sorted by position, so a naive head-N slice cuts off the
    attackers entirely.
    """
    if not players:
        return "  (lineup not posted yet)"
    by_pos: Dict[str, list] = {"GK": [], "DF": [], "MF": [], "FW": []}
    for p in players:
        bucket = by_pos.get(p.position, by_pos["MF"])
        if len(bucket) < per_position_cap:
            bucket.append(p)
    ordered = by_pos["GK"] + by_pos["DF"] + by_pos["MF"] + by_pos["FW"]
    rows = [
        f"  {p.position:<3} {p.name}"
        + (f"  (age {p.age})" if p.age else "")
        for p in ordered
    ]
    return "\n".join(rows) if rows else "  (lineup not posted yet)"


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


def _brief_player(
    match_id: str,
    home_fifa: str,
    away_fifa: str,
    gw,
    home_team_name: Optional[str] = None,
    away_team_name: Optional[str] = None,
) -> str:
    """
    Pull the 26-man squad from football-data first (real player names,
    DOB, position). Fall back to soccersapi's lineup endpoint when
    football-data is unavailable. Pre-match lineups proper require the
    paid tier on both providers, so squad is the most realistic real-
    data we can show before kickoff.
    """
    home: list = []
    away: list = []
    # Football-data wants the full team name ("Mexico"), not the FIFA code
    home_query = home_team_name or home_fifa
    away_query = away_team_name or away_fifa
    try:
        home = gw.football_data.fetch_squad(home_query)
    except Exception as e:
        logger.warning("football_data squad fetch (%s) failed: %s", home_query, e)
    try:
        away = gw.football_data.fetch_squad(away_query)
    except Exception as e:
        logger.warning("football_data squad fetch (%s) failed: %s", away_query, e)
    if not home or not away:
        # Fall back to soccersapi lineup stub so we never return empty
        try:
            home = home or gw.soccersapi.fetch_lineup(match_id, home_fifa)
            away = away or gw.soccersapi.fetch_lineup(match_id, away_fifa)
        except Exception as e:
            logger.warning("lineup fetch failed for %s: %s", match_id, e)
            return "Lineups: (unavailable)"
    return (
        f"Home squad ({home_fifa}, {len(home)} players):\n" + _format_lineup(home) +
        f"\n\nAway squad ({away_fifa}, {len(away)} players):\n" + _format_lineup(away)
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


# ----------------------------- match-id auto resolve -----------------------------


_match_info_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}


def _resolve_match_info(match_id: str) -> Dict[str, str]:
    """
    Auto-resolve team names + FIFA codes from soccersapi by match_id when
    the caller didn't pass match_data. 5-minute cache. Returns {} on
    failure so the brief falls back to "Home / Away".
    """
    hit = _match_info_cache.get(match_id)
    if hit and time.time() - hit[0] < 300:
        return hit[1]

    import os
    user = os.getenv("AEGEANBENCH_SOCCERSAPI_USER")
    token = os.getenv("AEGEANBENCH_SOCCERSAPI_KEY")
    if not user or not token:
        return {}
    try:
        import requests
        r = requests.get(
            "https://api.soccersapi.com/v2.2/fixtures/",
            params={"user": user, "token": token, "t": "info", "id": match_id},
            timeout=8,
        )
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data") or {}
        if isinstance(data, list):
            data = data[0] if data else {}
        teams = data.get("teams") or {}
        home = teams.get("home") or {}
        away = teams.get("away") or {}
        info = {
            "home_team": home.get("name", "Home"),
            "away_team": away.get("name", "Away"),
            "home_fifa": (home.get("country_iso") or home.get("short_code")
                          or _fifa_guess(home.get("name", "Home"))),
            "away_fifa": (away.get("country_iso") or away.get("short_code")
                          or _fifa_guess(away.get("name", "Away"))),
            "venue_city": (data.get("venue") or {}).get("city", ""),
        }
        _match_info_cache[match_id] = (time.time(), info)
        return info
    except Exception as e:
        logger.warning("soccersapi t=info lookup failed for %s: %s", match_id, e)
        return {}


# ----------------------------- top-level dispatch -----------------------------


_BRIEF_FNS = {
    "market_specialist":  ["odds"],
    "player_specialist":  ["lineup"],
    "strategy_specialist": ["h2h", "lineup"],
    "stats_specialist":   ["h2h", "odds"],
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

    # Merge caller-supplied match_data with auto-resolved match info.
    # Caller wins where it supplies a value; missing fields are filled
    # from the soccersapi lookup. This means the front-end can omit
    # match_data entirely and still get a correct brief.
    md = dict(match_data or {})
    if not md.get("home_team") or not md.get("away_team"):
        md = {**_resolve_match_info(match_id), **md}
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
        sections.append(_brief_player(
            match_id, home_fifa, away_fifa, gw,
            home_team_name=home_team, away_team_name=away_team,
        ))
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
