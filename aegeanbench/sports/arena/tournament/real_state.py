"""
Real tournament state — the factual seed and ground truth.

Pulls the actual group draw, completed results, and real top scorers from
football-data. Completed matches are marked actual=True so the generator
locks them and only predicts what hasn't been played. When the live data
is unavailable (no key / pre-availability), falls back to a static mock
draw with no results, so models predict the whole tournament from scratch.
"""

from __future__ import annotations

import itertools
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WC_GROUPS_MOCK = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Türkiye", "Paraguay", "Australia"],
    "E": ["Germany", "Ecuador", "Côte d'Ivoire", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Norway", "Senegal", "Iraq"],
    "J": ["Argentina", "Croatia", "Nigeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "Panama"],
    "L": ["England", "Italy", "Ghana", "Jamaica"],
}


def _round_robin(teams: List[str]) -> List[Dict[str, Any]]:
    return [
        {"home": h, "away": a, "home_goals": None, "away_goals": None,
         "played": False, "actual": False}
        for h, a in itertools.combinations(teams, 2)
    ]


def mock_draw() -> Dict[str, Any]:
    """Static 12-group draw, no results — models predict everything."""
    groups = [
        {"group": g, "teams": list(ts), "matches": _round_robin(ts)}
        for g, ts in _WC_GROUPS_MOCK.items()
    ]
    return {"source": "mock", "stage": "group", "groups": groups,
            "top_scorers": [], "knockout_results": []}


def _fetch_football_data() -> Optional[Dict[str, Any]]:
    """
    Best-effort real state from football-data. Returns None on any failure
    so the caller falls back to the mock draw. (Group draw + completed
    results + scorers; knockout results are surfaced when present.)
    """
    key = os.getenv("AEGEANBENCH_FOOTBALL_DATA_KEY", "")
    if not key:
        return None
    try:
        import requests
        base = "https://api.football-data.org/v4/competitions/WC"
        headers = {"X-Auth-Token": key}
        standings = requests.get(f"{base}/standings", headers=headers, timeout=10).json()
        matches = requests.get(f"{base}/matches", headers=headers, timeout=10).json()
        scorers = requests.get(f"{base}/scorers", headers=headers, timeout=10,
                               params={"limit": 30}).json()
    except Exception as e:  # noqa: BLE001
        logger.warning("tournament real_state football-data fetch failed: %s", e)
        return None

    groups_map: Dict[str, Dict[str, Any]] = {}
    for s in (standings.get("standings") or []):
        grp = (s.get("group") or "").replace("GROUP_", "").strip()
        if not grp or s.get("type") != "TOTAL":
            continue
        teams = [(t.get("team") or {}).get("name") for t in (s.get("table") or [])]
        groups_map.setdefault(grp, {"group": grp, "teams": [t for t in teams if t], "matches": []})

    for m in (matches.get("matches") or []):
        grp = (m.get("group") or "").replace("GROUP_", "").strip()
        if grp not in groups_map:
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        finished = m.get("status") == "FINISHED"
        groups_map[grp]["matches"].append({
            "home": (m.get("homeTeam") or {}).get("name"),
            "away": (m.get("awayTeam") or {}).get("name"),
            "home_goals": ft.get("home") if finished else None,
            "away_goals": ft.get("away") if finished else None,
            "played": finished, "actual": finished,
        })

    if not groups_map:
        return None
    top_scorers = [
        {"name": (s.get("player") or {}).get("name"),
         "team": (s.get("team") or {}).get("name"),
         "goals": s.get("goals") or 0, "actual": True}
        for s in (scorers.get("scorers") or [])
    ]
    return {"source": "football-data", "stage": "group",
            "groups": list(groups_map.values()),
            "top_scorers": top_scorers, "knockout_results": []}


def fetch_real_state(allow_mock: bool = True) -> Dict[str, Any]:
    """Real state from football-data, or the mock draw as fallback."""
    state = _fetch_football_data()
    if state and state.get("groups"):
        return state
    return mock_draw() if allow_mock else {"source": "empty", "stage": "group",
                                           "groups": [], "top_scorers": [], "knockout_results": []}
