"""
Empirical "form profile" computed from real international match results.

Replaces the rank-derived xG approximation with EMPIRICAL averages from
40,000+ recorded national-team fixtures (1872 → present). Source:
https://github.com/martj42/international_results  — community-maintained
CSV, updated within a few weeks of every international break.

What we compute per team (last 12 months by default):
    goals_for_per_game        avg goals scored
    goals_against_per_game    avg goals conceded
    matches_played
    win_rate, draw_rate, loss_rate
    last5_streak               compact "WWDLW" string

These are REAL numbers, not formulas. They're not the same as Opta xG
(which models shot quality), but they're the next-best empirical
strength signal available for free at the international level.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    "master/results.csv"
)
_CACHE_PATH = Path(os.path.expanduser(
    os.getenv("AEGEANBENCH_INTL_RESULTS_CACHE",
              "~/.aegeanbench/intl_results.csv")
))
_CACHE_TTL = timedelta(days=7)


_lock = threading.Lock()
_rows_cache: Optional[List[Dict[str, str]]] = None
_team_name_to_fifa: Dict[str, str] = {}


# Mapping FIFA 3-letter codes to the full names used by martj42's CSV.
# Only WC 2026 contenders + a handful of common opponents.
_FIFA_TO_NAME: Dict[str, str] = {
    "ARG": "Argentina", "ESP": "Spain", "FRA": "France", "ENG": "England",
    "BRA": "Brazil", "POR": "Portugal", "NED": "Netherlands",
    "BEL": "Belgium", "GER": "Germany", "CRO": "Croatia", "ITA": "Italy",
    "URU": "Uruguay", "MAR": "Morocco", "COL": "Colombia", "USA": "United States",
    "MEX": "Mexico", "JPN": "Japan", "SUI": "Switzerland", "DEN": "Denmark",
    "SEN": "Senegal", "POL": "Poland", "KOR": "South Korea", "AUS": "Australia",
    "ECU": "Ecuador", "AUT": "Austria", "WAL": "Wales", "UKR": "Ukraine",
    "TUN": "Tunisia", "CIV": "Ivory Coast", "PER": "Peru", "SRB": "Serbia",
    "EGY": "Egypt", "PAR": "Paraguay", "TUR": "Turkey", "NOR": "Norway",
    "VEN": "Venezuela", "PAN": "Panama", "CAN": "Canada", "QAT": "Qatar",
    "KSA": "Saudi Arabia", "JOR": "Jordan", "UZB": "Uzbekistan",
    "RSA": "South Africa", "IRN": "Iran", "JAM": "Jamaica", "CRC": "Costa Rica",
    "GHA": "Ghana", "CPV": "Cape Verde",
}


def _ensure_cached_csv() -> Optional[str]:
    """Download the CSV once, cache to disk for 7 days. Returns the text."""
    try:
        if _CACHE_PATH.exists():
            age = time.time() - _CACHE_PATH.stat().st_mtime
            if age < _CACHE_TTL.total_seconds():
                return _CACHE_PATH.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug("intl_results cache stat failed: %s", e)

    try:
        import requests
        r = requests.get(_CSV_URL, timeout=20)
        if r.status_code != 200 or len(r.text) < 1000:
            logger.warning("intl_results fetch HTTP %s (%d bytes)",
                           r.status_code, len(r.text or ""))
            # fall back to whatever is on disk even if stale
            return _CACHE_PATH.read_text(encoding="utf-8") if _CACHE_PATH.exists() else None
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(r.text, encoding="utf-8")
        logger.info("intl_results CSV cached (%d KB)", len(r.text) // 1024)
        return r.text
    except Exception as e:
        logger.warning("intl_results CSV download failed: %s", e)
        return _CACHE_PATH.read_text(encoding="utf-8") if _CACHE_PATH.exists() else None


def _load_rows() -> List[Dict[str, str]]:
    global _rows_cache
    with _lock:
        if _rows_cache is not None:
            return _rows_cache
        text = _ensure_cached_csv()
        if not text:
            _rows_cache = []
            return _rows_cache
        reader = csv.DictReader(io.StringIO(text))
        _rows_cache = list(reader)
        logger.info("intl_results loaded: %d matches", len(_rows_cache))
        return _rows_cache


def _team_matches(name: str, window_days: int) -> List[Dict[str, str]]:
    """Return rows where the team played, sorted newest first."""
    rows = _load_rows()
    if not rows:
        return []
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    out: List[Tuple[datetime, Dict[str, str]]] = []
    for row in rows:
        if row.get("home_team") != name and row.get("away_team") != name:
            continue
        try:
            d = datetime.strptime(row.get("date", "")[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if d < cutoff:
            continue
        out.append((d, row))
    out.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in out]


def form_profile(
    fifa_code: str,
    last_n: int = 10,
    window_days: int = 365,
) -> Dict[str, Any]:
    """
    Real empirical form for the given national team.

    Returns shape:
        {
          "goals_for_per_game": 1.45,
          "goals_against_per_game": 0.92,
          "matches_played": 9,
          "wins": 6, "draws": 2, "losses": 1,
          "win_rate": 0.67, "draw_rate": 0.22, "loss_rate": 0.11,
          "last5_streak": "WWDLW",
          "source": "martj42/international_results",
        }

    Returns {"source": "unavailable"} when the dataset wasn't loadable.
    """
    name = _FIFA_TO_NAME.get(fifa_code.upper())
    if not name:
        return {"source": "unavailable", "reason": f"no name for {fifa_code}"}

    matches = _team_matches(name, window_days)[:last_n]
    if not matches:
        return {"source": "unavailable", "reason": "no recent matches"}

    gf = ga = wins = draws = losses = 0
    streak: List[str] = []
    for row in matches:
        try:
            h_goals = int(row.get("home_score") or 0)
            a_goals = int(row.get("away_score") or 0)
        except ValueError:
            continue
        if row.get("home_team") == name:
            gf += h_goals
            ga += a_goals
            outcome = "W" if h_goals > a_goals else ("D" if h_goals == a_goals else "L")
        else:
            gf += a_goals
            ga += h_goals
            outcome = "W" if a_goals > h_goals else ("D" if a_goals == h_goals else "L")
        streak.append(outcome)
        if outcome == "W":
            wins += 1
        elif outcome == "D":
            draws += 1
        else:
            losses += 1

    n = len(matches)
    return {
        "goals_for_per_game": round(gf / n, 2),
        "goals_against_per_game": round(ga / n, 2),
        "matches_played": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / n, 2),
        "draw_rate": round(draws / n, 2),
        "loss_rate": round(losses / n, 2),
        "last5_streak": "".join(streak[:5]),
        "window_days": window_days,
        "source": "martj42/international_results",
    }
