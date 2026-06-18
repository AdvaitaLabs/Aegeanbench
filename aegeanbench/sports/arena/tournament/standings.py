"""
Deterministic group-stage tables and qualification.

The LLM predicts match scores; everything below is computed in CODE so a
model can never hand us a table that contradicts its own scorelines, and
the set of advancing teams is always exactly right.

2026 format: 12 groups of 4. Top 2 of each group (24) plus the 8
best-ranked third-placed teams (8) advance to the Round of 32.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _blank_row(team: str) -> Dict[str, Any]:
    return {"team": team, "played": 0, "won": 0, "drawn": 0, "lost": 0,
            "gf": 0, "ga": 0, "gd": 0, "points": 0}


def compute_group_standings(teams: List[str], matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a sorted standings table for one group from its match results.
    Only matches with both scores set are counted. Returns rows sorted by
    points, goal difference, goals for, then team name (stable).
    """
    rows: Dict[str, Dict[str, Any]] = {t: _blank_row(t) for t in teams}
    for m in matches or []:
        h, a = m.get("home"), m.get("away")
        hg, ag = m.get("home_goals"), m.get("away_goals")
        if h not in rows or a not in rows or hg is None or ag is None:
            continue
        hg, ag = int(hg), int(ag)
        rh, ra = rows[h], rows[a]
        rh["played"] += 1; ra["played"] += 1
        rh["gf"] += hg; rh["ga"] += ag
        ra["gf"] += ag; ra["ga"] += hg
        if hg > ag:
            rh["won"] += 1; rh["points"] += 3; ra["lost"] += 1
        elif hg < ag:
            ra["won"] += 1; ra["points"] += 3; rh["lost"] += 1
        else:
            rh["drawn"] += 1; ra["drawn"] += 1; rh["points"] += 1; ra["points"] += 1
    for r in rows.values():
        r["gd"] = r["gf"] - r["ga"]
    ordered = sorted(rows.values(), key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(ordered, start=1):
        r["rank"] = i
        r["advanced"] = i <= 2          # top two always advance
    return ordered


def apply_best_thirds(groups: List[Dict[str, Any]], n_thirds: int = 8) -> List[str]:
    """
    Mark the best `n_thirds` third-placed teams as advanced (in place) and
    return the list of all advancing team names (24 group winners/runners-
    up + the best thirds).
    """
    thirds = []
    for g in groups:
        for r in g["standings"]:
            if r.get("rank") == 3:
                thirds.append((g["group"], r))
    thirds.sort(key=lambda gr: (-gr[1]["points"], -gr[1]["gd"], -gr[1]["gf"], gr[1]["team"]))
    for _, r in thirds[:n_thirds]:
        r["advanced"] = True

    advancers: List[str] = []
    for g in groups:
        for r in g["standings"]:
            if r.get("advanced"):
                advancers.append(r["team"])
    return advancers


def build_groups_with_standings(group_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    group_inputs: [{"group": "A", "teams": [...], "matches": [...]}, ...]
    Returns groups with computed "standings", plus best-thirds applied.
    """
    groups = []
    for gi in group_inputs:
        groups.append({
            "group": gi["group"],
            "standings": compute_group_standings(gi.get("teams") or [], gi.get("matches") or []),
            "matches": gi.get("matches") or [],
        })
    apply_best_thirds(groups)
    return groups


def advancing_teams(groups: List[Dict[str, Any]]) -> List[str]:
    return [r["team"] for g in groups for r in g["standings"] if r.get("advanced")]
