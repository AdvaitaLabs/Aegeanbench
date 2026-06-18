"""
Segmented tournament-forecast generation with deterministic validation.

Pipeline (per model):
  1. LLM predicts scores for the REMAINING group matches (played ones are
     locked to real results).
  2. CODE computes group tables + the 32 advancers (standings.py).
  3. CODE builds the bracket structure and propagates winners; the LLM
     supplies the scorelines (looked up by team pair). This guarantees the
     bracket is always internally consistent — a model can't advance a
     team it didn't win with.
  4. CODE aggregates the top-scorer board (real goals + predicted).
  5. LLM writes the narrative.

Pass client=None for a deterministic offline mock (dev / CI / no key).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from aegeanbench.sports.arena.tournament import standings as st

logger = logging.getLogger(__name__)

_KO_ROUNDS = [
    ("round_of_32", 32), ("round_of_16", 16),
    ("quarter_final", 8), ("semi_final", 4), ("final", 2),
]


# ----------------------------- helpers -----------------------------

def _mock_score(a: str, b: str) -> Tuple[int, int]:
    s = int(hashlib.sha256(f"{a}|{b}".encode()).hexdigest()[:6], 16)
    return s % 4, (s // 4) % 3


def _llm_json(client, system: str, user: str) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        from aegeanbench.sports.predictors.llm import _extract_json
        text, _ = client.complete(system, user)
        return _extract_json(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("tournament LLM call failed: %s", e)
        return None


# ----------------------------- stage 1: groups -----------------------------

def _predict_group_scores(client, groups: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    """Fill unplayed group matches with predicted scores. Played stay locked."""
    pending = []
    for g in groups:
        for m in g["matches"]:
            if not m.get("played") and m.get("home_goals") is None:
                pending.append((g["group"], m))

    predicted: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    if pending and client is not None:
        zh = lang.lower().startswith("zh")
        sys = ("只输出 JSON。" if zh else "Output only JSON. ") + (
            'Predict the scoreline of each fixture. '
            'Return {"matches":[{"group":"A","home":"X","away":"Y","home_goals":int,"away_goals":int}]}')
        lines = [f"{g}: {m['home']} vs {m['away']}" for g, m in pending]
        played = [f"{g}: {m['home']} {m['home_goals']}-{m['away_goals']} {m['away']} (final)"
                  for g in groups for m in g['matches'] if g.get('played')]
        user = "Played:\n" + "\n".join(played) + "\n\nPredict these remaining group fixtures:\n" + "\n".join(lines)
        # mypy: g is a dict above; rebuild prompt correctly
        user = "Predict the remaining FIFA World Cup 2026 group fixtures:\n" + "\n".join(lines)
        out = _llm_json(client, sys, user) or {}
        for r in out.get("matches", []) or []:
            try:
                predicted[(r["group"], r["home"], r["away"])] = (int(r["home_goals"]), int(r["away_goals"]))
            except (KeyError, TypeError, ValueError):
                continue

    for g, m in pending:
        hg_ag = predicted.get((g, m["home"], m["away"])) or _mock_score(m["home"], m["away"])
        m["home_goals"], m["away_goals"] = int(hg_ag[0]), int(hg_ag[1])
        m["played"] = True            # predicted counts as played for the table
        m["actual"] = False
    return groups


# ----------------------------- stage 3: knockout -----------------------------

def _seed_order(groups: List[Dict[str, Any]]) -> List[str]:
    """Advancers ordered as seeds: group winners, then runners-up, then thirds."""
    winners, runners, thirds = [], [], []
    for g in groups:
        for r in g["standings"]:
            if not r.get("advanced"):
                continue
            if r["rank"] == 1:
                winners.append(r["team"])
            elif r["rank"] == 2:
                runners.append(r["team"])
            else:
                thirds.append(r["team"])
    return winners + runners + thirds


def _knockout_scores_from_llm(client, advancers: List[str], lang: str) -> Dict[frozenset, Tuple[int, int]]:
    """Ask the LLM for predicted knockout scorelines; index by team pair."""
    lookup: Dict[frozenset, Tuple[int, int]] = {}
    if client is None:
        return lookup
    zh = lang.lower().startswith("zh")
    sys = ("只输出 JSON。" if zh else "Output only JSON. ") + (
        'Given these 32 teams that reached the Round of 32, predict the whole '
        'knockout bracket. Return {"ties":[{"home":"X","away":"Y","home_goals":int,'
        '"away_goals":int}]} covering every tie you expect through the final and '
        'third-place match. Draws are decided on penalties — still give 90-min goals.')
    user = "32 teams (seed order):\n" + ", ".join(advancers)
    out = _llm_json(client, sys, user) or {}
    for t in out.get("ties", []) or []:
        try:
            lookup[frozenset((t["home"], t["away"]))] = (int(t["home_goals"]), int(t["away_goals"]))
        except (KeyError, TypeError, ValueError):
            continue
    return lookup


def _build_knockout(advancers: List[str], score_fn: Callable[[str, str, int], Tuple[int, int]]):
    """
    Propagate winners through the bracket deterministically. score_fn(home,
    away, seed_gap) -> (hg, ag). Ties resolve to the higher seed (earlier in
    `advancers`). Returns (rounds, champion, runner_up, third_place).
    """
    seed = {t: i for i, t in enumerate(advancers)}
    rounds: List[Dict[str, Any]] = []
    sf_losers: List[str] = []
    current = list(advancers)

    def play(home: str, away: str) -> Dict[str, Any]:
        hg, ag = score_fn(home, away, 0)
        if hg > ag:
            winner = home
        elif ag > hg:
            winner = away
        else:
            winner = home if seed.get(home, 1e9) < seed.get(away, 1e9) else away
        return {"home": home, "away": away, "home_goals": hg, "away_goals": ag,
                "winner": winner, "actual": False}

    for name, size in _KO_ROUNDS:
        if len(current) < size:
            break
        matches = []
        winners = []
        for i in range(0, size, 2):
            mt = play(current[i], current[i + 1])
            matches.append(mt)
            winners.append(mt["winner"])
            if name == "semi_final":
                loser = mt["away"] if mt["winner"] == mt["home"] else mt["home"]
                sf_losers.append(loser)
        rounds.append({"name": name, "matches": matches})
        current = winners

    champion = current[0] if current else None
    final_round = next((r for r in rounds if r["name"] == "final"), None)
    runner_up = None
    if final_round and final_round["matches"]:
        fm = final_round["matches"][0]
        runner_up = fm["away"] if fm["winner"] == fm["home"] else fm["home"]

    third_place = None
    if len(sf_losers) == 2:
        tp = play(sf_losers[0], sf_losers[1])
        rounds.append({"name": "third_place", "matches": [tp]})
        third_place = tp["winner"]

    return rounds, champion, runner_up, third_place


# ----------------------------- stage 4: scorers -----------------------------

def _top_scorers(real_scorers: List[Dict[str, Any]], champion: Optional[str]) -> List[Dict[str, Any]]:
    """
    Merge real goals with a light predicted boost. Real scorers keep their
    actual goals; we surface them sorted. (Predicted-only scorers can be
    layered in later; v1 leans on real data + the deep-run teams.)
    """
    rows = []
    for s in (real_scorers or []):
        g = int(s.get("goals") or 0)
        rows.append({"name": s.get("name"), "team": s.get("team"),
                     "goals": g, "actual_goals": g, "predicted_goals": 0})
    rows.sort(key=lambda r: -r["goals"])
    for i, r in enumerate(rows[:20], start=1):
        r["rank"] = i
    return rows[:20]


# ----------------------------- orchestrator -----------------------------

def generate_forecast(*, client, real_state: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
    """Run the full segmented pipeline into a tournament-forecast dict."""
    # Work on a deep-ish copy of the groups so we don't mutate the shared seed.
    import copy
    groups_in = copy.deepcopy(real_state.get("groups") or [])

    # 1. predict remaining group scores
    _predict_group_scores(client, groups_in, lang)
    # 2. compute tables + advancers (deterministic)
    groups = st.build_groups_with_standings(
        [{"group": g["group"], "teams": g.get("teams") or [t["team"] for t in g.get("standings", [])],
          "matches": g["matches"]} for g in groups_in]
    )
    advancers = _seed_order(groups)

    # 3. knockout — LLM scores, code propagation
    ko_lookup = _knockout_scores_from_llm(client, advancers, lang)

    def score_fn(home: str, away: str, _gap: int) -> Tuple[int, int]:
        return ko_lookup.get(frozenset((home, away))) or _mock_score(home, away)

    rounds, champion, runner_up, third = _build_knockout(advancers, score_fn)

    # 4. scorers
    scorers = _top_scorers(real_state.get("top_scorers") or [], champion)

    # 5. narrative
    narrative = ""
    if client is not None:
        zh = lang.lower().startswith("zh")
        sys = "用中文写 2-3 段赛事预测说明。" if zh else "Write a 2-3 paragraph tournament narrative."
        user = (f"Champion: {champion}; runner-up: {runner_up}; third: {third}. "
                f"Group winners: {advancers[:12]}. Explain the forecast.")
        try:
            narrative, _ = client.complete(sys, user)
        except Exception:  # noqa: BLE001
            narrative = ""
    if not narrative:
        narrative = (f"预测冠军 {champion}、亚军 {runner_up}、季军 {third}。"
                     if lang.lower().startswith("zh")
                     else f"Predicted champion {champion}, runner-up {runner_up}, third {third}.")

    return {
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third,
        "narrative": narrative,
        "groups": groups,
        "knockout": {"rounds": rounds},
        "top_scorers": scorers,
        "source_state": real_state.get("source"),
    }
