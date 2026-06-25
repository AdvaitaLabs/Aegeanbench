"""
Turn (our model probability) + (Polymarket implied probability) into a
betting strategy: edge + fractional-Kelly stake + a recommendation.

Reuses the existing pure-math primitives (betting/kelly.py). Polymarket's
price IS the implied probability, so the decimal odds we'd take are
1 / price.

Only aegean-consensus feeds this — it's our calibrated probability vs the
market. Benchmark models are not run through it.

Degrades gracefully: when the market is missing ("none") we still return
the model probabilities with edge/stake = null and status "no_market", so
the front-end always has our prediction and just hides the edge column.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from aegeanbench.sports.arena.polymarket import PMMarket
from aegeanbench.sports.betting.kelly import kelly_fraction

_MIN_EDGE = float(os.getenv("STRATEGY_MIN_EDGE", "0.05"))


def _outcome_row(label: str, p_model: float, p_market: Optional[float]) -> Dict[str, Any]:
    if not p_market or p_market <= 0:
        return {"label": label, "p_model": round(p_model, 4), "p_market": None,
                "decimal_odds": None, "edge": None, "kelly_stake": None,
                "recommendation": "no_market"}
    decimal_odds = 1.0 / p_market
    k = kelly_fraction(p_model, decimal_odds)
    if k.edge < 0:
        rec = "avoid"            # market thinks it's likelier than we do
    elif k.edge >= _MIN_EDGE:
        rec = "value"            # we think it's underpriced -> bet
    else:
        rec = "no_value"         # positive but within noise/cost margin
    return {
        "label": label,
        "p_model": round(p_model, 4),
        "p_market": round(p_market, 4),
        "decimal_odds": round(decimal_odds, 3),
        "edge": round(k.edge, 4),
        "kelly_stake": round(k.fractional_kelly, 4),   # fraction of bankroll
        "recommendation": rec,
    }


def _wrap(kind: str, pm: PMMarket, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    has_value = any(o["recommendation"] == "value" for o in outcomes)
    return {
        "kind": kind,
        "market_status": pm.status,             # ok | thin | none
        "market_title": pm.title,
        "market_url": pm.url,
        "volume_usd": pm.volume_usd,
        "reason": pm.reason,
        "has_value_bet": has_value if pm.available else False,
        "note": ("仅参考:Polymarket 流动性低" if pm.status == "thin"
                 else ("Polymarket 暂无该市场,仅显示模型预测" if pm.status == "none" else "")),
        "outcomes": outcomes,
    }


def match_strategy(model_probs: Dict[str, float], pm: PMMarket) -> Dict[str, Any]:
    """model_probs: {home_win, draw, away_win}. pm: match-winner market."""
    rows = []
    for label in ("home_win", "draw", "away_win"):
        p_model = float(model_probs.get(label, 0) or 0)
        p_market = pm.prices.get(label) if pm.available else None
        rows.append(_outcome_row(label, p_model, p_market))
    return _wrap("match", pm, rows)


def outright_strategy(model_team_probs: Dict[str, float], pm: PMMarket,
                      kind: str = "champion", top_n: int = 20) -> Dict[str, Any]:
    """
    model_team_probs: {team: probability} from aegean's per-team table.
    pm: champion/qualification outright market (per-team prices).
    Rows are produced for every team we have a model probability for; the
    market price is attached when present, else null (per-team no_market).
    """
    rows = []
    for team, p_model in sorted(model_team_probs.items(), key=lambda kv: -float(kv[1] or 0))[:top_n]:
        p_market = None
        if pm.available:
            # match team name case-insensitively against market outcomes
            p_market = pm.prices.get(team)
            if p_market is None:
                for k, v in pm.prices.items():
                    if k.lower() == team.lower():
                        p_market = v
                        break
        rows.append(_outcome_row(team, float(p_model or 0), p_market))
    return _wrap(kind, pm, rows)
