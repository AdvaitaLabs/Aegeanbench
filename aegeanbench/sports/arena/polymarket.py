"""
Polymarket (prediction-market) odds reader.

Polymarket has no bookmaker — a market's "odds" are the live order-book
price of each outcome share ($0–$1), i.e. the crowd's money-weighted
implied probability. Reading those prices is free and key-less:
  * Gamma API  (gamma-api.polymarket.com)  — find markets/events
  * CLOB API   (clob.polymarket.com)       — prices / midpoints / book

We only READ prices (to compute edge vs our model); we never trade, so no
wallet/auth is involved.

Graceful degradation is the whole point of this module: World Cup single
fixtures (especially minnows) often have NO Polymarket market or very thin
liquidity, while the outright "winner" / "to advance" markets are liquid.
Every fetch returns a PMMarket with an explicit status:
  * "ok"   — market found, liquidity adequate
  * "thin" — found but low volume / wide spread (use with caution)
  * "none" — no market for this event (caller shows model prob only)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_GAMMA = "https://gamma-api.polymarket.com"
_CLOB = "https://clob.polymarket.com"
# Liquidity thresholds (USD volume). Below MIN_OK -> "thin"; below MIN_ANY
# we still surface it but flagged. Tunable via env.
_MIN_OK_VOLUME = float(os.getenv("POLYMARKET_MIN_OK_VOLUME", "20000"))
_MAX_OK_SPREAD = float(os.getenv("POLYMARKET_MAX_OK_SPREAD", "0.06"))


@dataclass
class PMMarket:
    """A resolved Polymarket market with per-outcome implied probabilities."""
    status: str                                  # "ok" | "thin" | "none"
    kind: str                                    # "match" | "champion" | "qualification"
    title: str = ""
    url: str = ""
    volume_usd: float = 0.0
    spread: float = 0.0
    # outcome label -> implied probability (price). For a match:
    # {"home_win":.., "draw":.., "away_win":..}. For champion/qualification:
    # {team_name: prob}.
    prices: Dict[str, float] = field(default_factory=dict)
    reason: str = ""                             # why none/thin

    @property
    def available(self) -> bool:
        return self.status in ("ok", "thin")


class PolymarketClient:
    """
    Thin read-only client. When disabled (POLYMARKET_DISABLED=1) or the
    network/library is unavailable, returns status="none" so callers fall
    back to model-only output cleanly. Pass mock=True for offline tests.
    """

    def __init__(self, mock: bool = False, timeout: float = 8.0):
        self.mock = mock or os.getenv("POLYMARKET_DISABLED", "").lower() in ("1", "true", "yes")
        self.timeout = timeout

    # ---------------- low-level ----------------

    def _get(self, base: str, path: str, params: dict = None) -> Optional[object]:
        try:
            import requests
            r = requests.get(f"{base}{path}", params=params or {}, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("polymarket GET %s%s failed: %s", base, path, e)
            return None

    def _none(self, kind: str, reason: str) -> PMMarket:
        return PMMarket(status="none", kind=kind, reason=reason)

    @staticmethod
    def _classify(volume: float, spread: float) -> str:
        if volume <= 0:
            return "thin"
        if volume >= _MIN_OK_VOLUME and spread <= _MAX_OK_SPREAD:
            return "ok"
        return "thin"

    # ---------------- public: match winner ----------------

    def match_market(self, home_team: str, away_team: str) -> PMMarket:
        """Find the 3-way (or 2-way) match-winner market for a fixture."""
        if self.mock:
            return self._mock_match(home_team, away_team)
        # Gamma search: most WC single fixtures simply have no market.
        data = self._get(_GAMMA, "/markets", {"search": f"{home_team} {away_team}", "limit": 5, "active": "true"})
        market = _best_match_market(data, home_team, away_team)
        if not market:
            return self._none("match", f"no Polymarket market for {home_team} vs {away_team}")
        prices, vol, spread = _extract_outcome_prices(market, home_team, away_team)
        if not prices:
            return self._none("match", "market found but prices unavailable")
        return PMMarket(status=self._classify(vol, spread), kind="match",
                        title=market.get("question", ""), url=_market_url(market),
                        volume_usd=vol, spread=spread, prices=prices)

    # ---------------- public: outright (champion / qualification) ----------------

    def champion_market(self) -> PMMarket:
        """The 'World Cup winner' outright market: per-team prices."""
        if self.mock:
            return self._mock_outright("champion")
        data = self._get(_GAMMA, "/markets", {"search": "World Cup winner 2026", "limit": 10, "active": "true"})
        market = _best_outright_market(data)
        if not market:
            return self._none("champion", "no Polymarket World Cup winner market found")
        prices, vol, spread = _extract_team_prices(market)
        if not prices:
            return self._none("champion", "winner market found but prices unavailable")
        return PMMarket(status=self._classify(vol, spread), kind="champion",
                        title=market.get("question", ""), url=_market_url(market),
                        volume_usd=vol, spread=spread, prices=prices)

    def qualification_market(self) -> PMMarket:
        """'To advance / qualify' market: per-team advance prices (if any)."""
        if self.mock:
            return self._mock_outright("qualification")
        data = self._get(_GAMMA, "/markets", {"search": "World Cup to advance qualify group", "limit": 10, "active": "true"})
        market = _best_outright_market(data)
        if not market:
            return self._none("qualification", "no Polymarket qualification market found")
        prices, vol, spread = _extract_team_prices(market)
        if not prices:
            return self._none("qualification", "qualification market found but prices unavailable")
        return PMMarket(status=self._classify(vol, spread), kind="qualification",
                        title=market.get("question", ""), url=_market_url(market),
                        volume_usd=vol, spread=spread, prices=prices)

    # ---------------- mocks (offline dev / tests) ----------------

    def _mock_match(self, home: str, away: str) -> PMMarket:
        import hashlib
        h = int(hashlib.sha256(f"{home}|{away}".encode()).hexdigest()[:6], 16)
        # ~1 in 3 fixtures: pretend no market exists, to exercise the none path.
        if h % 3 == 0:
            return self._none("match", "[mock] no market for this fixture")
        ph = 0.30 + (h % 30) / 100.0
        pa = 0.25 + (h // 30 % 25) / 100.0
        pd = max(0.05, 1 - ph - pa)
        s = ph + pd + pa
        return PMMarket(status="ok", kind="match", title=f"[mock] {home} vs {away}",
                        url="https://polymarket.com/mock", volume_usd=50000, spread=0.03,
                        prices={"home_win": round(ph / s, 3), "draw": round(pd / s, 3),
                                "away_win": round(pa / s, 3)})

    def _mock_outright(self, kind: str) -> PMMarket:
        teams = ["Brazil", "France", "Spain", "Argentina", "England", "Germany",
                 "Portugal", "Netherlands"]
        raw = {t: 1.0 / (i + 2) for i, t in enumerate(teams)}
        s = sum(raw.values())
        return PMMarket(status="ok", kind=kind, title=f"[mock] {kind} market",
                        url="https://polymarket.com/mock", volume_usd=200000, spread=0.02,
                        prices={t: round(v / s, 3) for t, v in raw.items()})


# ----------------------------- parsing helpers -----------------------------
# Gamma payloads vary; these are defensive and return None/{} on any shape
# we don't recognise so the client degrades to "none" rather than crashing.

def _market_url(market: dict) -> str:
    slug = market.get("slug") or ""
    return f"https://polymarket.com/event/{slug}" if slug else ""


def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _best_match_market(data, home: str, away: str) -> Optional[dict]:
    if not isinstance(data, list):
        data = (data or {}).get("data") if isinstance(data, dict) else None
    if not isinstance(data, list):
        return None
    hl, al = home.lower(), away.lower()
    for m in data:
        q = (m.get("question") or "").lower()
        if hl in q and al in q:
            return m
    return None


def _best_outright_market(data) -> Optional[dict]:
    if not isinstance(data, list):
        data = (data or {}).get("data") if isinstance(data, dict) else None
    if not isinstance(data, list) or not data:
        return None
    # Highest-volume active market wins.
    return max(data, key=lambda m: _to_float(m.get("volume") or m.get("volumeNum")))


def _extract_outcome_prices(market: dict, home: str, away: str):
    """Map a match market's outcomes to home_win/draw/away_win probabilities."""
    import json
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    if isinstance(outcomes, str):
        try: outcomes = json.loads(outcomes)
        except Exception: outcomes = []
    if isinstance(prices, str):
        try: prices = json.loads(prices)
        except Exception: prices = []
    if not outcomes or not prices or len(outcomes) != len(prices):
        return {}, 0.0, 0.0
    hl, al = home.lower(), away.lower()
    out = {}
    for label, price in zip(outcomes, prices):
        l = str(label).lower()
        p = _to_float(price)
        if hl in l:
            out["home_win"] = p
        elif al in l:
            out["away_win"] = p
        elif "draw" in l or "tie" in l:
            out["draw"] = p
    vol = _to_float(market.get("volume") or market.get("volumeNum"))
    spread = _to_float(market.get("spread"))
    return out, vol, spread


def _extract_team_prices(market: dict):
    """Per-team prices for an outright (winner / advance) market."""
    import json
    outcomes = market.get("outcomes")
    prices = market.get("outcomePrices")
    if isinstance(outcomes, str):
        try: outcomes = json.loads(outcomes)
        except Exception: outcomes = []
    if isinstance(prices, str):
        try: prices = json.loads(prices)
        except Exception: prices = []
    if not outcomes or not prices or len(outcomes) != len(prices):
        return {}, 0.0, 0.0
    out = {str(t): _to_float(p) for t, p in zip(outcomes, prices)}
    vol = _to_float(market.get("volume") or market.get("volumeNum"))
    spread = _to_float(market.get("spread"))
    return out, vol, spread
