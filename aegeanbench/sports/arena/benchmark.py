"""
Run one model through the arena prompt and assemble its per-model entry.

The same ArenaModelRunner serves both sides:
  * benchmark competitors — bare context (team names only), direct call.
  * aegean-consensus       — enriched `data_block` from our gateway.

A model that errors or returns unparseable output is reported with
status="unavailable" (mirroring the reference UI's "暂不可用") instead of
breaking the whole comparison.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, Optional

from aegeanbench.sports.arena.prompt import build_arena_prompt
from aegeanbench.sports.predictors.llm import _extract_json, _normalize_probs

logger = logging.getLogger(__name__)


_OUTCOME = ("home", "draw", "away")


def _winner(p_home: float, p_draw: float, p_away: float) -> str:
    return _OUTCOME[max(range(3), key=lambda i: (p_home, p_draw, p_away)[i])]


def _mock_arena_response(home: str, away: str, runner_id: str) -> Dict[str, Any]:
    """
    Deterministic full-schema response for offline dev / CI / when no API
    key is set. Varies by (teams, runner_id) so head-to-head comparisons
    look distinct without any network call.
    """
    seed = int(hashlib.sha256(f"{home}|{away}|{runner_id}".encode()).hexdigest()[:8], 16)
    ph = 0.30 + ((seed % 40) / 100.0)          # 0.30-0.70
    pa = 0.20 + ((seed // 40 % 35) / 100.0)    # 0.20-0.55
    ph, pd, pa = _normalize_probs(ph, max(0.1, 1 - ph - pa), pa)
    return {
        "p_home_win": round(ph, 3), "p_draw": round(pd, 3), "p_away_win": round(pa, 3),
        "confidence": round(max(ph, pd, pa), 2),
        "predicted_score": "1-1" if pd >= max(ph, pa) else ("2-1" if ph > pa else "1-2"),
        "rationale": f"[mock {runner_id}] 离线占位推理:基于队名哈希生成的可复现预测。",
        "score_distribution": [
            {"score": "1-1", "prob": 0.16}, {"score": "1-0", "prob": 0.12},
            {"score": "0-1", "prob": 0.11}, {"score": "2-1", "prob": 0.10},
            {"score": "1-2", "prob": 0.09}, {"score": "0-0", "prob": 0.08},
        ],
        "sections": {k: f"[mock {runner_id}] {k}" for k in (
            "odds_market", "lineup_analysis", "tactical", "h2h_recent",
            "player_matchups", "injuries", "upset_paths", "score_logic")},
        "lineup": {
            "home": {"formation": "4-3-3", "goalkeeper": "GK", "defenders": ["D1", "D2", "D3", "D4"],
                      "midfielders": ["M1", "M2", "M3"], "forwards": ["F1", "F2", "F3"], "subs": ["S1", "S2"]},
            "away": {"formation": "4-4-2", "goalkeeper": "GK", "defenders": ["D1", "D2", "D3", "D4"],
                      "midfielders": ["M1", "M2", "M3", "M4"], "forwards": ["F1", "F2"], "subs": ["S1", "S2"]},
        },
        "event_timeline": [
            {"minute": 23, "team": "home", "kind": "goal", "player": "F1", "detail": "assist: M1"},
            {"minute": 67, "team": "away", "kind": "goal", "player": "F1", "detail": ""},
        ],
        "match_stats": {
            "home": {"possession": 54, "shots": 12, "shots_on_target": 5, "corners": 6},
            "away": {"possession": 46, "shots": 9, "shots_on_target": 3, "corners": 4},
        },
    }


class ArenaModelRunner:
    """Runs one model through the arena prompt into a per-model entry dict."""

    def __init__(
        self,
        *,
        runner_id: str,
        display_name: str,
        kind: str,             # "aegean" | "benchmark"
        model_id: str,
        cost_usd: float = 0.0,
        client=None,           # LLMClient or None -> mock
    ):
        self.runner_id = runner_id
        self.display_name = display_name
        self.kind = kind
        self.model_id = model_id
        self.cost_usd = cost_usd
        self.client = client

    def run(
        self,
        *,
        home_team: str,
        away_team: str,
        kickoff_iso: Optional[str],
        venue: Optional[str],
        stage: Optional[str],
        lang: str = "en",
        data_block: Optional[str] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        tokens = 0
        try:
            if self.client is None:
                parsed = _mock_arena_response(home_team, away_team, self.runner_id)
                status = "ok"
            else:
                prompts = build_arena_prompt(
                    home_team=home_team, away_team=away_team, kickoff_iso=kickoff_iso,
                    venue=venue, stage=stage, lang=lang, data_block=data_block,
                )
                text, tokens = self.client.complete(prompts["system"], prompts["user"])
                parsed = _extract_json(text)
                status = "ok"
        except Exception as e:  # noqa: BLE001
            logger.warning("arena model %s failed: %s: %s", self.runner_id, type(e).__name__, e)
            return self._entry(status="unavailable", parsed={}, tokens=tokens,
                               latency_ms=int((time.perf_counter() - start) * 1000),
                               error=f"{type(e).__name__}: {e}")

        return self._entry(status=status, parsed=parsed, tokens=tokens,
                           latency_ms=int((time.perf_counter() - start) * 1000))

    def _entry(self, *, status: str, parsed: Dict[str, Any], tokens: int,
               latency_ms: int, error: str = None) -> Dict[str, Any]:
        ph = float(parsed.get("p_home_win", 0) or 0)
        pd = float(parsed.get("p_draw", 0) or 0)
        pa = float(parsed.get("p_away_win", 0) or 0)
        if status == "ok" and (ph + pd + pa) > 0:
            ph, pd, pa = _normalize_probs(ph, pd, pa)
            winner = _winner(ph, pd, pa)
        else:
            winner = None
        return {
            "runner_id": self.runner_id,
            "display_name": self.display_name,
            "kind": self.kind,
            "model": self.model_id,
            "status": status,
            "error": error,
            "cost_usd": self.cost_usd,
            "tokens_used": tokens,
            "latency_ms": latency_ms,
            "predicted_winner": winner,
            "predicted_score": parsed.get("predicted_score"),
            "win_probabilities": {"home": round(ph, 4), "draw": round(pd, 4), "away": round(pa, 4)}
                if winner else {"home": 0, "draw": 0, "away": 0},
            "confidence": parsed.get("confidence"),
            "score_distribution": parsed.get("score_distribution") or [],
            "rationale": parsed.get("rationale") or "",
            "sections": parsed.get("sections") or {},
            "lineup": parsed.get("lineup") or {},
            "event_timeline": parsed.get("event_timeline") or [],
            "match_stats": parsed.get("match_stats") or {},
        }
