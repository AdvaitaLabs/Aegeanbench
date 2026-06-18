"""
Tournament-forecast service: per-model orchestration, the factual
"actual" board, and caching.

Each model (aegean-consensus + benchmarks) produces a full forecast via
the segmented generate pipeline, seeded with the shared real state
(completed results locked). For the tournament view aegean uses the
consensus's base model — the multi-agent consensus is a per-match tool,
so here every model is differentiated only by the LLM, all on the same
real-results seed.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from aegeanbench.sports.arena.config import load_benchmark_models
from aegeanbench.sports.arena.tournament import generate as gen
from aegeanbench.sports.arena.tournament import real_state as rs
from aegeanbench.sports.arena.tournament import standings as st

logger = logging.getLogger(__name__)

_AEGEAN_RUNNER_ID = "aegean-consensus"
_AEGEAN_DISPLAY = "Aegean Consensus"
_FORECAST_TTL = 24 * 3600.0     # regenerated daily
_ACTUAL_TTL = 30 * 60.0         # real state refreshed every 30 min


class _Runner:
    def __init__(self, runner_id, display_name, kind, model_id, cost_usd):
        self.runner_id = runner_id
        self.display_name = display_name
        self.kind = kind
        self.model_id = model_id
        self.cost_usd = cost_usd


class TournamentService:
    def __init__(self, gateway=None):
        self.gateway = gateway
        self._cache: Dict[str, tuple] = {}      # "runner|lang" -> (expires, forecast)
        self._actual: Dict[str, tuple] = {}     # lang -> (expires, payload)
        self._state_cache: Optional[tuple] = None  # (expires, real_state)

    # ---------------- runners ----------------

    def _runners(self) -> List[_Runner]:
        runners = [_Runner(_AEGEAN_RUNNER_ID, _AEGEAN_DISPLAY, "aegean",
                           os.getenv("OPENAI_MODEL", "claude-opus-4-6"),
                           float(os.getenv("AEGEAN_TOURNAMENT_COST", "0.50")))]
        for m in load_benchmark_models():
            runners.append(_Runner(m.runner_id, m.display_name, "benchmark", m.model_id, m.cost_usd))
        return runners

    @staticmethod
    def _client(model_id: str):
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("sk-your"):
            return None
        from aegeanbench.sports.predictors.llm import OpenAICompatibleClient
        return OpenAICompatibleClient(
            api_key=key, model=model_id,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=float(os.getenv("TOURNAMENT_LLM_TIMEOUT", "240")),
        )

    # ---------------- shared real state ----------------

    def _real_state(self) -> Dict[str, Any]:
        if self._state_cache and time.time() < self._state_cache[0]:
            return self._state_cache[1]
        state = rs.fetch_real_state()
        self._state_cache = (time.time() + _ACTUAL_TTL, state)
        return state

    # ---------------- actual (factual) board ----------------

    def get_actual(self, lang: str = "en", use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and lang in self._actual and time.time() < self._actual[lang][0]:
            return self._actual[lang][1]
        state = self._real_state()
        groups = st.build_groups_with_standings([
            {"group": g["group"],
             "teams": g.get("teams") or [],
             "matches": [m for m in g.get("matches", []) if m.get("actual")]}
            for g in state.get("groups", [])
        ])
        payload = {
            "_meta": {"endpoint": "GET /api/v1/arena/tournament/actual",
                      "source": state.get("source")},
            "groups": groups,
            "knockout": {"rounds": state.get("knockout_results") or []},
            "top_scorers": [
                {**s, "rank": i + 1} for i, s in enumerate(state.get("top_scorers") or [])
            ],
            "champion": None, "runner_up": None, "third_place": None,
        }
        self._actual[lang] = (time.time() + _ACTUAL_TTL, payload)
        return payload

    # ---------------- per-model forecast ----------------

    def compute_model(self, runner_id: str, lang: str = "en", use_cache: bool = True) -> Optional[Dict[str, Any]]:
        key = f"{runner_id}|{lang}"
        if use_cache and key in self._cache and time.time() < self._cache[key][0]:
            out = dict(self._cache[key][1]); out["_meta"] = {**out.get("_meta", {}), "cache_hit": True}
            return out
        runner = next((r for r in self._runners() if r.runner_id == runner_id), None)
        if runner is None:
            return None
        state = self._real_state()
        start = time.time()
        try:
            forecast = gen.generate_forecast(client=self._client(runner.model_id),
                                            real_state=state, lang=lang)
            status = "ok" if forecast.get("champion") else "unavailable"
        except Exception as e:  # noqa: BLE001
            logger.warning("tournament generate failed for %s: %s", runner_id, e)
            return self._entry_meta(runner, status="unavailable", forecast={},
                                    latency_ms=int((time.time() - start) * 1000))
        payload = self._entry_meta(runner, status=status, forecast=forecast,
                                   latency_ms=int((time.time() - start) * 1000))
        if status == "ok":
            self._cache[key] = (time.time() + _FORECAST_TTL, payload)
        return payload

    @staticmethod
    def _entry_meta(runner: _Runner, *, status: str, forecast: Dict[str, Any], latency_ms: int) -> Dict[str, Any]:
        return {
            "_meta": {"endpoint": "GET /api/v1/arena/tournament/{runner_id}", "cache_hit": False},
            "runner_id": runner.runner_id, "display_name": runner.display_name,
            "kind": runner.kind, "model": runner.model_id, "cost_usd": runner.cost_usd,
            "status": status, "latency_ms": latency_ms,
            "champion": forecast.get("champion"),
            "runner_up": forecast.get("runner_up"),
            "third_place": forecast.get("third_place"),
            "narrative": forecast.get("narrative", ""),
            "groups": forecast.get("groups", []),
            "knockout": forecast.get("knockout", {"rounds": []}),
            "top_scorers": forecast.get("top_scorers", []),
        }

    # ---------------- list (summaries) ----------------

    def list_models(self, lang: str = "en") -> Dict[str, Any]:
        """Champion/runner-up/third per model — cached only; uncomputed = pending."""
        models = []
        for r in self._runners():
            key = f"{r.runner_id}|{lang}"
            hit = self._cache.get(key)
            if hit and time.time() < hit[0]:
                f = hit[1]
                models.append({"runner_id": r.runner_id, "display_name": r.display_name,
                               "kind": r.kind, "cost_usd": r.cost_usd, "status": f["status"],
                               "champion": f["champion"], "runner_up": f["runner_up"],
                               "third_place": f["third_place"]})
            else:
                models.append({"runner_id": r.runner_id, "display_name": r.display_name,
                               "kind": r.kind, "cost_usd": r.cost_usd, "status": "pending",
                               "champion": None, "runner_up": None, "third_place": None})
        return {"_meta": {"endpoint": "GET /api/v1/arena/tournament"}, "models": models}

    def compute_all(self, lang: str = "en", use_cache: bool = False) -> Dict[str, Any]:
        """Regenerate every model's forecast (used by the warmer). Concurrent."""
        runners = self._runners()
        with ThreadPoolExecutor(max_workers=min(8, len(runners))) as pool:
            pool.map(lambda r: self.compute_model(r.runner_id, lang, use_cache=use_cache), runners)
        return self.list_models(lang)
