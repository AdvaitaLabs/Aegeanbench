"""
Arena aggregation service.

For one match it runs every model through the arena prompt and assembles
a side-by-side comparison:
  * aegean-consensus — enriched with our gateway data (data_block).
  * benchmark models — bare context, direct call.

Models run concurrently in a thread pool (the LLM clients are blocking).
Results are cached per (match_id, lang) with a long pre-match TTL so the
upcoming list and repeated views are instant and don't re-spend tokens.

Cost control: /arena/upcoming never computes — it only reads the cache
and reports status="pending" for matches not yet run (mirroring the
reference UI's "尚未运行"). Computation happens on /arena/matches/{id}
(or a future background warmer).
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aegeanbench.sports.arena.benchmark import ArenaModelRunner
from aegeanbench.sports.arena.config import load_benchmark_models

logger = logging.getLogger(__name__)

_AEGEAN_RUNNER_ID = "aegean-consensus"
_AEGEAN_DISPLAY = "Aegean Consensus"
_CACHE_TTL = 6 * 3600.0     # pre-match predictions stay fresh 6h
_COMPETITION = "FIFA World Cup 2026"


class ArenaService:
    def __init__(self, gateway=None, prediction_cache=None):
        self.gateway = gateway
        # The /predict cache. When set, the arena's aegean consensus run
        # also writes the standard /predict payload here, so one run serves
        # both the arena and /predict (no separate PrematchWarmer needed).
        self.prediction_cache = prediction_cache
        self._cache: Dict[str, tuple] = {}     # key -> (expires_at, payload)

    # ---------------- client wiring ----------------

    @staticmethod
    def _make_client(model_id: str):
        """OpenAI-compatible client via the shared Praka key, or None (mock)."""
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("sk-your"):
            return None
        from aegeanbench.sports.predictors.llm import OpenAICompatibleClient
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return OpenAICompatibleClient(
            api_key=key, model=model_id, base_url=base_url,
            timeout=float(os.getenv("ARENA_LLM_TIMEOUT", "120")),
        )

    def _aegean_runner(self) -> ArenaModelRunner:
        model_id = os.getenv("OPENAI_MODEL", "claude-opus-4-6")
        return ArenaModelRunner(
            runner_id=_AEGEAN_RUNNER_ID, display_name=_AEGEAN_DISPLAY,
            kind="aegean", model_id=model_id,
            cost_usd=float(os.getenv("AEGEAN_COST_USD", "0.11")),
            client=self._make_client(model_id),
        )

    def _benchmark_runners(self) -> List[ArenaModelRunner]:
        runners = []
        for m in load_benchmark_models():
            runners.append(ArenaModelRunner(
                runner_id=m.runner_id, display_name=m.display_name,
                kind="benchmark", model_id=m.model_id, cost_usd=m.cost_usd,
                client=self._make_client(m.model_id),
            ))
        return runners

    # ---------------- data block (our gateway data) ----------------

    def _data_block(self, match) -> Optional[str]:
        """
        Build the enriched context text for aegean from the gateway. This
        is the data that is ours-only — benchmarks never see it. Reuses the
        existing prompt builder's user section so the formatting matches
        what the consensus already consumes. Returns None on any failure
        (aegean then predicts bare, like a benchmark).
        """
        if self.gateway is None:
            return None
        try:
            ctx = self.gateway.build_context(match)
            from aegeanbench.sports.prompts import build_full_prompt
            return build_full_prompt(ctx)["user"]
        except Exception as e:  # noqa: BLE001
            logger.warning("arena data_block build failed for %s: %s", match.match_id, e)
            return None

    # ---------------- compute one match ----------------

    def compute_match(
        self,
        *,
        match_id: str,
        home_team: str,
        away_team: str,
        kickoff_iso: Optional[str] = None,
        venue: Optional[str] = None,
        stage: Optional[str] = None,
        lang: str = "en",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        cache_key = f"{match_id}|{lang}"
        if use_cache:
            hit = self._cache.get(cache_key)
            if hit and time.time() < hit[0]:
                out = dict(hit[1])
                out["_meta"] = {**out.get("_meta", {}), "cache_hit": True}
                return out

        # Resolve the Match (team names + shell) once. aegean needs it to
        # build the enriched context; benchmarks just need the names.
        match = None
        if self.gateway is not None:
            try:
                from aegeanbench.sports.reporter.prediction_service import resolve_match
                match = resolve_match(
                    match_id=match_id, home_team=home_team, away_team=away_team,
                    kickoff_iso=kickoff_iso, venue=venue,
                )
                if match is not None:
                    home_team = match.home_team.name or home_team
                    away_team = match.away_team.name or away_team
            except Exception as e:  # noqa: BLE001
                logger.warning("arena resolve failed for %s: %s", match_id, e)

        common = dict(home_team=home_team, away_team=away_team, kickoff_iso=kickoff_iso,
                      venue=venue, stage=stage, lang=lang)

        # aegean-consensus: the REAL multi-agent consensus on our enriched
        # data. Benchmarks: direct bare-context calls. All run concurrently.
        def _aegean_task() -> Dict[str, Any]:
            if self.gateway is not None and match is not None:
                from aegeanbench.sports.arena.aegean import build_aegean_entry
                return build_aegean_entry(
                    gateway=self.gateway, match=match, lang=lang,
                    cost_usd=float(os.getenv("AEGEAN_COST_USD", "0.11")),
                    predict_cache=self.prediction_cache,
                )
            # No gateway/match (e.g. offline dev): fall back to the bare
            # arena runner so the entry still appears.
            return self._aegean_runner().run(data_block=None, **common)

        tasks = [_aegean_task] + [
            (lambda r=r: r.run(data_block=None, **common))
            for r in self._benchmark_runners()
        ]

        # Concurrent — clients block on HTTP / the consensus call. Cap modestly.
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as pool:
            entries = list(pool.map(lambda fn: fn(), tasks))

        market = self._market_summary(entries)
        payload = {
            "_meta": {"endpoint": "GET /api/v1/arena/matches/{match_id}", "cache_hit": False},
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "kickoff_at": kickoff_iso,
            "venue": venue,
            "stage": stage or _COMPETITION,
            "market": market,
            "models": entries,
        }
        # Cache only if at least one model produced a real prediction.
        if any(e["status"] == "ok" for e in entries):
            self._cache[cache_key] = (time.time() + _CACHE_TTL, payload)
        return payload

    @staticmethod
    def _market_summary(entries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Blended win% across models that produced a prediction."""
        ok = [e for e in entries if e["status"] == "ok" and e["predicted_winner"]]
        if not ok:
            return {"home": 0, "draw": 0, "away": 0}
        n = len(ok)
        agg = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for e in ok:
            wp = e["win_probabilities"]
            for k in agg:
                agg[k] += float(wp.get(k, 0) or 0)
        return {k: round(v / n, 4) for k, v in agg.items()}

    # ---------------- upcoming (cache-only) ----------------

    def upcoming(self, *, lang: str = "en", hours: int = 72) -> Dict[str, Any]:
        """
        List upcoming fixtures with whatever is already cached. Never
        computes (cost-safe) — uncomputed models show status="pending".
        """
        matches: List[Dict[str, Any]] = []
        fixtures = []
        if self.gateway is not None:
            try:
                fixtures = self.gateway.list_fixtures(_COMPETITION) or []
            except Exception as e:  # noqa: BLE001
                logger.warning("arena upcoming: list_fixtures failed: %s", e)
        now = datetime.now(timezone.utc)
        for m in fixtures:
            ko = getattr(m, "kickoff_at", None)
            if ko is None:
                continue
            if ko.tzinfo is None:
                ko = ko.replace(tzinfo=timezone.utc)
            secs = (ko - now).total_seconds()
            if secs <= 0 or secs > hours * 3600:
                continue
            cached = self._cache.get(f"{m.match_id}|{lang}")
            if cached and time.time() < cached[0]:
                full = cached[1]
                matches.append(self._summary(full))
            else:
                matches.append({
                    "match_id": m.match_id,
                    "home_team": m.home_team.name,
                    "away_team": m.away_team.name,
                    "kickoff_at": ko.isoformat(),
                    "venue": getattr(m, "venue", None),
                    "market": {"home": 0, "draw": 0, "away": 0},
                    "models": [
                        {"runner_id": _AEGEAN_RUNNER_ID, "display_name": _AEGEAN_DISPLAY,
                         "kind": "aegean", "status": "pending"},
                    ] + [
                        {"runner_id": b.runner_id, "display_name": b.display_name,
                         "kind": "benchmark", "status": "pending"}
                        for b in load_benchmark_models()
                    ],
                })
        return {
            "_meta": {"endpoint": "GET /api/v1/arena/upcoming", "hours": hours},
            "matches": matches,
        }

    @staticmethod
    def _summary(full: Dict[str, Any]) -> Dict[str, Any]:
        """Light per-match card for the upcoming list (no heavy sections)."""
        return {
            "match_id": full["match_id"],
            "home_team": full["home_team"],
            "away_team": full["away_team"],
            "kickoff_at": full["kickoff_at"],
            "venue": full.get("venue"),
            "market": full["market"],
            "models": [
                {
                    "runner_id": e["runner_id"],
                    "display_name": e["display_name"],
                    "kind": e["kind"],
                    "status": e["status"],
                    "cost_usd": e.get("cost_usd"),
                    "predicted_winner": e.get("predicted_winner"),
                    "predicted_score": e.get("predicted_score"),
                    "win_probabilities": e.get("win_probabilities"),
                }
                for e in full["models"]
            ],
        }
