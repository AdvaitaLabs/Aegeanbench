"""
Optional FastAPI server that exposes the reporter endpoints over HTTP.

V1 (Day 6) endpoints:
    GET  /api/v1/runners                     predictor catalog
    GET  /api/v1/tournaments                 tournament list
    GET  /api/v1/leaderboard                 ranked across runs
    GET  /api/v1/runs                        recent run manifests
    GET  /api/v1/runs/{id}                   full run bundle
    GET  /api/v1/runs/{id}/matches/{mid}     match drill-down
    GET  /api/v1/runners/{rid}/card          predictor profile

V2 (added in Day 6 extension) endpoints:
    POST /api/v1/agents/{agent_id}/answer    @-mention single-agent Q&A
    WS   /ws/predictions                     live prediction updates push
    WS   /ws/matches/{match_id}              live score + re-evaluation push

The server has no state across requests for V1 reads. V2 WebSocket
endpoints keep per-connection state (subscriptions, last-pushed payload)
but never persist anything user-specific to disk.

Usage:
    uvicorn aegeanbench.sports.reporter.server:app --port 8200

FastAPI is loaded lazily so this module can be imported without the
dependency installed; the actual create_app() call will fail loudly with
an install hint if FastAPI is missing.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel

    class AgentAnswerRequest(BaseModel):
        """JSON body schema for POST /api/v1/agents/{id}/answer."""
        question: str
        match_id: Optional[str] = None
        # 'zh' or 'en'. When omitted, auto-detected from the question.
        lang: Optional[str] = None
        match_context: Optional[str] = None
        # When match_context is omitted but match_id + match_data are
        # provided, the server fetches a role-specific brief (odds for
        # market_specialist, lineups for player_specialist, etc.) and
        # uses it as context. match_data accepts home_team/away_team/
        # home_fifa/away_fifa/kickoff_at — all optional.
        match_data: Optional[Dict[str, Any]] = None
        user_name: Optional[str] = None
        room_id: Optional[str] = None
        recent_messages: Optional[List[Dict[str, Any]]] = None

    class PublishRequest(BaseModel):
        channel: str = "predictions"
        payload: Dict[str, Any] = {}

    class ChatSignalRequest(BaseModel):
        """
        JSON body for POST /api/v1/chat/signal.

        Sent by the chat service when activity in a room crosses a threshold
        that may warrant a fresh consensus run (e.g. a flurry of messages
        right after a goal, or a sudden sentiment shift).
        """
        match_id: str
        room_id: Optional[str] = None
        message_count: int = 0
        window_seconds: int = 300
        strong_sentiment: bool = False
        sentiment_label: Optional[str] = None

    class ChatMessageInput(BaseModel):
        """One chat message from the user-managed chat room."""
        user_name: str = "anon"
        text: str = ""
        timestamp: Optional[str] = None
        language: Optional[str] = None

    class MatchDataInput(BaseModel):
        """Optional rich match context the frontend can include."""
        home_team: Optional[str] = None
        away_team: Optional[str] = None
        kickoff_at: Optional[str] = None
        venue: Optional[str] = None
        odds: Optional[Dict[str, float]] = None
        # Optional override for reply language: 'zh' or 'en'. When
        # omitted, the server auto-detects from chat snippets / falls
        # back to English.
        lang: Optional[str] = None

    class PredictRequest(BaseModel):
        """
        JSON body for POST /api/v1/predict.

        Frontend builds a 'table' (chat room) and picks which agents to
        include. chat_specialist is always added automatically by the
        server; the frontend's agent_ids list does not need to mention it.
        """
        match_id: str
        agent_ids: List[str] = []
        match_data: Optional[MatchDataInput] = None
        chat_messages: Optional[List[ChatMessageInput]] = None
        table_id: Optional[str] = None
        user_id: Optional[str] = None
        # Reply language: 'zh' or 'en'. Front-end should pass this from
        # the user's locale (zh-CN / en-US). When omitted the server
        # tries chat_messages and falls back to English.
        lang: Optional[str] = None

    class DivinationRequest(BaseModel):
        """
        JSON body for POST /api/v1/divination.

        User points at the tarot or 周易 widget in the UI, picks their own
        cards or hexagram, and we return a reading. card_indices is used
        for tarot, hexagram_index for iching - the other field can be
        omitted depending on type.
        """
        type: str   # "tarot" or "iching"
        match_id: str
        home_team: Optional[str] = "Home"
        away_team: Optional[str] = "Away"
        table_id: Optional[str] = None
        # 'zh' or 'en'. Auto-detected from home_team / away_team when omitted.
        lang: Optional[str] = None
        card_indices: Optional[List[int]] = None        # tarot
        hexagram_index: Optional[int] = None             # iching
except ImportError:
    AgentAnswerRequest = None  # type: ignore[assignment]
    PublishRequest = None  # type: ignore[assignment]
    ChatSignalRequest = None  # type: ignore[assignment]
    ChatMessageInput = None  # type: ignore[assignment]
    MatchDataInput = None  # type: ignore[assignment]
    PredictRequest = None  # type: ignore[assignment]
    DivinationRequest = None  # type: ignore[assignment]

from aegeanbench.sports.orchestrator.persistence import (
    DEFAULT_RUNS_DIR,
    list_runs,
    load_run,
)
from aegeanbench.sports.reporter.agent_registry import build_agents_endpoint
from aegeanbench.sports.reporter.endpoints import (
    RUNNER_REGISTRY,
    build_dashboard_endpoint,
    build_leaderboard_endpoint,
    build_match_detail_endpoint,
    build_match_state_endpoint,
    build_run_endpoint,
    build_runner_card_endpoint,
    build_runners_endpoint,
    build_tournaments_endpoint,
)

logger = logging.getLogger(__name__)


def create_app(
    runs_dir: Optional[Path] = None,
    qa_handler: Optional[object] = None,
):
    """
    Build a FastAPI application with V1 reads + V2 Q&A + WebSocket push.

    Args:
        runs_dir: override the persisted-run directory (default ~/.aegeanbench/...).
        qa_handler: optional ChatQAHandler (from aegean-consensus) for the
            V2 @-mention POST endpoint. When None, the endpoint returns 503.

    Returns:
        FastAPI app instance ready to be served by uvicorn.

    Raises:
        RuntimeError: when fastapi isn't installed.
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise RuntimeError(
            "fastapi is required to use reporter.server. "
            "Install with: pip install fastapi uvicorn"
        ) from e

    runs_root = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR

    app = FastAPI(
        title="AegeanBench World Cup Reporter",
        version="0.2.0",
        description="Reporter endpoints + V2 Q&A + live WebSocket streams.",
    )

    # Permissive CORS for sprint dev; tighten before production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # In-process pub/sub for WebSocket fan-out. Keeps per-connection
    # subscriptions only - no persistent storage of user data.
    from aegeanbench.sports.reporter.realtime import LiveHub
    from aegeanbench.sports.live.scheduler import MatchEventScheduler
    live_hub = LiveHub()
    scheduler = MatchEventScheduler()
    app.state.live_hub = live_hub
    app.state.qa_handler = qa_handler
    app.state.scheduler = scheduler

    # Prediction cache (60s TTL, LRU) so repeated /predict for the same
    # table doesn't re-run consensus. Goal/red-card events from the
    # live poller invalidate the relevant match below.
    from aegeanbench.sports.reporter.prediction_cache import PredictionCache
    app.state.prediction_cache = PredictionCache()

    # Live event poller wired into the asyncio loop on app startup. It
    # asks soccersapi every 30s for in-play matches, pushes events
    # onto the WebSocket channel, and invalidates the prediction cache
    # when high-priority events (goal / red card / half-time / full-
    # time) land. Soft-fails if the soccersapi plan doesn't cover
    # events — logged once, then idle.
    from aegeanbench.sports.live.poller import LiveEventPoller
    from aegeanbench.sports.live.prematch import PrematchWarmer

    # Language the background workers compute live / pre-match consensus
    # in. All viewers of a match share one broadcast, so we pick a single
    # default (override with AEGEAN_LIVE_LANG). Per-user language still
    # works on an explicit force=True /predict.
    _live_lang = os.getenv("AEGEAN_LIVE_LANG", "zh")

    def _shared_gateway():
        gw = getattr(app.state, "gateway", None)
        if gw is None:
            from aegeanbench.sports.gateway import SportsDataGateway
            gw = SportsDataGateway(mock=False)
            app.state.gateway = gw
        return gw

    @app.on_event("startup")
    async def _start_live_poller():
        if os.getenv("LIVE_POLLER_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.warning("LIVE_POLLER_DISABLED is set — poller not started")
            return
        logger.warning("startup hook: building LiveEventPoller")
        # Feature B: hand the poller the gateway + default panel so it can
        # re-run consensus itself and broadcast the result (one run per
        # match per refresh, independent of how many viewers are watching)
        # instead of just telling every client to re-fetch.
        poller = LiveEventPoller(
            live_hub=live_hub,
            scheduler=scheduler,
            prediction_cache=app.state.prediction_cache,
            gateway=_shared_gateway(),
            lang=_live_lang,
        )
        poller.start()
        app.state.live_poller = poller

    @app.on_event("startup")
    async def _start_prematch_warmer():
        if os.getenv("PREMATCH_WARMER_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.warning("PREMATCH_WARMER_DISABLED is set — warmer not started")
            return
        logger.warning("startup hook: building PrematchWarmer")
        # Feature A: pre-compute consensus at T-24h / T-1h so users get
        # instant cached reads before kickoff. Shares the scheduler with
        # the poller for consistent throttle / once-per-checkpoint state.
        warmer = PrematchWarmer(
            gateway=_shared_gateway(),
            scheduler=scheduler,
            prediction_cache=app.state.prediction_cache,
            lang=_live_lang,
        )
        warmer.start()
        app.state.prematch_warmer = warmer

    @app.on_event("shutdown")
    async def _stop_background_tasks():
        for attr in ("live_poller", "prematch_warmer", "arena_warmer", "tournament_warmer"):
            task = getattr(app.state, attr, None)
            if task is not None:
                await task.stop()

    # Pre-warm cold caches in a background thread at boot so the first
    # /answer or /predict doesn't wait for: (a) the martj42 results CSV
    # download (~3 MB, ~2-3s) and (b) the FIFA rank live fetch. Both
    # have their own internal cache; this just triggers it eagerly.
    def _prewarm():
        # Use logger.warning so messages survive INFO->WARNING filtering
        # in production. These run once at boot, not noisy.
        try:
            from aegeanbench.sports.sources.international_results import _load_rows
            n = len(_load_rows())
            logger.warning("prewarm: international_results loaded %d matches", n)
        except Exception as e:
            logger.warning("prewarm international_results failed: %s", e)
        try:
            from aegeanbench.sports.sources.fifa_rankings import (
                _get_rankings, last_source,
            )
            n = len(_get_rankings())
            logger.warning("prewarm: fifa_rankings loaded %d teams (source=%s)",
                           n, last_source())
        except Exception as e:
            logger.warning("prewarm fifa_rankings failed: %s", e)

    import threading
    threading.Thread(target=_prewarm, daemon=True, name="prewarm").start()

    def _all_runs():
        runs = []
        if runs_root.exists():
            for entry in runs_root.iterdir():
                if entry.is_dir() and (entry / "manifest.json").exists():
                    try:
                        runs.append(load_run(entry))
                    except Exception as exc:
                        logger.warning("skipping run %s: %s", entry.name, exc)
        return runs

    @app.get("/api/v1/runners")
    def get_runners():
        return build_runners_endpoint()

    @app.get("/api/v1/tournaments")
    def get_tournaments():
        return build_tournaments_endpoint()

    @app.get("/api/v1/leaderboard")
    def get_leaderboard(tournament_id: str = "fifa-world-cup-2026"):
        return build_leaderboard_endpoint(_all_runs(), tournament_id=tournament_id)

    @app.get("/api/v1/runs")
    def get_runs(limit: int = 50):
        manifests = list_runs(runs_root)[:limit]
        return {"runs": manifests}

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str):
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        try:
            return build_run_endpoint(load_run(run_dir))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/v1/runs/{run_id}/matches/{match_id}")
    def get_match_detail(run_id: str, match_id: str):
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        payload = build_match_detail_endpoint(load_run(run_dir), match_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"unknown match {match_id}")
        return payload

    @app.get("/api/v1/runners/{runner_id}/card")
    def get_runner_card(runner_id: str):
        if runner_id not in RUNNER_REGISTRY:
            raise HTTPException(status_code=404, detail=f"unknown runner {runner_id}")
        return build_runner_card_endpoint(_all_runs(), runner_id)

    # ---------- sports agent metadata (public) ----------

    @app.get("/api/v1/agents")
    def get_agents():
        """The sports agents users can pick into a chat room."""
        return build_agents_endpoint()

    # ---------- one-shot consensus prediction ----------

    @app.post("/api/v1/predict")
    async def post_predict(body: PredictRequest):
        """
        Run a consensus prediction for one match with a caller-specified
        agent panel + caller-supplied chat history. Stateless: the server
        does not remember the table between calls.

        Behaviour:
            - default_agent_ids (currently just chat_specialist) are
              merged into the panel automatically
            - Unknown agent_ids are dropped (logged as warning)
            - chat_messages are forwarded into the ChatAgent's prompt
            - Returns the full prediction + discussion trace
        """
        from aegeanbench.sports.reporter.agent_registry import (
            SPORTS_AGENTS_PUBLIC,
            get_default_agent_ids,
        )
        valid_ids = {a["id"] for a in SPORTS_AGENTS_PUBLIC}
        requested = [aid for aid in (body.agent_ids or []) if aid in valid_ids]
        for must_have in get_default_agent_ids():
            if must_have not in requested:
                requested.append(must_have)
        if not requested:
            raise HTTPException(
                status_code=400,
                detail="at least one valid agent_id is required",
            )

        # Reply language priority:
        #   1. body.lang (explicit override from caller)
        #   2. match_data.lang (legacy alias)
        #   3. auto-detect from chat_messages (CJK in any msg -> zh)
        #   4. English fallback when nothing is detectable
        from aegeanbench.sports.gateway import SportsDataGateway
        from aegeanbench.sports.lang import detect_from_signals
        from aegeanbench.sports.reporter import prediction_service as _svc

        md = body.match_data or MatchDataInput()
        explicit_lang = body.lang or (
            body.match_data.model_dump().get("lang") if body.match_data else None
        )
        chat_texts = [m.text for m in (body.chat_messages or [])]
        lang = explicit_lang or detect_from_signals(None, chat_texts) or "en"

        # Singleton gateway so adapter HTTP clients and the file cache
        # stay warm across requests. Lazy-built on first /predict.
        gw = getattr(app.state, "gateway", None)
        if gw is None:
            gw = SportsDataGateway(mock=False)
            app.state.gateway = gw

        cache = app.state.prediction_cache
        cache_key = cache.make_key(
            match_id=body.match_id,
            agent_ids=requested,
            lang=lang,
            chat_messages=body.chat_messages,
        )

        def _echo_cached(p: Dict[str, Any], reused: bool) -> Dict[str, Any]:
            out = dict(p)
            meta = dict(out.get("_meta") or {})
            meta["cache_hit"] = True
            if reused:
                meta["reused"] = True
            out["_meta"] = meta
            out["table_id"] = body.table_id   # echo per-caller field
            return out

        # 1) Exact hit: byte-identical request within TTL, including a
        #    pre-warmed pre-match prediction.
        exact = cache.get(cache_key)
        if exact is not None:
            return _echo_cached(exact, reused=False)

        # 2) Reuse the latest real prediction for this match in this
        #    language (features C + B), regardless of chat signature or
        #    agent panel. This is fully transparent to the front-end —
        #    same request, same response shape:
        #      - C (chat debounce): casual group chat re-POSTs /predict on
        #        every message; all reuse one prediction. Refreshes are
        #        driven by the schedulers (pre-match checkpoints / live
        #        ticker), NOT by user chat, so chatter never burns opus.
        #      - B (live): every viewer reuses the single prediction the
        #        live ticker computed instead of each running consensus.
        #    The TTL encodes freshness (pre-match warm = long, live ticker
        #    = short and always newest), so once a match is under way the
        #    ticker's entry wins and we never serve a stale pre-match one.
        #    lang-filtered so a zh viewer is never served en text.
        latest = cache.get_latest_for_match(body.match_id, lang=lang)
        if latest is not None:
            return _echo_cached(latest, reused=True)

        # 3) Fresh consensus: only when nothing is cached for this match +
        #    language yet (first request, or everything expired). The heavy
        #    work runs in a worker thread inside the service so the event
        #    loop (and WebSocket fan-out) stays responsive.
        payload = await _svc.run_and_cache(
            gateway=gw,
            cache=cache,
            match_id=body.match_id,
            agent_ids=requested,
            lang=lang,
            home_team=md.home_team,
            away_team=md.away_team,
            kickoff_iso=md.kickoff_at,
            venue=md.venue,
            chat_messages=body.chat_messages,
            table_id=body.table_id,
            source="predict",
        )
        if payload is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Could not resolve teams for match_id={body.match_id}. "
                    "Pass match_data with home_team and away_team explicitly."
                ),
            )
        return payload

    # ---------- bundle endpoints (reduce frontend chattiness) ----------

    @app.get("/api/v1/dashboard")
    def get_dashboard(tournament_id: str = "fifa-world-cup-2026"):
        """One-call home-page payload: leaderboard + recent runs + stats."""
        return build_dashboard_endpoint(_all_runs(), tournament_id=tournament_id)

    @app.get("/api/v1/matches/{match_id}/state")
    def get_match_state(match_id: str):
        """One-call match drill-down: predictions + discussion + bets + live state."""
        # Best-effort live state from soccersapi (mock-safe)
        live_state = None
        live_events = []
        try:
            from aegeanbench.sports.sources.soccersapi_live import SoccersAPILiveClient
            client = SoccersAPILiveClient()
            matches = client.fetch_live_matches()
            for m in matches:
                if m.match_id == match_id:
                    live_state = m.to_dict()
                    live_events = [e.to_dict() for e in client.fetch_match_events(match_id)]
                    break
        except Exception as e:
            logger.warning("live state fetch failed for %s: %s", match_id, e)
        payload = build_match_state_endpoint(
            _all_runs(),
            match_id,
            live_state=live_state,
            live_events=live_events,
        )
        if payload is None:
            raise HTTPException(status_code=404, detail=f"no data for match {match_id}")
        return payload

    # ---------------------- Arena: model head-to-head ----------------------
    # Additive only — these do not touch any existing endpoint. For each
    # upcoming match they compare aegean-consensus (our enriched, data-
    # grounded model) against the configured benchmark competitors
    # (BENCHMARK_MODELS in .env), each returning the same rich schema.

    def _arena_service():
        svc = getattr(app.state, "arena_service", None)
        if svc is None:
            from aegeanbench.sports.arena.service import ArenaService
            from aegeanbench.sports.gateway import SportsDataGateway
            gw = getattr(app.state, "gateway", None)
            if gw is None:
                gw = SportsDataGateway(mock=False)
                app.state.gateway = gw
            # Share the /predict cache so the arena's aegean consensus run
            # also warms /predict (one run, both consumers).
            svc = ArenaService(gateway=gw, prediction_cache=app.state.prediction_cache)
            app.state.arena_service = svc
        return svc

    @app.get("/api/v1/arena/models")
    def get_arena_models():
        """
        The arena line-up: aegean-consensus + configured benchmark models.
        Cheap (config only, no LLM) — front-end uses it to render the model
        list / columns before any prediction is computed.
        """
        from aegeanbench.sports.arena.config import arena_roster
        return {"_meta": {"endpoint": "GET /api/v1/arena/models"}, "models": arena_roster()}

    @app.get("/api/v1/arena/upcoming")
    def get_arena_upcoming(lang: str = "en", hours: int = 72):
        """
        Upcoming matches with each model's cached prediction summary.
        Never computes — models not yet run show status="pending".
        """
        return _arena_service().upcoming(lang=lang, hours=hours)

    @app.get("/api/v1/arena/matches/{match_id}")
    async def get_arena_match(
        match_id: str,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        kickoff_at: Optional[str] = None,
        venue: Optional[str] = None,
        lang: str = "en",
        refresh: bool = False,
    ):
        """
        Full side-by-side comparison for one match (all models, rich
        schema). Computes + caches on first call; LLM work runs in a
        worker thread so the event loop stays responsive.
        """
        svc = _arena_service()
        payload = await asyncio.to_thread(
            svc.compute_match,
            match_id=match_id, home_team=home_team, away_team=away_team,
            kickoff_iso=kickoff_at, venue=venue, lang=lang, use_cache=not refresh,
        )
        return payload

    @app.get("/api/v1/arena/matches/{match_id}/strategy")
    def get_match_strategy(match_id: str, home_team: Optional[str] = None,
                           away_team: Optional[str] = None, lang: str = "en"):
        """
        Polymarket value-bet strategy for one match: aegean's cached
        probabilities vs the live Polymarket market-winner odds (edge +
        Kelly). aegean-only. Degrades to model-only when no market exists.
        """
        from aegeanbench.sports.arena.polymarket import PolymarketClient, PMMarket
        from aegeanbench.sports.arena import strategy as strat
        cached = app.state.prediction_cache.get_latest_for_match(match_id, lang=lang)
        model_probs = {"home_win": 0, "draw": 0, "away_win": 0}
        if cached:
            p = cached.get("prediction") or {}
            model_probs = {"home_win": p.get("p_home_win", 0), "draw": p.get("p_draw", 0),
                           "away_win": p.get("p_away_win", 0)}
        if home_team and away_team:
            pm = PolymarketClient().match_market(home_team, away_team)
        else:
            pm = PMMarket(status="none", kind="match",
                          reason="pass home_team & away_team to look up the Polymarket market")
        result = strat.match_strategy(model_probs, pm)
        result["_meta"] = {"endpoint": "GET /api/v1/arena/matches/{match_id}/strategy",
                           "match_id": match_id, "model_available": bool(cached)}
        return result

    @app.on_event("startup")
    async def _start_arena_warmer():
        if os.getenv("ARENA_WARMER_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.warning("ARENA_WARMER_DISABLED is set — arena warmer not started")
            return
        logger.warning("startup hook: building ArenaWarmer")
        # Pre-compute the full arena (all models) at T-24h / T-1h so the
        # upcoming list shows real predictions. Own scheduler instance so
        # its once-per-checkpoint state is independent of the PrematchWarmer.
        from aegeanbench.sports.arena.warmer import ArenaWarmer
        from aegeanbench.sports.live.scheduler import MatchEventScheduler
        warmer = ArenaWarmer(
            arena_service=_arena_service(),
            scheduler=MatchEventScheduler(),
            lang=os.getenv("AEGEAN_LIVE_LANG", "zh"),
        )
        warmer.start()
        app.state.arena_warmer = warmer

    # ---------------------- Arena: full tournament forecast ----------------------

    def _tournament_service():
        svc = getattr(app.state, "tournament_service", None)
        if svc is None:
            from aegeanbench.sports.arena.tournament.service import TournamentService
            from aegeanbench.sports.gateway import SportsDataGateway
            gw = getattr(app.state, "gateway", None)
            if gw is None:
                gw = SportsDataGateway(mock=False)
                app.state.gateway = gw
            svc = TournamentService(gateway=gw)
            app.state.tournament_service = svc
        return svc

    @app.get("/api/v1/arena/tournament")
    def get_tournament(lang: str = "en"):
        """All models' champion / runner-up / third (cached; uncomputed = pending)."""
        return _tournament_service().list_models(lang=lang)

    @app.get("/api/v1/arena/tournament/actual")
    def get_tournament_actual(lang: str = "en"):
        """The factual board: real group tables, results, and top scorers."""
        return _tournament_service().get_actual(lang=lang)

    @app.get("/api/v1/arena/tournament/strategy")
    def get_tournament_strategy(market: str = "champion", lang: str = "en"):
        """
        Polymarket outright strategy: aegean's per-team champion%/advance%
        vs the live Polymarket winner / to-advance market (edge + Kelly per
        team). market=champion|qualification. aegean-only; model-only when
        no market exists.
        """
        from aegeanbench.sports.arena.polymarket import PolymarketClient
        from aegeanbench.sports.arena import strategy as strat
        svc = _tournament_service()
        probs = svc.team_probabilities(lang=lang)
        client = PolymarketClient()
        if market == "qualification":
            pm = client.qualification_market()
            model = probs.get("advance", {})
        else:
            market = "champion"
            pm = client.champion_market()
            model = probs.get("champion", {})
        result = strat.outright_strategy(model, pm, kind=market)
        result["_meta"] = {"endpoint": "GET /api/v1/arena/tournament/strategy", "market": market}
        return result

    @app.get("/api/v1/arena/tournament/{runner_id}")
    async def get_tournament_model(runner_id: str, lang: str = "en", refresh: bool = False):
        """One model's full tournament forecast (bracket + tables + scorers)."""
        svc = _tournament_service()
        payload = await asyncio.to_thread(svc.compute_model, runner_id, lang, not refresh)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"unknown runner {runner_id}")
        return payload

    @app.on_event("startup")
    async def _start_tournament_warmer():
        if os.getenv("TOURNAMENT_WARMER_DISABLED", "").lower() in ("1", "true", "yes"):
            logger.warning("TOURNAMENT_WARMER_DISABLED is set — tournament warmer not started")
            return
        logger.warning("startup hook: building TournamentWarmer")
        from aegeanbench.sports.arena.tournament.warmer import TournamentWarmer
        warmer = TournamentWarmer(
            tournament_service=_tournament_service(),
            lang=os.getenv("AEGEAN_LIVE_LANG", "zh"),
        )
        warmer.start()
        app.state.tournament_warmer = warmer

    @app.get("/api/v1/health")
    def health():
        return {
            "status": "ok",
            "runs_dir": str(runs_root),
            "qa_handler": qa_handler is not None,
            "live_hub_subscribers": live_hub.subscriber_count(),
        }

    # ---------------------- V2: @-mention agent Q&A ----------------------

    @app.post("/api/v1/agents/{agent_id}/answer")
    async def post_agent_answer(agent_id: str, body: AgentAnswerRequest):
        """
        @-mention single agent Q&A handler.

        Body schema:
          question        required, the user's prompt
          match_id        optional, used for telemetry / context lookup
          match_context   optional, prebuilt match-context string
          user_name       optional, used to address the user in the reply
          recent_messages optional, last few chat turns for session-only memory
        """
        if app.state.qa_handler is None:
            raise HTTPException(
                status_code=503,
                detail="Q&A handler not configured; pass qa_handler= to create_app()",
            )
        if not body.question:
            raise HTTPException(status_code=400, detail="'question' is required")

        # Build a role-specific match brief unless the caller supplied
        # match_context already. Market specialist gets odds, player
        # specialist gets lineups, etc. The fetch is cached per
        # (match_id, role) for 60s so chat-room bursts don't hammer
        # upstream APIs.
        match_context = body.match_context
        if not match_context and body.match_id:
            try:
                from aegeanbench.sports.reporter.match_brief import build_brief_for_role
                match_context = build_brief_for_role(
                    role=agent_id,
                    match_id=body.match_id,
                    match_data=body.match_data,
                )
            except Exception as exc:
                logger.warning("brief build failed: %s", exc)
                match_context = None

        # Reply language priority:
        #   1. body.lang explicit from front-end (preferred)
        #   2. detected from question text (CJK -> zh)
        #   3. recent_messages as weak secondary signal
        from aegeanbench.sports.lang import detect_from_signals
        secondary = [m.get("text", "") for m in (body.recent_messages or [])]
        lang = body.lang or detect_from_signals(body.question, secondary)

        response = await app.state.qa_handler.answer(
            agent_id=agent_id,
            question=body.question,
            match_context=match_context,
            recent_messages=body.recent_messages,
            user_name=body.user_name,
            room_id=body.room_id,
            lang=lang,
        )
        return response.to_dict()

    # ---------------------- V2: WebSocket live updates ----------------------

    @app.websocket("/ws/predictions")
    async def ws_predictions(websocket: WebSocket):
        """
        Push channel for prediction updates.

        After accept the server sends a small {"type":"hello"} envelope,
        then forwards every prediction event published via
        app.state.live_hub.publish("predictions", payload).
        """
        await websocket.accept()
        queue = await live_hub.subscribe("predictions")
        try:
            await websocket.send_json({"type": "hello", "channel": "predictions"})
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await live_hub.unsubscribe("predictions", queue)

    @app.websocket("/ws/matches/{match_id}")
    async def ws_match(websocket: WebSocket, match_id: str):
        """
        Per-match live channel.

        Sends:
          - score updates  ({"type":"score", "home":2, "away":1, "minute":67})
          - re-evaluations ({"type":"prediction_update", "runner_id":..., ...})
          - final settle   ({"type":"settled", "outcome":"home_win"})
        """
        await websocket.accept()
        channel = f"match:{match_id}"
        queue = await live_hub.subscribe(channel)
        try:
            await websocket.send_json({"type": "hello", "channel": channel})
            while True:
                payload = await queue.get()
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            await live_hub.unsubscribe(channel, queue)

    # Convenience admin endpoint to test the live hub from curl during dev
    @app.post("/api/v1/_internal/publish")
    async def publish_test(body: PublishRequest):
        n = await live_hub.publish(body.channel, body.payload)
        return {"published_to": n}

    # ---------- user-initiated divination ----------

    @app.get("/api/v1/divination/tarot/deck")
    def get_tarot_deck():
        """The full 78-card tarot deck. Frontend renders this as the picker UI."""
        from aegeanbench.sports.divination import get_tarot_catalog
        return get_tarot_catalog()

    @app.get("/api/v1/divination/iching/hexagrams")
    def get_iching_hexagrams():
        """All 64 I Ching hexagrams. Frontend renders this as the picker UI."""
        from aegeanbench.sports.divination import get_iching_catalog
        return get_iching_catalog()

    @app.post("/api/v1/divination")
    async def post_divination(body: DivinationRequest):
        """
        User-initiated tarot or 周易 reading. User picks the cards /
        hexagram on the frontend; we just interpret. Returns a reading
        plus an optional outcome lean (not the same as the consensus
        prediction).
        """
        from aegeanbench.sports.divination import perform_divination
        if body.type not in ("tarot", "iching"):
            raise HTTPException(
                status_code=400,
                detail="type must be 'tarot' or 'iching'",
            )
        # Decide reading language: caller can pin via body.lang, else
        # infer from the team names that were sent in.
        from aegeanbench.sports.lang import detect_from_signals
        lang = body.lang or detect_from_signals(
            body.home_team, [body.away_team or ""]
        )
        try:
            result = perform_divination(
                div_type=body.type,
                match_id=body.match_id,
                home_team=body.home_team or "Home",
                away_team=body.away_team or "Away",
                table_id=body.table_id,
                card_indices=body.card_indices,
                hexagram_index=body.hexagram_index,
                llm_call=None,    # use template fallback for now;
                                  # wire to OpenAI later if needed
                lang=lang,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return result.to_dict()

    # ---------- ops: cache stats ----------

    @app.get("/api/v1/_stats/cache")
    def get_cache_stats():
        """Cheap diagnostic for ops/dev — predict-cache hit rate."""
        return app.state.prediction_cache.stats()

    # ---------- chat signal (chat-service notifying us of room activity) ----------

    @app.post("/api/v1/chat/signal")
    async def post_chat_signal(body: ChatSignalRequest):
        """
        Chat service tells us a room has crossed an activity threshold.

        We forward the signal to MatchEventScheduler which decides whether
        to trigger a fresh consensus run. The actual consensus invocation
        happens in a background worker - here we just record the trigger
        and emit a WebSocket event so any dashboards can react.

        Returns a flag indicating whether we accepted the signal as a
        consensus trigger (subject to throttling), so the chat service
        can stop pinging us when we're already busy.
        """
        trigger = app.state.scheduler.on_chat_heat(
            match_id=body.match_id,
            message_count_in_window=body.message_count,
            strong_sentiment=body.strong_sentiment,
        )
        if trigger:
            payload = {
                "type": "chat_signal_accepted",
                "match_id": body.match_id,
                "room_id": body.room_id,
                "trigger": trigger.to_dict(),
            }
            await live_hub.publish(f"match:{body.match_id}", payload)
        return {
            "accepted": trigger is not None,
            "trigger": trigger.to_dict() if trigger else None,
            "match_id": body.match_id,
            "room_id": body.room_id,
        }

    return app


# Module-level app for `uvicorn aegeanbench.sports.reporter.server:app`
try:
    # Build a default LocalQAHandler so @-mention works out of the box.
    # The handler reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    # from the environment and falls back to a deterministic reply when
    # no key is configured.
    from aegeanbench.sports.reporter.local_qa import LocalQAHandler
    app = create_app(qa_handler=LocalQAHandler())
except RuntimeError as e:
    # FastAPI not installed; module is still importable
    app = None  # type: ignore[assignment]
    logger.info("reporter.server.app not created: %s", e)
