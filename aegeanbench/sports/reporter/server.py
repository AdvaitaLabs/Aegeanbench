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
        from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse
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

        # Build the Match shell from the caller's body, then let the
        # SportsDataGateway pull odds / lineups / h2h / xG / weather
        # from the real adapters. The enriched ctx is what makes the
        # consensus prompt non-empty.
        from aegeanbench.sports.predictors.aegean import AegeanPredictor
        from aegeanbench.sports.gateway import MatchContext, SportsDataGateway
        from aegeanbench.sports.models import (
            CompetitionStage, Match, Team,
        )
        from datetime import datetime as _dt

        md = body.match_data or MatchDataInput()
        kickoff = (
            _dt.fromisoformat(md.kickoff_at.replace("Z", "+00:00"))
            if md.kickoff_at else _dt.now()
        )

        # FIFA code lookup table for the names the front-end is most
        # likely to send. Falls back to the first 3 chars when unknown.
        _FIFA = {
            "Mexico": "MEX", "South Africa": "RSA", "United States": "USA",
            "Argentina": "ARG", "Brazil": "BRA", "France": "FRA",
            "Germany": "GER", "Spain": "ESP", "England": "ENG",
            "Portugal": "POR", "Netherlands": "NED", "Italy": "ITA",
            "Belgium": "BEL", "Croatia": "CRO", "Japan": "JPN",
            "Korea Republic": "KOR", "South Korea": "KOR",
            "Morocco": "MAR", "Saudi Arabia": "KSA",
            "Canada": "CAN", "Australia": "AUS",
        }
        def _fifa(name: Optional[str], default: str) -> str:
            if not name:
                return default
            return _FIFA.get(name, name[:3].upper())

        match = Match(
            match_id=body.match_id,
            competition="FIFA World Cup 2026",
            stage=CompetitionStage.GROUP,
            kickoff_at=kickoff,
            home_team=Team(
                fifa_code=_fifa(md.home_team, "BRA"),
                name=md.home_team or "Home",
            ),
            away_team=Team(
                fifa_code=_fifa(md.away_team, "ARG"),
                name=md.away_team or "Away",
            ),
            venue=md.venue,
        )

        # Singleton gateway so adapter HTTP clients and the file cache
        # stay warm across requests. Lazy-built on first /predict.
        gw = getattr(app.state, "gateway", None)
        if gw is None:
            gw = SportsDataGateway(mock=False)
            app.state.gateway = gw

        try:
            ctx = gw.build_context(match)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gateway.build_context failed for %s: %s", body.match_id, exc)
            ctx = MatchContext(
                match=match, home_history=[], away_history=[], h2h=[],
                home_xg_profile={}, away_xg_profile={},
            )

        # Carry chat snapshot via ctx.chat_summary (NOT on match.h2h_last5
        # — that field belongs to real head-to-head data and is now
        # populated by the gateway).
        if body.chat_messages:
            ctx.chat_summary = "\n".join(
                f"  - {m.user_name}: {m.text}"
                for m in body.chat_messages[-30:]
            )

        # Reply language priority:
        #   1. body.lang (front-end's explicit choice from user locale)
        #   2. match_data.lang (legacy alias)
        #   3. detected from chat snippets
        #   4. English fallback
        from aegeanbench.sports.lang import detect_from_signals
        explicit_lang = body.lang or (
            body.match_data.model_dump().get("lang") if body.match_data else None
        )
        chat_texts = [m.text for m in (body.chat_messages or [])]
        lang = explicit_lang or detect_from_signals(None, chat_texts) or "en"

        predictor = AegeanPredictor(agent_types=requested)
        prediction = predictor.predict(ctx, lang=lang)

        return {
            "_meta": {
                "endpoint": "POST /api/v1/predict",
                "description": "One-shot consensus prediction for a user-defined table",
            },
            "table_id": body.table_id,
            "match_id": body.match_id,
            "agents_used": requested,
            "prediction": {
                "p_home_win": prediction.p_home_win,
                "p_draw": prediction.p_draw,
                "p_away_win": prediction.p_away_win,
                "confidence": prediction.confidence,
                "rationale": prediction.rationale,
                "latency_ms": prediction.latency_ms,
                "tokens_used": prediction.tokens_used,
            },
            "discussion": (prediction.metadata or {}).get("discussion"),
        }

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

    # ---------- admin: runtime-tunable global prompt ----------

    def _check_admin(request_token: Optional[str]) -> None:
        """Block writes when the caller doesn't present the admin token."""
        expected = os.getenv("ADMIN_TOKEN", "")
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="ADMIN_TOKEN is not configured on the server",
            )
        if request_token != expected:
            raise HTTPException(status_code=401, detail="invalid admin token")

    @app.get("/api/v1/admin/prompts/global")
    def get_global_prompt():
        """Return the active global prompt addendum + last 5 versions."""
        from aegeanbench.sports.prompts.runtime_store import get_full_state
        return get_full_state()

    @app.post("/api/v1/admin/prompts/global")
    async def set_global_prompt(request: Request):
        """Replace the global prompt. Header X-Admin-Token required."""
        _check_admin(request.headers.get("X-Admin-Token"))
        body = await request.json()
        prompt = (body or {}).get("prompt", "")
        if not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="`prompt` must be a string")
        updated_by = (body or {}).get("updated_by") or "admin"
        from aegeanbench.sports.prompts.runtime_store import set_prompt
        return set_prompt(prompt, updated_by=updated_by)

    @app.post("/api/v1/admin/prompts/global/rollback")
    async def rollback_global_prompt(request: Request):
        """Revert to the most-recent history entry. Header X-Admin-Token required."""
        _check_admin(request.headers.get("X-Admin-Token"))
        from aegeanbench.sports.prompts.runtime_store import rollback_to_previous
        return rollback_to_previous()

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page():
        """Tiny self-contained HTML form for the product team."""
        from aegeanbench.sports.prompts.admin_page import ADMIN_HTML
        return ADMIN_HTML

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
