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
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel

    class AgentAnswerRequest(BaseModel):
        """JSON body schema for POST /api/v1/agents/{id}/answer."""
        question: str
        match_id: Optional[str] = None
        match_context: Optional[str] = None
        user_name: Optional[str] = None
        recent_messages: Optional[List[Dict[str, Any]]] = None

    class PublishRequest(BaseModel):
        channel: str = "predictions"
        payload: Dict[str, Any] = {}
except ImportError:
    AgentAnswerRequest = None  # type: ignore[assignment]
    PublishRequest = None  # type: ignore[assignment]

from aegeanbench.sports.orchestrator.persistence import (
    DEFAULT_RUNS_DIR,
    list_runs,
    load_run,
)
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
    live_hub = LiveHub()
    app.state.live_hub = live_hub
    app.state.qa_handler = qa_handler

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
        response = await app.state.qa_handler.answer(
            agent_id=agent_id,
            question=body.question,
            match_context=body.match_context,
            recent_messages=body.recent_messages,
            user_name=body.user_name,
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

    return app


# Module-level app for `uvicorn aegeanbench.sports.reporter.server:app`
try:
    app = create_app()
except RuntimeError as e:
    # FastAPI not installed; module is still importable
    app = None  # type: ignore[assignment]
    logger.info("reporter.server.app not created: %s", e)
