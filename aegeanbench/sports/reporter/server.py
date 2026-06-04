"""
Optional FastAPI server that exposes the 6 reporter endpoints over HTTP.

This is a thin shim: each route calls one builder function from
endpoints.py. The server has no state - it reads runs from disk on every
request, which is fine for the sprint scale (a few dozen runs).

Usage:
    uvicorn aegeanbench.sports.reporter.server:app --port 8200

FastAPI is loaded lazily so this module can be imported without the
dependency installed; the actual create_app() call will fail loudly with
an install hint if FastAPI is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from aegeanbench.sports.orchestrator.persistence import (
    DEFAULT_RUNS_DIR,
    list_runs,
    load_run,
)
from aegeanbench.sports.reporter.endpoints import (
    RUNNER_REGISTRY,
    build_leaderboard_endpoint,
    build_match_detail_endpoint,
    build_run_endpoint,
    build_runner_card_endpoint,
    build_runners_endpoint,
    build_tournaments_endpoint,
)

logger = logging.getLogger(__name__)


def create_app(runs_dir: Optional[Path] = None):
    """
    Build a FastAPI application.

    Args:
        runs_dir: override the persisted-run directory (default ~/.aegeanbench/...).

    Returns:
        FastAPI app instance ready to be served by uvicorn.

    Raises:
        RuntimeError: when fastapi isn't installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise RuntimeError(
            "fastapi is required to use reporter.server. "
            "Install with: pip install fastapi uvicorn"
        ) from e

    runs_root = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR

    app = FastAPI(
        title="AegeanBench World Cup Reporter",
        version="0.1.0",
        description="6 endpoints powering the World Cup 2026 product front-end.",
    )

    # Permissive CORS for sprint dev; tighten before production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

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

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "runs_dir": str(runs_root)}

    return app


# Module-level app for `uvicorn aegeanbench.sports.reporter.server:app`
try:
    app = create_app()
except RuntimeError as e:
    # FastAPI not installed; module is still importable
    app = None  # type: ignore[assignment]
    logger.info("reporter.server.app not created: %s", e)
