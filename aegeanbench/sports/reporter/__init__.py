"""Reporter: convert persisted runs into product-facing JSON endpoints."""

from aegeanbench.sports.reporter.builder import build_all_endpoints
from aegeanbench.sports.reporter.endpoints import (
    RUNNER_REGISTRY,
    build_leaderboard_endpoint,
    build_match_detail_endpoint,
    build_run_endpoint,
    build_runner_card_endpoint,
    build_runners_endpoint,
    build_tournaments_endpoint,
)

__all__ = [
    "build_all_endpoints",
    "build_runners_endpoint",
    "build_tournaments_endpoint",
    "build_leaderboard_endpoint",
    "build_run_endpoint",
    "build_match_detail_endpoint",
    "build_runner_card_endpoint",
    "RUNNER_REGISTRY",
]
