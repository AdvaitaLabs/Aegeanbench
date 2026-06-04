"""
Build all 6 endpoint JSON files from persisted runs.

This is the "give the front-end team a folder of JSON" delivery mode -
the simplest possible integration that lets the product team work
without needing a running server.

Usage:
    from aegeanbench.sports.reporter import build_all_endpoints
    build_all_endpoints(output_dir=Path("./mock_api"))

Reads from ~/.aegeanbench/worldcup_runs/ by default; writes a fixed set
of JSON files into output_dir:

    output_dir/
        runners.json
        tournaments.json
        leaderboard.json
        runs/<run_id>.json
        runs/<run_id>/matches/<match_id>.json
        runners/<runner_id>/card.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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


def _write_json(path: Path, data: Any) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _collect_runs(runs_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Load all run bundles under runs_dir."""
    base = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    runs: List[Dict[str, Any]] = []
    if not base.exists():
        return runs
    for entry in base.iterdir():
        if entry.is_dir() and (entry / "manifest.json").exists():
            try:
                runs.append(load_run(entry))
            except Exception as e:
                logger.warning("skipping unreadable run %s: %s", entry.name, e)
    return runs


def build_all_endpoints(
    output_dir: Path,
    runs_dir: Optional[Path] = None,
    tournament_id: str = "fifa-world-cup-2026",
) -> Dict[str, int]:
    """
    Render every endpoint JSON into output_dir.

    Args:
        output_dir: where to write the JSON files
        runs_dir: source of pipeline runs. Defaults to DEFAULT_RUNS_DIR.
        tournament_id: filter / label for the leaderboard

    Returns:
        Counts dict: {endpoint_kind: number_of_files_written}.
    """
    output_dir = Path(output_dir)
    runs = _collect_runs(runs_dir)
    counts = {"runners": 0, "tournaments": 0, "leaderboard": 0, "runs": 0,
              "match_details": 0, "runner_cards": 0}

    # Static endpoints
    _write_json(output_dir / "runners.json", build_runners_endpoint())
    counts["runners"] = 1
    _write_json(output_dir / "tournaments.json", build_tournaments_endpoint())
    counts["tournaments"] = 1

    # Leaderboard aggregates across all runs
    _write_json(
        output_dir / "leaderboard.json",
        build_leaderboard_endpoint(runs, tournament_id=tournament_id),
    )
    counts["leaderboard"] = 1

    # Per-run + per-match drill-downs
    runs_out = output_dir / "runs"
    matches_root = output_dir / "matches"
    for run in runs:
        manifest = run.get("manifest", {})
        run_id = manifest.get("run_id")
        if not run_id:
            continue
        _write_json(runs_out / f"{run_id}.json", build_run_endpoint(run))
        counts["runs"] += 1

        # Match drill-downs: find unique match_ids in this run's predictions
        match_ids = _collect_match_ids(run)
        for mid in match_ids:
            payload = build_match_detail_endpoint(run, mid)
            if payload is None:
                continue
            _write_json(matches_root / run_id / f"{mid}.json", payload)
            counts["match_details"] += 1

    # Per-runner cards aggregated across all runs
    runner_cards_root = output_dir / "runners"
    for runner_id in RUNNER_REGISTRY:
        card = build_runner_card_endpoint(runs, runner_id)
        _write_json(runner_cards_root / runner_id / "card.json", card)
        counts["runner_cards"] += 1

    return counts


def _collect_match_ids(run: Dict[str, Any]) -> List[str]:
    """Return the unique match_ids referenced in a run's predictions."""
    seen: List[str] = []
    seen_set = set()
    for preds in run.get("predictions", {}).values():
        for p in preds:
            mid = p.get("match_id")
            if mid and mid not in seen_set:
                seen.append(mid)
                seen_set.add(mid)
    return seen
