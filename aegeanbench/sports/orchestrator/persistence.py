"""
Persistence layer for World Cup pipeline runs.

A "run" is one execution of the pipeline: a set of predictions across
fixtures, the betting portfolio it produced, and any retrospective
evaluation against actual results. Runs are stored as JSON files under
~/.aegeanbench/worldcup_runs/ so they survive process restarts and can
be served back to the dashboard / front-end.

Storage layout:
    ~/.aegeanbench/worldcup_runs/
        run_<timestamp>_<short_hash>/
            manifest.json           run metadata
            predictions.json        all predictions across runners and matches
            portfolio.json          betting portfolio (if betting was enabled)
            evaluation.json         retrospective evaluation (filled later)

The format intentionally uses JSON so anyone can `cat` a run, and so the
Day 6 Reporter can read them without depending on this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aegeanbench.sports.betting import BettingPortfolio, SizedBet
from aegeanbench.sports.models import Prediction
from aegeanbench.sports.orchestrator.evaluator import (
    BettingEvaluation,
    RunnerEvaluation,
)


DEFAULT_RUNS_DIR = Path.home() / ".aegeanbench" / "worldcup_runs"


def _short_hash(s: str, length: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def _make_run_id(label: Optional[str] = None) -> str:
    """Generate a human-readable run_id: run_<YYYYmmdd_HHMMSS>_<hash>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    salt = label or ts
    return f"run_{ts}_{_short_hash(salt)}"


# ----------------------------- manifest -----------------------------


@dataclass
class RunManifest:
    """Metadata for a single pipeline run."""
    run_id: str
    label: str
    created_at: str
    competition: str
    n_matches: int
    runner_ids: List[str]
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------- write -----------------------------


def _bet_to_dict(bet: SizedBet) -> Dict[str, Any]:
    return bet.to_dict()


def _portfolio_to_dict(p: BettingPortfolio) -> Dict[str, Any]:
    return {
        "config": asdict(p.config),
        "bets": [_bet_to_dict(b) for b in p.bets],
        "skipped_candidates": [c.to_dict() for c in p.skipped_candidates],
        "total_stake": p.total_stake,
        "total_expected_return": p.total_expected_return,
        "exposure_pct": p.exposure_pct,
        "summary": p.summary(),
    }


def save_run(
    predictions_by_runner: Dict[str, List[Prediction]],
    portfolio: Optional[BettingPortfolio] = None,
    competition: str = "FIFA World Cup 2026",
    label: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    runs_dir: Optional[Path] = None,
) -> Path:
    """
    Persist a run to disk and return the run directory path.

    Args:
        predictions_by_runner: {runner_id: [Prediction, ...]}
        portfolio: optional BettingPortfolio
        competition: name of the competition (for the manifest)
        label: human-readable label (defaults to timestamp)
        config: arbitrary run config dict to record in the manifest
        runs_dir: override the default runs directory

    Returns:
        Path to the created run directory.
    """
    base = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    base.mkdir(parents=True, exist_ok=True)
    run_id = _make_run_id(label)
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    # Manifest
    n_matches = max((len(v) for v in predictions_by_runner.values()), default=0)
    manifest = RunManifest(
        run_id=run_id,
        label=label or run_id,
        created_at=datetime.now().isoformat(),
        competition=competition,
        n_matches=n_matches,
        runner_ids=list(predictions_by_runner.keys()),
        config=config or {},
    )
    _write_json(run_dir / "manifest.json", manifest.to_dict())

    # Predictions
    serialised: Dict[str, List[Dict[str, Any]]] = {
        runner_id: [p.to_dict() for p in preds]
        for runner_id, preds in predictions_by_runner.items()
    }
    _write_json(run_dir / "predictions.json", serialised)

    # Portfolio
    if portfolio is not None:
        _write_json(run_dir / "portfolio.json", _portfolio_to_dict(portfolio))

    return run_dir


def save_evaluation(
    run_dir: Path,
    runner_evaluations: Dict[str, RunnerEvaluation],
    betting_evaluation: Optional[BettingEvaluation] = None,
) -> Path:
    """Attach a retrospective evaluation to an existing run directory."""
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    payload = {
        "evaluated_at": datetime.now().isoformat(),
        "runner_evaluations": {
            runner_id: ev.to_dict()
            for runner_id, ev in runner_evaluations.items()
        },
        "betting_evaluation": (
            betting_evaluation.to_dict() if betting_evaluation else None
        ),
    }
    path = run_dir / "evaluation.json"
    _write_json(path, payload)
    return path


# ----------------------------- read -----------------------------


def list_runs(runs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return manifests of all known runs, newest first."""
    base = Path(runs_dir) if runs_dir else DEFAULT_RUNS_DIR
    if not base.exists():
        return []
    out: List[Dict[str, Any]] = []
    for child in base.iterdir():
        manifest_path = child / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open() as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def load_run(run_dir: Path) -> Dict[str, Any]:
    """Load all artifacts of a run as a single dict (for serving/reporting)."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    bundle: Dict[str, Any] = {}
    for fname in ("manifest", "predictions", "portfolio", "evaluation"):
        path = run_dir / f"{fname}.json"
        if path.exists():
            with path.open() as f:
                bundle[fname] = json.load(f)
    return bundle


# ----------------------------- internal -----------------------------


def _write_json(path: Path, data: Any) -> None:
    """Atomic JSON write: temp file then rename."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)
