"""
Pure functions that convert persisted run data into product-facing JSON.

Each function returns a Python dict shaped exactly like the public API
contract. The Day 6 builder writes these to disk; the optional FastAPI
server returns them over HTTP.

Endpoints:
    runners            list of predictors (models) with metadata
    tournaments        list of tournaments / datasets the user can pick
    leaderboard        rankings across runners with brier / hit_rate / ROI
    run                full data for a single pipeline run
    match_detail       drill-down for one match across all predictors
    runner_card        per-predictor profile aggregating across runs

All functions are pure: they take dicts (loaded by persistence.load_run)
and return dicts. Side-effect-free and trivially testable.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


# ----------------------------- runner metadata -----------------------------

# Static metadata served by /runners. Augmented at runtime if evaluation
# data is available.
RUNNER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "elo": {
        "name": "Elo Ratings",
        "type": "classical",
        "provider": "Aegean",
        "is_ours": True,
        "description": "World Football Elo with stage-based K-factor and goal margin multiplier.",
        "cost_per_match_usd": 0.0,
    },
    "dixon_coles": {
        "name": "Dixon-Coles",
        "type": "classical",
        "provider": "Aegean",
        "is_ours": True,
        "description": "Bivariate Poisson with low-scoreline correction (Dixon & Coles, 1997).",
        "cost_per_match_usd": 0.0,
    },
    "gpt-5": {
        "name": "GPT-5",
        "type": "single_llm",
        "provider": "OpenAI",
        "is_ours": False,
        "description": "Single-shot prediction from OpenAI GPT-5.",
        "cost_per_match_usd": 0.04,
    },
    "claude-opus-4-7": {
        "name": "Claude Opus 4.7",
        "type": "single_llm",
        "provider": "Anthropic",
        "is_ours": False,
        "description": "Single-shot prediction from Anthropic Claude Opus 4.7.",
        "cost_per_match_usd": 0.06,
    },
    "deepseek-v3": {
        "name": "DeepSeek V3",
        "type": "single_llm",
        "provider": "DeepSeek",
        "is_ours": False,
        "description": "Single-shot prediction from DeepSeek V3.",
        "cost_per_match_usd": 0.001,
    },
    "aegean": {
        "name": "Aegean Consensus",
        "type": "consensus",
        "provider": "Aegean",
        "is_ours": True,
        "description": "Multi-agent consensus across 7 specialist agents (stats / player / strategy / market / news / occult / chat).",
        "cost_per_match_usd": 0.11,
    },
    "random": {
        "name": "Random Baseline",
        "type": "baseline",
        "provider": "Aegean",
        "is_ours": True,
        "description": "Uniform random outcome. Floor for benchmark sanity.",
        "cost_per_match_usd": 0.0,
    },
}


def build_runners_endpoint() -> Dict[str, Any]:
    """Static-ish list of available predictors."""
    return {
        "_meta": {
            "endpoint": "GET /api/v1/runners",
            "description": "Available predictors for the World Cup benchmark",
        },
        "runners": [
            {"id": rid, **meta} for rid, meta in RUNNER_REGISTRY.items()
        ],
    }


# ----------------------------- tournaments -----------------------------


def build_tournaments_endpoint() -> Dict[str, Any]:
    """List of tournaments / fixture sets the user can pick."""
    return {
        "_meta": {
            "endpoint": "GET /api/v1/tournaments",
            "description": "Available tournaments / datasets",
        },
        "tournaments": [
            {
                "id": "fifa-world-cup-2026",
                "name": "FIFA World Cup 2026",
                "asset_class": "sports_football",
                "starts_at": "2026-06-11T00:00:00Z",
                "ends_at": "2026-07-19T00:00:00Z",
                "n_matches": 64,
                "data_sources": ["football-data.org", "soccersapi", "fbref"],
                "is_active": True,
                "description": "48-team World Cup expanded format, 64 matches across group + knockout stages.",
            },
            {
                "id": "uefa-euro-2024-replay",
                "name": "UEFA Euro 2024 Replay",
                "asset_class": "sports_football",
                "starts_at": "2024-06-14T00:00:00Z",
                "ends_at": "2024-07-14T00:00:00Z",
                "n_matches": 51,
                "data_sources": ["football-data.org", "fbref"],
                "is_active": False,
                "description": "Historical replay dataset for offline model calibration.",
            },
        ],
    }


# ----------------------------- leaderboard -----------------------------


def build_leaderboard_endpoint(
    runs: Iterable[Dict[str, Any]],
    tournament_id: str = "fifa-world-cup-2026",
) -> Dict[str, Any]:
    """
    Aggregate metrics across all evaluated runs for one tournament.

    Args:
        runs: iterable of full run bundles (output of persistence.load_run)
        tournament_id: filter to this competition / tournament

    Returns:
        Leaderboard rows: one per runner, sorted by composite_score desc.
        Each row includes: brier, hit_rate, roi, n_matches, leakage_warning
        (currently 'low' for all - World Cup has no memorization gap concept).
    """
    # Aggregator: runner_id -> list of per-runner metrics from each run
    accum: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "n_evaluated": 0,
            "sum_brier": 0.0,
            "sum_log_loss": 0.0,
            "sum_hit": 0,
            "sum_confidence": 0.0,
            "n_runs": 0,
            "total_stake": 0.0,
            "total_returned": 0.0,
        }
    )

    for run in runs:
        manifest = run.get("manifest", {})
        # Filter by tournament if specified
        if tournament_id and manifest.get("competition") not in (
            "FIFA World Cup 2026", tournament_id
        ):
            continue
        evaluation = run.get("evaluation") or {}
        for runner_id, rev in (evaluation.get("runner_evaluations") or {}).items():
            accum[runner_id]["n_evaluated"] += rev.get("n_evaluated", 0)
            accum[runner_id]["sum_brier"] += (
                rev.get("mean_brier", 0.0) * rev.get("n_evaluated", 0)
            )
            accum[runner_id]["sum_log_loss"] += (
                rev.get("mean_log_loss", 0.0) * rev.get("n_evaluated", 0)
            )
            accum[runner_id]["sum_hit"] += int(
                rev.get("hit_rate", 0.0) * rev.get("n_evaluated", 0)
            )
            accum[runner_id]["sum_confidence"] += (
                rev.get("mean_confidence", 0.0) * rev.get("n_evaluated", 0)
            )
            accum[runner_id]["n_runs"] += 1
        bet_eval = evaluation.get("betting_evaluation")
        if bet_eval:
            # ROI is attributed to the run's betting_runner_id, recorded in config
            betting_runner = manifest.get("config", {}).get("betting_runner_id")
            if betting_runner and betting_runner in accum:
                accum[betting_runner]["total_stake"] += bet_eval.get("total_stake", 0.0)
                accum[betting_runner]["total_returned"] += bet_eval.get("total_returned", 0.0)

    rows: List[Dict[str, Any]] = []
    for runner_id, stats in accum.items():
        n = stats["n_evaluated"]
        if n == 0:
            continue
        mean_brier = stats["sum_brier"] / n
        mean_log_loss = stats["sum_log_loss"] / n
        hit_rate = stats["sum_hit"] / n
        mean_conf = stats["sum_confidence"] / n
        roi = (
            (stats["total_returned"] - stats["total_stake"]) / stats["total_stake"]
            if stats["total_stake"] > 0
            else 0.0
        )
        rows.append({
            "runner_id": runner_id,
            "runner_name": RUNNER_REGISTRY.get(runner_id, {}).get("name", runner_id),
            "is_ours": RUNNER_REGISTRY.get(runner_id, {}).get("is_ours", False),
            "n_matches": n,
            "mean_brier_score": round(mean_brier, 4),
            "mean_log_loss": round(mean_log_loss, 4),
            "hit_rate": round(hit_rate, 4),
            "mean_confidence": round(mean_conf, 4),
            "roi": round(roi, 4),
            "composite_score": _composite_score(mean_brier, hit_rate, roi),
            "leakage_warning": "low",   # WC has no memorization-gap concept
        })

    rows.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    return {
        "_meta": {
            "endpoint": "GET /api/v1/leaderboard",
            "description": "Aggregate metrics across all evaluated runs",
            "tournament_id": tournament_id,
        },
        "tournament_id": tournament_id,
        "evaluated_at": datetime.now().isoformat(),
        "n_runs_aggregated": sum(1 for r in runs if r.get("evaluation")),
        "rows": rows,
    }


def _composite_score(mean_brier: float, hit_rate: float, roi: float) -> float:
    """
    Combine the three core metrics into a 0-100 composite.

        composite = 100 * (0.40 * (1 - brier/2)        # brier in [0,2]
                         + 0.30 * hit_rate             # already in [0,1]
                         + 0.30 * sigmoid(roi*2))      # roi can be negative
    """
    import math
    brier_score = max(0.0, 1.0 - mean_brier / 2.0)
    roi_score = 1.0 / (1.0 + math.exp(-roi * 2.0))
    return round(100 * (0.40 * brier_score + 0.30 * hit_rate + 0.30 * roi_score), 2)


# ----------------------------- single run -----------------------------


def build_run_endpoint(run: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a single persisted run bundle with the standard envelope."""
    manifest = run.get("manifest", {})
    return {
        "_meta": {
            "endpoint": "GET /api/v1/runs/{run_id}",
            "description": "Full data for one pipeline run",
        },
        "run_id": manifest.get("run_id"),
        "manifest": manifest,
        "predictions": run.get("predictions", {}),
        "portfolio": run.get("portfolio"),
        "evaluation": run.get("evaluation"),
    }


# ----------------------------- match detail -----------------------------


def build_match_detail_endpoint(
    run: Dict[str, Any], match_id: str
) -> Optional[Dict[str, Any]]:
    """
    Drill-down for one match: show every predictor's call and (if available)
    the realised result.
    """
    predictions_by_runner = run.get("predictions", {})
    if not predictions_by_runner:
        return None

    per_predictor: List[Dict[str, Any]] = []
    for runner_id, preds in predictions_by_runner.items():
        match_pred = next((p for p in preds if p.get("match_id") == match_id), None)
        if match_pred is None:
            continue
        per_predictor.append({
            "runner_id": runner_id,
            "runner_name": RUNNER_REGISTRY.get(runner_id, {}).get("name", runner_id),
            "p_home_win": match_pred.get("p_home_win"),
            "p_draw": match_pred.get("p_draw"),
            "p_away_win": match_pred.get("p_away_win"),
            "confidence": match_pred.get("confidence"),
            "rationale": match_pred.get("rationale"),
            "latency_ms": match_pred.get("latency_ms"),
            "tokens_used": match_pred.get("tokens_used"),
        })

    if not per_predictor:
        return None

    # Pull the betting summary for this match
    portfolio = run.get("portfolio") or {}
    bets_on_match: List[Dict[str, Any]] = []
    for bet in portfolio.get("bets", []):
        cand = bet.get("candidate", {})
        if cand.get("match_id") == match_id:
            bets_on_match.append(bet)

    # Pull evaluation per match if available
    evaluation = run.get("evaluation") or {}
    per_match_eval: List[Dict[str, Any]] = []
    for runner_id, rev in (evaluation.get("runner_evaluations") or {}).items():
        for pm in rev.get("per_match", []):
            if pm.get("match_id") == match_id:
                per_match_eval.append({"runner_id": runner_id, **pm})

    return {
        "_meta": {
            "endpoint": "GET /api/v1/runs/{run_id}/matches/{match_id}",
            "description": "Drill-down for one match across all predictors",
        },
        "run_id": run.get("manifest", {}).get("run_id"),
        "match_id": match_id,
        "predictions_per_predictor": per_predictor,
        "bets_on_match": bets_on_match,
        "evaluation_per_predictor": per_match_eval,
    }


# ----------------------------- runner card -----------------------------


def _classify_matches_by_time(
    runs_list: List[Dict[str, Any]],
    now_iso: Optional[str] = None,
    live_window_hours: int = 3,
    upcoming_window_hours: int = 24,
    recent_window_hours: int = 24,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pull match-level info from persisted runs and bucket them into
    live / upcoming / recent for the dashboard.

    A match is identified by match_id within predictions. We use the
    most recent run that mentions each match_id as the source of truth.
    """
    from datetime import datetime, timedelta, timezone

    now = (
        datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
        if now_iso else datetime.now(timezone.utc)
    )
    live_lo = now - timedelta(hours=live_window_hours)
    upcoming_hi = now + timedelta(hours=upcoming_window_hours)
    recent_lo = now - timedelta(hours=recent_window_hours)

    # Aggregate: for each match_id, hold the latest prediction from "aegean"
    # plus the manifest's created_at so we can window them.
    match_index: Dict[str, Dict[str, Any]] = {}

    for run in sorted(
        runs_list,
        key=lambda r: r.get("manifest", {}).get("created_at", ""),
        reverse=True,
    ):
        manifest = run.get("manifest", {})
        run_id = manifest.get("run_id")
        for runner_id, preds in (run.get("predictions") or {}).items():
            for p in preds:
                mid = p.get("match_id")
                if not mid or mid in match_index:
                    continue
                match_index[mid] = {
                    "match_id": mid,
                    "run_id": run_id,
                    "aegean_prediction": None,
                    "predictions_by_runner": {},
                    "kickoff_at": p.get("metadata", {}).get("kickoff_at"),
                }
            # Track per-runner prediction once match is known
            for p in preds:
                mid = p.get("match_id")
                if mid in match_index:
                    if runner_id not in match_index[mid]["predictions_by_runner"]:
                        match_index[mid]["predictions_by_runner"][runner_id] = {
                            "p_home_win": p.get("p_home_win"),
                            "p_draw": p.get("p_draw"),
                            "p_away_win": p.get("p_away_win"),
                            "confidence": p.get("confidence"),
                        }
                    if runner_id == "aegean":
                        match_index[mid]["aegean_prediction"] = {
                            "p_home_win": p.get("p_home_win"),
                            "p_draw": p.get("p_draw"),
                            "p_away_win": p.get("p_away_win"),
                            "confidence": p.get("confidence"),
                            "rationale": p.get("rationale"),
                        }

    live: List[Dict[str, Any]] = []
    upcoming: List[Dict[str, Any]] = []
    recent: List[Dict[str, Any]] = []

    for mid, entry in match_index.items():
        kickoff_str = entry.get("kickoff_at")
        if not kickoff_str:
            # Fall back to run creation time when match metadata missing
            kickoff_str = entry.get("created_at")
        if not kickoff_str:
            continue
        try:
            kickoff = datetime.fromisoformat(str(kickoff_str).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=timezone.utc)

        if live_lo <= kickoff <= now:
            live.append(entry)
        elif now < kickoff <= upcoming_hi:
            upcoming.append(entry)
        elif recent_lo <= kickoff < live_lo:
            recent.append(entry)

    return {"live": live, "upcoming": upcoming, "recent": recent}


def build_dashboard_endpoint(
    runs: Iterable[Dict[str, Any]],
    tournament_id: str = "fifa-world-cup-2026",
    max_recent_runs: int = 5,
) -> Dict[str, Any]:
    """
    One-stop endpoint for the front-end home page.

    Bundles leaderboard + most recent runs + last evaluated runner stats
    into a single payload so the front-end can render the dashboard
    without making 5 parallel calls.

    Returns:
        {
          "leaderboard": <build_leaderboard_endpoint output>,
          "recent_runs": [{run_id, label, created_at, n_matches, runner_ids}, ...],
          "top_runner_card": {runner_id, lifetime: {...}},
          "summary_stats": {
              "total_runs": int,
              "total_predictions": int,
              "total_bets": int,
              "betting_roi_lifetime": float | None,
          }
        }
    """
    runs_list = list(runs)
    leaderboard = build_leaderboard_endpoint(runs_list, tournament_id=tournament_id)

    # Recent runs - just the manifest essentials
    recent: List[Dict[str, Any]] = []
    for run in sorted(
        runs_list,
        key=lambda r: r.get("manifest", {}).get("created_at", ""),
        reverse=True,
    )[:max_recent_runs]:
        m = run.get("manifest", {})
        recent.append({
            "run_id": m.get("run_id"),
            "label": m.get("label"),
            "created_at": m.get("created_at"),
            "n_matches": m.get("n_matches"),
            "runner_ids": m.get("runner_ids", []),
            "has_evaluation": bool(run.get("evaluation")),
        })

    # Lifetime stats - cheap aggregations
    total_predictions = 0
    total_bets = 0
    total_stake = 0.0
    total_returned = 0.0
    for run in runs_list:
        for preds in (run.get("predictions") or {}).values():
            total_predictions += len(preds)
        portfolio = run.get("portfolio") or {}
        bets = portfolio.get("bets") or []
        total_bets += len(bets)
        eval_data = run.get("evaluation") or {}
        bet_eval = eval_data.get("betting_evaluation")
        if bet_eval:
            total_stake += float(bet_eval.get("total_stake", 0))
            total_returned += float(bet_eval.get("total_returned", 0))

    lifetime_roi = (
        (total_returned - total_stake) / total_stake
        if total_stake > 0 else None
    )

    # Top runner card - just the rank-1 entry
    top_card = None
    if leaderboard["rows"]:
        top_runner_id = leaderboard["rows"][0]["runner_id"]
        top_card = build_runner_card_endpoint(runs_list, top_runner_id)

    # Sports-product home page additions
    match_buckets = _classify_matches_by_time(runs_list)

    return {
        "_meta": {
            "endpoint": "GET /api/v1/dashboard",
            "description": "Single-call home-page payload (sports product view)",
        },
        "tournament_id": tournament_id,
        "live_matches": match_buckets["live"],
        "upcoming_matches": match_buckets["upcoming"],
        "recent_results": match_buckets["recent"],
        "leaderboard": leaderboard,
        "recent_runs": recent,
        "top_runner_card": top_card,
        "summary_stats": {
            "total_runs": len(runs_list),
            "total_predictions": total_predictions,
            "total_bets": total_bets,
            "betting_roi_lifetime": (
                round(lifetime_roi, 4) if lifetime_roi is not None else None
            ),
        },
    }


def build_match_state_endpoint(
    runs: Iterable[Dict[str, Any]],
    match_id: str,
    live_state: Optional[Dict[str, Any]] = None,
    live_events: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    One-stop endpoint for the per-match drill-down page.

    Combines:
      - the latest run that includes this match
      - every predictor's call for this match
      - the Aegean discussion trace (multi-round refinement)
      - bets placed on this match
      - per-match evaluation if available
      - optional live state (current score, minute, recent events)

    Front-end uses this single call to render the per-match page.
    """
    runs_list = list(runs)
    # Find the most recent run that includes this match_id
    latest_run = None
    for run in sorted(
        runs_list,
        key=lambda r: r.get("manifest", {}).get("created_at", ""),
        reverse=True,
    ):
        preds = run.get("predictions") or {}
        if any(
            any(p.get("match_id") == match_id for p in pred_list)
            for pred_list in preds.values()
        ):
            latest_run = run
            break

    if latest_run is None:
        return None

    match_detail = build_match_detail_endpoint(latest_run, match_id)
    if match_detail is None:
        return None

    # Pull the discussion trace from the Aegean prediction if present
    discussion: Optional[Dict[str, Any]] = None
    aegean_pred = next(
        (p for p in match_detail["predictions_per_predictor"]
         if p.get("runner_id") == "aegean"),
        None,
    )
    if aegean_pred:
        # The discussion lives in the prediction's metadata.discussion
        # We have to dig into the source run.predictions["aegean"] entry
        aegean_predictions = (latest_run.get("predictions") or {}).get("aegean", [])
        for p in aegean_predictions:
            if p.get("match_id") == match_id:
                discussion = (p.get("metadata") or {}).get("discussion")
                break

    return {
        "_meta": {
            "endpoint": "GET /api/v1/matches/{match_id}/state",
            "description": "Single-call match drill-down with live state",
        },
        "match_id": match_id,
        "run_id": latest_run.get("manifest", {}).get("run_id"),
        "predictions": match_detail["predictions_per_predictor"],
        "discussion": discussion,
        "bets": match_detail.get("bets_on_match", []),
        "evaluation": match_detail.get("evaluation_per_predictor", []),
        "live_state": live_state,
        "live_events": live_events or [],
    }


def build_runner_card_endpoint(
    runs: Iterable[Dict[str, Any]],
    runner_id: str,
) -> Dict[str, Any]:
    """
    Per-predictor profile aggregated across all runs.

    Includes static metadata + historical performance arc + a sample of
    recent predictions for the front-end to render an example reel.
    """
    meta = RUNNER_REGISTRY.get(runner_id, {"name": runner_id, "type": "unknown"})

    runs_list = list(runs)
    history: List[Dict[str, Any]] = []
    sample_predictions: List[Dict[str, Any]] = []
    for run in runs_list:
        manifest = run.get("manifest", {})
        evaluation = run.get("evaluation") or {}
        rev = (evaluation.get("runner_evaluations") or {}).get(runner_id)
        if rev:
            history.append({
                "run_id": manifest.get("run_id"),
                "evaluated_at": evaluation.get("evaluated_at"),
                "n_matches": rev.get("n_evaluated"),
                "mean_brier": rev.get("mean_brier"),
                "mean_log_loss": rev.get("mean_log_loss"),
                "hit_rate": rev.get("hit_rate"),
            })
        # Pull last few raw predictions for the example reel (no eval needed)
        preds = run.get("predictions", {}).get(runner_id, [])
        for p in preds[:3]:
            if len(sample_predictions) >= 5:
                break
            sample_predictions.append({
                "run_id": manifest.get("run_id"),
                "match_id": p.get("match_id"),
                "p_home_win": p.get("p_home_win"),
                "p_draw": p.get("p_draw"),
                "p_away_win": p.get("p_away_win"),
                "rationale": (p.get("rationale") or "")[:160],
            })

    # Compute lifetime aggregates
    total_n = sum(h["n_matches"] for h in history)
    lifetime_brier = (
        sum(h["mean_brier"] * h["n_matches"] for h in history) / total_n
        if total_n else None
    )
    lifetime_hit = (
        sum(h["hit_rate"] * h["n_matches"] for h in history) / total_n
        if total_n else None
    )

    return {
        "_meta": {
            "endpoint": "GET /api/v1/runners/{runner_id}/card",
            "description": "Profile of one predictor across all runs",
        },
        "runner_id": runner_id,
        "metadata": meta,
        "lifetime": {
            "n_matches_evaluated": total_n,
            "mean_brier": round(lifetime_brier, 4) if lifetime_brier is not None else None,
            "hit_rate": round(lifetime_hit, 4) if lifetime_hit is not None else None,
            "n_runs": len(history),
        },
        "history": history,
        "sample_predictions": sample_predictions,
    }
