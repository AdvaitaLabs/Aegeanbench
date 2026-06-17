"""
Shared prediction routine used by every consensus entry point.

Before this module the /api/v1/predict HTTP handler was the only place
that knew how to turn a match_id into a cached consensus prediction. The
background workers (pre-match warmer, live ticker) need the exact same
behaviour, so the logic lives here and the handler became a thin wrapper.

One function does the real work — ``run_and_cache`` — and three callers
use it:

  * the /predict endpoint (synchronous user request)
  * the pre-match scheduler (T-24h / T-1h warm, long cache TTL)
  * the live ticker (every refresh during play, short TTL + WS broadcast)

The heavy parts (gateway.build_context + predictor.predict) are blocking
network/LLM calls, so we run them in a worker thread via asyncio.to_thread
to keep the event loop (and therefore the WebSocket fan-out) responsive.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# FIFA code lookup for the names the front-end is most likely to send.
# Falls back to the first 3 chars when unknown. Kept here (not in the
# handler) so every caller resolves codes identically.
_FIFA = {
    "Mexico": "MEX", "South Africa": "RSA", "United States": "USA",
    "Argentina": "ARG", "Brazil": "BRA", "France": "FRA",
    "Germany": "GER", "Spain": "ESP", "England": "ENG",
    "Portugal": "POR", "Netherlands": "NED", "Italy": "ITA",
    "Belgium": "BEL", "Croatia": "CRO", "Japan": "JPN",
    "Korea Republic": "KOR", "South Korea": "KOR",
    "Morocco": "MAR", "Saudi Arabia": "KSA",
    "Canada": "CAN", "Australia": "AUS", "Cape Verde": "CPV",
}


def _fifa(name: Optional[str], default: str) -> str:
    if not name:
        return default
    return _FIFA.get(name, name[:3].upper())


def resolve_match(
    *,
    match_id: str,
    home_team: Optional[str],
    away_team: Optional[str],
    kickoff_iso: Optional[str],
    venue: Optional[str] = None,
):
    """
    Build a Match from caller-supplied fields, resolving missing team
    names from soccersapi (t=info) by match_id. Returns None when the
    teams cannot be determined — callers must handle that (the endpoint
    raises 400, background workers skip the match).
    """
    from aegeanbench.sports.models import CompetitionStage, Match, Team

    home = home_team
    away = away_team
    venue_city = venue

    if not home or not away:
        try:
            from aegeanbench.sports.reporter.match_brief import _resolve_match_info
            resolved = _resolve_match_info(match_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("match_id %s team resolution failed: %s", match_id, exc)
            resolved = None
        if resolved:
            home = home or resolved.get("home_team")
            away = away or resolved.get("away_team")
            venue_city = venue_city or resolved.get("venue_city")

    if not home or not away:
        return None

    kickoff = (
        datetime.fromisoformat(kickoff_iso.replace("Z", "+00:00"))
        if kickoff_iso else datetime.now()
    )

    return Match(
        match_id=match_id,
        competition="FIFA World Cup 2026",
        stage=CompetitionStage.GROUP,
        kickoff_at=kickoff,
        home_team=Team(fifa_code=_fifa(home, home[:3].upper()), name=home),
        away_team=Team(fifa_code=_fifa(away, away[:3].upper()), name=away),
        venue=venue_city,
    )


def _is_mock_prediction(prediction) -> bool:
    return (
        prediction.rationale.startswith("[mock")
        or (prediction.tokens_used == 0 and prediction.latency_ms <= 1)
    )


def _build_payload(
    *,
    match_id: str,
    table_id: Optional[str],
    agents_used: List[str],
    prediction,
    is_mock: bool,
    cache_hit: bool = False,
    source: str = "predict",
) -> Dict[str, Any]:
    md = prediction.metadata or {}
    return {
        "_meta": {
            "endpoint": "POST /api/v1/predict",
            "description": "One-shot consensus prediction for a user-defined table",
            "cache_hit": cache_hit,
            "is_mock": is_mock,
            "source": source,
        },
        "table_id": table_id,
        "match_id": match_id,
        "agents_used": agents_used,
        "prediction": {
            "p_home_win": prediction.p_home_win,
            "p_draw": prediction.p_draw,
            "p_away_win": prediction.p_away_win,
            "confidence": prediction.confidence,
            "rationale": prediction.rationale,
            "latency_ms": prediction.latency_ms,
            "tokens_used": prediction.tokens_used,
            "key_factors": md.get("key_factors") or [],
            "likely_scores": md.get("likely_scores") or [],
            "total_goals": md.get("total_goals"),
            "halves": md.get("halves"),
            "top_scorers": md.get("top_scorers") or [],
        },
        "discussion": md.get("discussion"),
    }


def _compute_sync(
    *,
    gateway,
    match,
    agent_ids: List[str],
    lang: str,
    chat_summary: Optional[str],
):
    """Blocking core: build context + run consensus. Runs in a thread."""
    from aegeanbench.sports.gateway import MatchContext
    from aegeanbench.sports.predictors.aegean import AegeanPredictor

    try:
        ctx = gateway.build_context(match)
    except Exception as exc:  # noqa: BLE001
        logger.warning("gateway.build_context failed for %s: %s", match.match_id, exc)
        ctx = MatchContext(
            match=match, home_history=[], away_history=[], h2h=[],
            home_xg_profile={}, away_xg_profile={},
        )
    if chat_summary:
        ctx.chat_summary = chat_summary

    is_in_play = bool(getattr(ctx, "live_state", None))
    predictor = AegeanPredictor(agent_types=agent_ids)
    prediction = predictor.predict(ctx, lang=lang)
    return prediction, is_in_play


async def run_and_cache(
    *,
    gateway,
    cache,
    match_id: str,
    agent_ids: List[str],
    lang: str = "en",
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    kickoff_iso: Optional[str] = None,
    venue: Optional[str] = None,
    chat_messages: Optional[List[Any]] = None,
    chat_summary: Optional[str] = None,
    table_id: Optional[str] = None,
    cache_ttl: Optional[float] = None,
    store: bool = True,
    source: str = "predict",
    live_hub=None,
    broadcast: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Resolve the match, run one consensus, build + cache the payload, and
    optionally broadcast it to the match's WebSocket channel.

    Returns the payload dict, or None when the match could not be
    resolved (no team names). Mock / failed predictions are never cached
    so a single Praka hiccup doesn't poison the cache window.
    """
    match = resolve_match(
        match_id=match_id, home_team=home_team, away_team=away_team,
        kickoff_iso=kickoff_iso, venue=venue,
    )
    if match is None:
        logger.warning("run_and_cache: could not resolve teams for %s", match_id)
        return None

    if chat_summary is None and chat_messages:
        chat_summary = "\n".join(
            f"  - {getattr(m, 'user_name', '?')}: {getattr(m, 'text', '')}"
            for m in chat_messages[-30:]
        )

    prediction, is_in_play = await asyncio.to_thread(
        _compute_sync,
        gateway=gateway, match=match, agent_ids=agent_ids,
        lang=lang, chat_summary=chat_summary,
    )

    is_mock = _is_mock_prediction(prediction)
    if is_mock:
        logger.warning(
            "%s for %s fell back to mock (latency=%dms, tokens=%d) — NOT caching",
            source, match_id, prediction.latency_ms, prediction.tokens_used,
        )

    payload = _build_payload(
        match_id=match_id, table_id=table_id, agents_used=agent_ids,
        prediction=prediction, is_mock=is_mock, source=source,
    )

    if store and not is_mock:
        key = cache.make_key(
            match_id=match_id, agent_ids=agent_ids, lang=lang,
            chat_messages=chat_messages,
        )
        cache.put(key, match_id=match_id, payload=payload, ttl=cache_ttl, lang=lang)

    if broadcast and live_hub is not None and not is_mock:
        # Broadcast the lean, language-neutral prediction_update defined in
        # the existing WebSocket contract (FRONTEND_API.md) — just the
        # probabilities, no rationale text — so subscribed clients update
        # live without any front-end change. Rich/localised text still
        # comes from /predict per the viewer's language.
        try:
            await live_hub.publish(
                f"match:{match_id}",
                {
                    "type": "prediction_update",
                    "match_id": match_id,
                    "runner_id": "aegean",
                    "p_home_win": prediction.p_home_win,
                    "p_draw": prediction.p_draw,
                    "p_away_win": prediction.p_away_win,
                    "confidence": prediction.confidence,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("broadcast failed for %s: %s", match_id, exc)

    return payload
