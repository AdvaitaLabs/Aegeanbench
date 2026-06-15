"""
Aegean consensus black-box predictor.

Talks to a running aegean-consensus HTTP server. The benchmark stays
agnostic about which agents the server is running internally; the
predictor just sends the same unified prompt and parses the consensus
output.

Mock mode: when the server is unreachable (sprint dev) the predictor
returns an averaged-and-smoothed version of multiple mock LLM outputs,
simulating the variance reduction that real consensus achieves.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from aegeanbench.sports.gateway import MatchContext
from aegeanbench.sports.models import Prediction
from aegeanbench.sports.predictors.base import Predictor
from aegeanbench.sports.predictors.llm import (
    LLMClient,
    MockLLMClient,
    _extract_json,
    _normalize_probs,
)
from aegeanbench.sports.prompts import build_full_prompt

logger = logging.getLogger(__name__)


DEFAULT_AGENT_TYPES = [
    "stats_specialist",
    "player_specialist",
    "strategy_specialist",
    "market_specialist",
    "news_specialist",
    "occult_specialist",
]

DEFAULT_AGENT_WEIGHTS = {
    "stats_specialist": 0.90,
    "player_specialist": 0.85,
    "strategy_specialist": 0.80,
    "market_specialist": 0.75,
    "news_specialist": 0.65,
    "occult_specialist": 0.10,
}


class AegeanPredictor(Predictor):
    """
    Calls aegean-consensus /api/v1 to obtain a multi-agent consensus
    prediction.

    Real mode (server reachable):
      1. POST /api/v1/groups  -> group_id
      2. POST /api/v1/groups/{id}/members  for each agent (with weight)
      3. POST /api/v1/groups/{id}/consensus  with task=unified_prompt
      4. parse JSON from final_solution.answer
      5. DELETE /api/v1/groups/{id}

    Mock mode (server unreachable OR mock=True):
      Build N MockLLMClient predictions with different personalities and
      aggregate via weighted average. This simulates consensus variance
      reduction while remaining fully offline.
    """

    runner_id = "aegean"
    display_name = "Aegean Consensus"

    def __init__(
        self,
        base_url: Optional[str] = None,
        agent_types: Optional[List[str]] = None,
        agent_weights: Optional[Dict[str, float]] = None,
        mock: Optional[bool] = None,
        quorum_threshold: float = 0.6,
        max_rounds: int = 4,
        # Bumped from 60s -> 120s -> 150s. The richer prompt (likely_
        # scores / halves / top_scorers extra fields) makes outputs
        # 30-40% longer, so consensus can run ~110-140s under Praka
        # load. nginx upstream timeout is 180s so we keep margin.
        timeout: float = 150.0,
    ):
        self.base_url = (
            base_url
            or os.getenv("AEGEAN_CONSENSUS_URL")
            or "http://localhost:8000"
        ).rstrip("/")
        self.agent_types = agent_types or list(DEFAULT_AGENT_TYPES)
        self.agent_weights = agent_weights or dict(DEFAULT_AGENT_WEIGHTS)
        # If not explicitly told, default to mock when no URL was configured
        if mock is None:
            mock = os.getenv("AEGEAN_CONSENSUS_URL") is None
        self.mock = mock
        self.quorum_threshold = quorum_threshold
        self.max_rounds = max_rounds
        self.timeout = timeout

    # ----------------------- predict -----------------------

    def predict(self, ctx: MatchContext, lang: str = "en") -> Prediction:
        prompts = build_full_prompt(ctx, lang=lang)
        if self.mock:
            return self._mock_predict(ctx, prompts)
        try:
            return self._real_predict(ctx, prompts, lang=lang)
        except Exception as e:
            # Log with exc_info so we can see WHERE it failed: timeout
            # vs HTTP error vs schema mismatch all look different in the
            # traceback. Without this, intermittent mocks are
            # invisible in production.
            logger.warning(
                "Aegean real call failed for match=%s (%s: %s); falling back to mock",
                ctx.match.match_id, type(e).__name__, e,
                exc_info=True,
            )
            return self._mock_predict(ctx, prompts)

    # ----------------------- real path -----------------------

    def _real_predict(
        self, ctx: MatchContext, prompts: Dict[str, str], lang: str = "en"
    ) -> Prediction:
        import requests

        start = time.perf_counter()
        group_id = None
        try:
            # Step 1: create group.
            # aegean-consensus' CreateGroupRequest schema requires
            # group_name + created_by; we used to send {"name": ...} and
            # got 422. The consensus-side AgentRegistry only knows
            # generic agent_0..agent_N IDs, so we map each requested
            # sports specialist to one of those slots and pass the
            # specialist name through the `role` field for telemetry.
            r = requests.post(
                f"{self.base_url}/api/v1/groups",
                json={
                    "group_name": f"wc-pred-{ctx.match.match_id}",
                    "created_by": "aegeanbench",
                    "mode": "consensus",
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            group_id = r.json()["group_id"]

            # Step 2: add members. Map sports role -> generic agent_id.
            for i, agent_type in enumerate(self.agent_types):
                weight = self.agent_weights.get(agent_type, 1.0)
                requests.post(
                    f"{self.base_url}/api/v1/groups/{group_id}/members",
                    json={
                        "agent_id": f"agent_{i}",
                        "role": agent_type,
                        "capability_weight": weight,
                    },
                    timeout=self.timeout,
                ).raise_for_status()

            # Step 3: trigger consensus
            task_payload = f"{prompts['system']}\n\n{prompts['user']}"
            r = requests.post(
                f"{self.base_url}/api/v1/groups/{group_id}/consensus",
                json={
                    "task": task_payload,
                    "quorum_threshold": self.quorum_threshold,
                    "max_rounds": self.max_rounds,
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            result = r.json()

            # Step 4: parse final answer. consensus can return
            # final_solution=null when success=False (e.g. when the
            # MatchOutcomeNormalizer hasn't bucketed enough weight onto
            # a single outcome). Recover by averaging probabilities
            # across the final round's agent answers — they're all real
            # LLM output, just didn't reach quorum.
            final = result.get("final_solution") or {}
            answer_text = final.get("answer", "") if isinstance(final, dict) else ""
            parsed = _extract_json(answer_text)
            if not parsed or all(parsed.get(k, 0) == 0 for k in ("p_home_win", "p_draw", "p_away_win")):
                # Salvage from the discussion: average probs across last round
                rounds = result.get("discussion_rounds") or []
                if rounds:
                    last = rounds[-1]
                    agent_solutions = last.get("agent_responses") or {}
                    accum = {"p_home_win": 0.0, "p_draw": 0.0, "p_away_win": 0.0}
                    n = 0
                    for s in agent_solutions.values():
                        sub = _extract_json((s or {}).get("answer", "") or "")
                        if not sub:
                            continue
                        accum["p_home_win"] += float(sub.get("p_home_win", 0) or 0)
                        accum["p_draw"] += float(sub.get("p_draw", 0) or 0)
                        accum["p_away_win"] += float(sub.get("p_away_win", 0) or 0)
                        n += 1
                    if n:
                        parsed = {
                            "p_home_win": accum["p_home_win"] / n,
                            "p_draw": accum["p_draw"] / n,
                            "p_away_win": accum["p_away_win"] / n,
                            "confidence": 0.5,
                            "rationale": "Salvaged from final-round agents (consensus did not reach quorum).",
                        }

            p_home, p_draw, p_away = _normalize_probs(
                float(parsed.get("p_home_win", 0)),
                float(parsed.get("p_draw", 0)),
                float(parsed.get("p_away_win", 0)),
            )
            latency_ms = int((time.perf_counter() - start) * 1000)

            # Capture the full multi-round discussion for the front-end.
            # Build the agent_N -> sports_role map from the order we
            # registered members in step 2, so the trace shows
            # "stats_specialist" instead of the generic "agent_2".
            role_map = {
                f"agent_{i}": role for i, role in enumerate(self.agent_types)
            }
            from aegeanbench.sports.discussion import parse_aegean_response_to_trace
            discussion = parse_aegean_response_to_trace(
                match_id=ctx.match.match_id,
                runner_id=self.runner_id,
                consensus_response=result,
                role_map=role_map,
            )

            # aegean-consensus reports tokens via tokens_prompt /
            # tokens_completion (or a usage.tokens_total roll-up); the
            # legacy `tokens_used` field never existed in the response.
            usage = result.get("usage") or {}
            tokens_total = (
                int(usage.get("tokens_total", 0))
                or int(result.get("tokens_prompt", 0))
                   + int(result.get("tokens_completion", 0))
            )

            # Auto-fill top_scorers when LLM returned [] but we have a
            # real squad — write back into `parsed` so the structured
            # rationale suffix renders these names too. This keeps the
            # response metadata and the inline summary in sync.
            if not parsed.get("top_scorers"):
                parsed["top_scorers"] = _auto_top_scorers(ctx, p_home, p_away)

            # Append a structured summary block to the rationale so a
            # front-end that doesn't yet render likely_scores / halves /
            # top_scorers separately still surfaces them inline.
            base_rationale = str(parsed.get("rationale", ""))
            structured_suffix = _format_structured_summary(
                parsed, lang=lang,
                live_state=getattr(ctx, "live_state", None),
            )
            rationale_with_summary = (
                base_rationale.rstrip() + "\n\n" + structured_suffix
                if structured_suffix else base_rationale
            )

            return Prediction(
                match_id=ctx.match.match_id,
                runner_id=self.runner_id,
                p_home_win=p_home,
                p_draw=p_draw,
                p_away_win=p_away,
                confidence=float(parsed.get("confidence", final.get("confidence", 0.6))),
                rationale=rationale_with_summary,
                latency_ms=latency_ms,
                tokens_used=tokens_total,
                metadata={
                    "model": "aegean",
                    "rounds_used": result.get("rounds_used", 0),
                    "weighted_votes": discussion.raw_metadata.get(
                        "weighted_votes", {}
                    ),
                    "agent_types": self.agent_types,
                    "discussion": discussion.to_dict(),
                    # Pass through the new richer JSON fields the prompt
                    # asks for. They're optional — model may omit when
                    # data is thin — so default to None / empty list.
                    "key_factors": parsed.get("key_factors") or [],
                    "likely_scores": parsed.get("likely_scores") or [],
                    "total_goals": parsed.get("total_goals"),
                    "halves": parsed.get("halves"),
                    # Already auto-filled above when LLM returned []
                    "top_scorers": parsed.get("top_scorers") or [],
                },
            )
        finally:
            # Step 5: cleanup
            if group_id:
                try:
                    requests.delete(
                        f"{self.base_url}/api/v1/groups/{group_id}",
                        timeout=5,
                    )
                except Exception:  # noqa: BLE001
                    pass

    # ----------------------- mock path -----------------------

    def _mock_predict(
        self, ctx: MatchContext, prompts: Dict[str, str]
    ) -> Prediction:
        """
        Simulate consensus offline: sample multiple mock LLM responses
        (one per agent type) and combine via weighted average.
        """
        start = time.perf_counter()
        # Each agent gets a slightly different prompt focus
        focus_map = {
            "stats_specialist": "stats",
            "player_specialist": "player",
            "strategy_specialist": "strategy",
            "market_specialist": "market",
            "news_specialist": "news",
            "occult_specialist": "occult",
        }
        agent_predictions: List[Tuple[Dict[str, float], float]] = []
        total_tokens = 0
        weighted_votes: Dict[str, float] = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}

        for agent_type in self.agent_types:
            focus = focus_map.get(agent_type)
            # Build a small per-agent prompt variant so MockLLMClient produces
            # different probabilities for each agent (its hash changes)
            user = prompts["user"] + (f"\n\n[lens: {focus}]" if focus else "")
            client = MockLLMClient(runner_id=f"aegean-{agent_type}")
            try:
                raw, tokens = client.complete(prompts["system"], user)
                parsed = _extract_json(raw)
                p_home = float(parsed.get("p_home_win", 0.4))
                p_draw = float(parsed.get("p_draw", 0.3))
                p_away = float(parsed.get("p_away_win", 0.3))
            except Exception:
                p_home, p_draw, p_away = 0.4, 0.3, 0.3
                tokens = 0

            weight = self.agent_weights.get(agent_type, 1.0)
            agent_predictions.append(({
                "home_win": p_home, "draw": p_draw, "away_win": p_away,
            }, weight))
            total_tokens += tokens

            # Symbolic per-agent weighted vote on the argmax outcome
            argmax = max(
                ("home_win", p_home), ("draw", p_draw), ("away_win", p_away),
                key=lambda x: x[1],
            )[0]
            weighted_votes[argmax] += weight

        # Weighted average of per-agent probability vectors
        total_weight = sum(w for _, w in agent_predictions)
        p_home = sum(p["home_win"] * w for p, w in agent_predictions) / total_weight
        p_draw = sum(p["draw"] * w for p, w in agent_predictions) / total_weight
        p_away = sum(p["away_win"] * w for p, w in agent_predictions) / total_weight
        p_home, p_draw, p_away = _normalize_probs(p_home, p_draw, p_away)

        # Confidence from the dispersion of agent predictions:
        # tight cluster => high confidence; wide spread => low confidence.
        def spread(key: str) -> float:
            xs = [p[key] for p, _ in agent_predictions]
            mean = sum(xs) / len(xs)
            return sum((x - mean) ** 2 for x in xs) / len(xs)

        avg_spread = (spread("home_win") + spread("draw") + spread("away_win")) / 3
        confidence = max(0.4, 0.9 - avg_spread * 10)  # heuristic

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Build a synthetic two-round DiscussionTrace from the per-agent
        # samples we just generated, so the front-end has consistent data
        # to render whether we're in mock or live mode.
        from aegeanbench.sports.discussion import (
            DiscussionAgentEntry,
            DiscussionRound,
            DiscussionTrace,
        )
        round1_agents: List[DiscussionAgentEntry] = []
        for agent_type, (probs, weight) in zip(self.agent_types, agent_predictions):
            argmax = max(probs.items(), key=lambda kv: kv[1])[0]
            round1_agents.append(
                DiscussionAgentEntry(
                    agent_id=agent_type,
                    role=agent_type,
                    p_home_win=probs["home_win"],
                    p_draw=probs["draw"],
                    p_away_win=probs["away_win"],
                    confidence=round(weight, 2),
                    rationale=f"[mock {agent_type}] lens-specific reasoning",
                    current_argmax=argmax,
                )
            )
        # Round 2 = convergence (everyone aligned to weighted average)
        consensus_argmax = max(
            ("home_win", p_home), ("draw", p_draw), ("away_win", p_away),
            key=lambda x: x[1],
        )[0]
        discussion = DiscussionTrace(
            match_id=ctx.match.match_id,
            runner_id=self.runner_id,
            enabled=True,
            rounds_used=2,
            rounds=[
                DiscussionRound(
                    round_number=1,
                    candidate_outcome=consensus_argmax,
                    candidate_confidence=round(confidence, 3),
                    quorum_reached=True,
                    agents=round1_agents,
                    weighted_votes=weighted_votes,
                ),
            ],
            final_summary=(
                f"Mock consensus across {len(self.agent_types)} agents; "
                f"final outcome {consensus_argmax}."
            ),
        )

        return Prediction(
            match_id=ctx.match.match_id,
            runner_id=self.runner_id,
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            confidence=round(confidence, 3),
            rationale=(
                f"[mock] Weighted average of {len(self.agent_types)} mock agents. "
                f"Weighted votes: {weighted_votes}"
            ),
            latency_ms=latency_ms,
            tokens_used=total_tokens,
            metadata={
                "model": "aegean-mock",
                "rounds_used": 2,
                "weighted_votes": weighted_votes,
                "agent_types": list(self.agent_types),
                "agent_count": len(self.agent_types),
                "discussion": discussion.to_dict(),
            },
        )



def _format_structured_summary(
    parsed: Dict[str, Any],
    lang: str = "en",
    live_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a human-readable text block that summarises the extra
    prediction fields (likely_scores / total_goals / halves / top_scorers).

    Appended to the consensus rationale so a front-end that hasn't yet
    added dedicated UI for these fields still shows them inline. Pure
    ASCII + minimal formatting so it renders cleanly inside a chat
    bubble.

    Returns "" when the parsed payload has none of these optional
    fields — avoids dangling "---" separators below the rationale.
    """
    likely = parsed.get("likely_scores") or []
    totals = parsed.get("total_goals")
    halves = parsed.get("halves")
    scorers = parsed.get("top_scorers") or []

    if not any([likely, totals, halves, scorers]):
        return ""

    is_zh = lang == "zh"
    lines: List[str] = []
    lines.append("———")

    if likely:
        head = "比分预测：" if is_zh else "Likely scores:"
        parts = []
        for it in likely[:3]:
            sc = str(it.get("score", "?"))
            try:
                prob = float(it.get("prob") or 0.0)
            except (TypeError, ValueError):
                prob = 0.0
            parts.append(f"{sc} ({int(round(prob * 100))}%)")
        lines.append(head + " " + " · ".join(parts))

    if isinstance(totals, dict):
        try:
            expected = float(totals.get("expected") or 0.0)
            over = float(totals.get("over_2_5_prob") or 0.0)
            under = float(totals.get("under_2_5_prob") or 0.0)
        except (TypeError, ValueError):
            expected = over = under = 0.0
        if is_zh:
            lines.append(
                f"总进球：预期 {expected:.1f} · 大球 2.5 {int(round(over*100))}% · 小球 2.5 {int(round(under*100))}%"
            )
        else:
            lines.append(
                f"Total goals: {expected:.1f} expected · Over 2.5 {int(round(over*100))}% · Under 2.5 {int(round(under*100))}%"
            )

    # For LIVE / FINISHED matches, override the model's halves
    # estimate with the ACTUAL HT and current scores from the live
    # feed. Model tends to interpret "second_half_goals_expected"
    # as "remaining goals from now", which gives 0 / 0 in a 78' 3-1
    # game. We have the real numbers — use them.
    if isinstance(live_state, dict) and live_state.get("status") in (
        "inplay", "ht", "ft"
    ):
        try:
            hg = int(live_state.get("home_goals") or 0)
            ag = int(live_state.get("away_goals") or 0)
            ht_h = int(live_state.get("ht_home_goals") or 0)
            ht_a = int(live_state.get("ht_away_goals") or 0)
            minute = int(live_state.get("minute") or 0)
        except (TypeError, ValueError):
            hg = ag = ht_h = ht_a = minute = 0

        first_half_goals = ht_h + ht_a
        second_half_goals = max(0, (hg + ag) - first_half_goals)
        # 1st half lean from actual HT result
        if ht_h > ht_a:
            lean_zh, lean_en = "主胜领先", "Home led"
        elif ht_h < ht_a:
            lean_zh, lean_en = "客胜领先", "Away led"
        else:
            lean_zh, lean_en = "平局", "tied"

        status = live_state.get("status")
        suffix_zh = "已完成" if status == "ft" else f"截至 {minute}'"
        suffix_en = "final" if status == "ft" else f"as of {minute}'"

        if is_zh:
            lines.append(
                f"上下半场（{suffix_zh}）：上半 {first_half_goals} 球 ({lean_zh})"
                f" · 下半 {second_half_goals} 球"
            )
        else:
            lines.append(
                f"Halves ({suffix_en}): 1st half {first_half_goals} goals ({lean_en})"
                f" · 2nd half {second_half_goals} goals"
            )

    elif isinstance(halves, dict):
        # Pre-match: trust the model's expected-goals estimate
        try:
            fh = float(halves.get("first_half_goals_expected") or 0.0)
            sh = float(halves.get("second_half_goals_expected") or 0.0)
        except (TypeError, ValueError):
            fh = sh = 0.0
        lean = str(halves.get("first_half_outcome_lean") or "").lower()
        lean_zh = {"home": "主胜倾向", "away": "客胜倾向", "draw": "平局倾向"}.get(lean, "")
        lean_en = lean.capitalize() if lean else ""
        if is_zh:
            tail = f" · 上半场{lean_zh}" if lean_zh else ""
            lines.append(f"上下半场（预测）：上半 {fh:.1f} 球 · 下半 {sh:.1f} 球{tail}")
        else:
            tail = f" · 1st half {lean_en} lean" if lean_en else ""
            lines.append(f"Halves (pre-match): 1st half {fh:.1f} goals · 2nd half {sh:.1f} goals{tail}")

    if scorers:
        head = "可能进球者：" if is_zh else "Top scorers:"
        parts = []
        for s in scorers[:3]:
            name = s.get("name") or "?"
            try:
                prob = float(s.get("prob") or 0.0)
            except (TypeError, ValueError):
                prob = 0.0
            team = (s.get("team") or "").lower()
            side_tag = ""
            if team == "home":
                side_tag = "(主)" if is_zh else "(home)"
            elif team == "away":
                side_tag = "(客)" if is_zh else "(away)"
            parts.append(f"{name}{side_tag} {int(round(prob * 100))}%")
        lines.append(head + " " + " · ".join(parts))

    return "\n".join(lines)


def _auto_top_scorers(ctx, p_home: float, p_away: float) -> List[Dict[str, Any]]:
    """
    Fallback when the LLM returned an empty `top_scorers` list but we
    DO have real squad names from football-data / soccersapi.

    Strategy:
        1. Take the squads from match.home_lineup / match.away_lineup
        2. Filter to forwards (FW) and attacking midfielders (MF)
        3. Pick top 2 from the team more likely to win, 1 from the other
        4. Assign decreasing probabilities (rough heuristic, not real xG)

    Returns [] only when neither side has real player names — in that
    case the prompt's "no squad data" rule legitimately kicks in.
    """
    home_pool = _filter_attackers(getattr(ctx.match, "home_lineup", None))
    away_pool = _filter_attackers(getattr(ctx.match, "away_lineup", None))
    if not home_pool and not away_pool:
        return []

    # Lean: more picks from the team with higher win probability.
    home_picks = 2 if p_home >= p_away else 1
    away_picks = 3 - home_picks
    base_prob = 0.30 if max(p_home, p_away) > 0.6 else 0.22

    out: List[Dict[str, Any]] = []
    decay = 0.0
    for p in home_pool[:home_picks]:
        out.append({
            "name": p.name,
            "team": "home",
            "prob": round(max(0.05, base_prob - decay), 2),
        })
        decay += 0.10
    decay = 0.0
    for p in away_pool[:away_picks]:
        # Away pool gets a lower starting prob if home is favoured
        away_base = base_prob - 0.10 if p_home > p_away else base_prob
        out.append({
            "name": p.name,
            "team": "away",
            "prob": round(max(0.04, away_base - decay), 2),
        })
        decay += 0.08
    return out


def _filter_attackers(lineup) -> List:
    """Prefer FW > attacking MF > the rest, skip placeholders."""
    if not lineup:
        return []
    forwards: List = []
    midfielders: List = []
    others: List = []
    for p in lineup:
        name = (getattr(p, "name", "") or "").strip()
        # Skip mock placeholders like "Player 1" / "REP Player 2"
        if not name or "Player " in name or name.lower().startswith("unknown"):
            continue
        pos = (getattr(p, "position", "") or "").upper()
        if pos == "FW":
            forwards.append(p)
        elif pos == "MF":
            midfielders.append(p)
        else:
            others.append(p)
    return forwards + midfielders + others
