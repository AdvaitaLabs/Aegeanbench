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
        max_rounds: int = 3,
        timeout: float = 60.0,
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
            return self._real_predict(ctx, prompts)
        except Exception as e:
            logger.warning("Aegean real call failed (%s); falling back to mock", e)
            return self._mock_predict(ctx, prompts)

    # ----------------------- real path -----------------------

    def _real_predict(
        self, ctx: MatchContext, prompts: Dict[str, str]
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

            # Step 4: parse final answer
            final = result.get("final_solution", {})
            answer_text = final.get("answer", "")
            parsed = _extract_json(answer_text)
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

            return Prediction(
                match_id=ctx.match.match_id,
                runner_id=self.runner_id,
                p_home_win=p_home,
                p_draw=p_draw,
                p_away_win=p_away,
                confidence=float(parsed.get("confidence", final.get("confidence", 0.6))),
                rationale=str(parsed.get("rationale", "")),
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
