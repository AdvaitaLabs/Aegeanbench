"""
Discussion trace types - mirror the investment-side DiscussionRound /
ConsensusTrace so the front-end can render multi-round agent debates
the same way it already renders the investment analyser output.

Each consensus call returns:
    - per-round agent solutions
    - vote tallies / weights
    - agreement & disagreement points (when the LLM emits them)
    - final picked solution

Used by AegeanPredictor to record what happened, by reporter endpoints
to expose it, and by the front-end to visualise "how the agents argued".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DiscussionAgentEntry:
    """One agent's view in one round."""
    agent_id: str
    role: str = ""                       # specialist role (stats_specialist, etc.)
    p_home_win: float = 0.0
    p_draw: float = 0.0
    p_away_win: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    changed_position: bool = False       # vs previous round
    previous_argmax: Optional[str] = None
    current_argmax: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "p_home_win": self.p_home_win,
            "p_draw": self.p_draw,
            "p_away_win": self.p_away_win,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "changed_position": self.changed_position,
            "previous_argmax": self.previous_argmax,
            "current_argmax": self.current_argmax,
        }


@dataclass
class DiscussionRound:
    """One round of refinement."""
    round_number: int
    candidate_outcome: str = ""          # home_win / draw / away_win
    candidate_confidence: float = 0.0
    quorum_reached: bool = False
    agents: List[DiscussionAgentEntry] = field(default_factory=list)
    weighted_votes: Dict[str, float] = field(default_factory=dict)
    agreement_points: List[str] = field(default_factory=list)
    disagreement_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "candidate_outcome": self.candidate_outcome,
            "candidate_confidence": self.candidate_confidence,
            "quorum_reached": self.quorum_reached,
            "agents": [a.to_dict() for a in self.agents],
            "weighted_votes": self.weighted_votes,
            "agreement_points": self.agreement_points,
            "disagreement_points": self.disagreement_points,
        }


@dataclass
class DiscussionTrace:
    """Full discussion record for one consensus call."""
    match_id: str
    runner_id: str
    enabled: bool = True
    rounds_used: int = 0
    rounds: List[DiscussionRound] = field(default_factory=list)
    final_summary: str = ""
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "runner_id": self.runner_id,
            "enabled": self.enabled,
            "rounds_used": self.rounds_used,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_summary": self.final_summary,
            "raw_metadata": self.raw_metadata,
        }


def parse_aegean_response_to_trace(
    match_id: str,
    runner_id: str,
    consensus_response: Dict[str, Any],
) -> DiscussionTrace:
    """
    Convert the raw JSON returned by aegean-consensus
        POST /api/v1/groups/{id}/consensus
    into a DiscussionTrace.

    The expected response shape (loosely typed):
        {
          "consensus_id": "...",
          "success": true,
          "final_solution": { "answer": "{...JSON...}", "confidence": 0.71 },
          "weighted_votes": { "buy": 1.75, "hold": 0.62 },
          "rounds_used": 2,
          "rounds_history": [
            {
              "round_number": 1,
              "quorum_reached": false,
              "agent_solutions": [
                {"agent_id": "stats_specialist", "answer": "{...}", "confidence": 0.72, "reasoning": "..."}
              ]
            },
            ...
          ]
        }

    If rounds_history is absent (older aegean-consensus version) we still
    construct a single-round trace from final_solution alone.
    """
    rounds_history: List[Dict[str, Any]] = consensus_response.get("rounds_history") or []
    rounds: List[DiscussionRound] = []

    prev_argmax_per_agent: Dict[str, str] = {}

    for raw in rounds_history:
        agents_entries: List[DiscussionAgentEntry] = []
        for sol in raw.get("agent_solutions", []):
            probs = _parse_probs(sol.get("answer", "{}"))
            argmax = _argmax(probs)
            agent_id = sol.get("agent_id", "")
            prev = prev_argmax_per_agent.get(agent_id)
            agents_entries.append(
                DiscussionAgentEntry(
                    agent_id=agent_id,
                    role=sol.get("role", agent_id),
                    p_home_win=probs.get("home_win", 0.0),
                    p_draw=probs.get("draw", 0.0),
                    p_away_win=probs.get("away_win", 0.0),
                    confidence=float(sol.get("confidence", 0.0)),
                    rationale=str(sol.get("reasoning", ""))[:500],
                    changed_position=(prev is not None and prev != argmax),
                    previous_argmax=prev,
                    current_argmax=argmax,
                )
            )
            prev_argmax_per_agent[agent_id] = argmax

        rounds.append(
            DiscussionRound(
                round_number=int(raw.get("round_number", len(rounds) + 1)),
                candidate_outcome=str(raw.get("candidate_outcome", "")),
                candidate_confidence=float(raw.get("candidate_confidence", 0.0)),
                quorum_reached=bool(raw.get("quorum_reached", False)),
                agents=agents_entries,
                weighted_votes=raw.get("weighted_votes") or {},
                agreement_points=raw.get("agreement_points") or [],
                disagreement_points=raw.get("disagreement_points") or [],
            )
        )

    final = consensus_response.get("final_solution") or {}
    return DiscussionTrace(
        match_id=match_id,
        runner_id=runner_id,
        enabled=True,
        rounds_used=int(consensus_response.get("rounds_used", len(rounds))),
        rounds=rounds,
        final_summary=str(final.get("reasoning", "")),
        raw_metadata={
            "weighted_votes": consensus_response.get("weighted_votes", {}),
            "consensus_reached": consensus_response.get("consensus_reached", False),
        },
    )


def _parse_probs(answer_blob: str) -> Dict[str, float]:
    """Robustly extract the 3 probabilities from a JSON answer string."""
    import json
    import re
    try:
        data = json.loads(answer_blob)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", answer_blob or "", re.DOTALL)
        if not m:
            return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}
    return {
        "home_win": float(data.get("p_home_win", 0.0)),
        "draw": float(data.get("p_draw", 0.0)),
        "away_win": float(data.get("p_away_win", 0.0)),
    }


def _argmax(probs: Dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0]
