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


def _normalize_vote_keys(votes: Dict[str, float]) -> Dict[str, float]:
    """
    aegean-consensus' DecisionEngine uses the raw answer string as the
    vote-counter key. For our prediction tasks the raw answer is a JSON
    blob, so the dict comes back like {"```json\\n{\\"p_home_win\\":...}": 3.0}.

    Collapse such keys to the matching outcome label so the front-end
    sees {"home_win": 3.0} like it expects. Non-JSON keys (e.g. when
    consensus already normalised) pass through unchanged.
    """
    if not isinstance(votes, dict):
        return {}
    collapsed: Dict[str, float] = {}
    for raw_key, weight in votes.items():
        key = str(raw_key)
        if "{" in key and "p_home_win" in key:
            probs = _parse_probs(key)
            label = _argmax(probs) if probs else key
        else:
            label = key
        collapsed[label] = collapsed.get(label, 0.0) + float(weight or 0.0)
    return collapsed


def parse_aegean_response_to_trace(
    match_id: str,
    runner_id: str,
    consensus_response: Dict[str, Any],
    role_map: Optional[Dict[str, str]] = None,
) -> DiscussionTrace:
    """
    Convert the GroupConsensusResult JSON returned by aegean-consensus
        POST /api/v1/groups/{id}/consensus
    into a DiscussionTrace.

    The aegean-consensus API returns:
        {
          "consensus_id": "...",
          "success": true,
          "final_solution": { "answer": "...", "confidence": 0.71, "reasoning": "..." },
          "weighted_votes": { "home_win": 1.75, "draw": 0.62 },
          "consensus_path": ["agent_0", "agent_1"],
          "rounds_used": 2,
          "discussion_rounds": [
            {
              "round_number": 1,
              "agent_responses": {
                "stats_specialist": {"agent_id": "...", "answer": "{...}", "confidence": 0.72, "reasoning": "..."},
                ...
              },
              "candidate_answer": "...",
              "candidate_confidence": 0.65,
              "stability_counter": 1,
              "consensus_status": "ongoing"
            },
            ...
          ]
        }

    We also tolerate the legacy / SDK-direct field `rounds_history` for
    callers that drive ConsensusCoordinator directly without going through
    GroupChatService.
    """
    raw_rounds: List[Dict[str, Any]] = (
        consensus_response.get("discussion_rounds")
        or consensus_response.get("rounds_history")
        or []
    )
    rounds: List[DiscussionRound] = []
    prev_argmax_per_agent: Dict[str, str] = {}

    for raw in raw_rounds:
        # The two shapes:
        #   discussion_rounds: agent_responses is a Dict[agent_id, Solution]
        #   rounds_history:    agent_solutions is a List[Solution]
        if "agent_responses" in raw:
            solution_iter = raw["agent_responses"].items()
            get_id = lambda item: item[0]
            get_sol = lambda item: item[1]
        else:
            solution_iter = enumerate(raw.get("agent_solutions", []))
            get_id = lambda item: item[1].get("agent_id", "")
            get_sol = lambda item: item[1]

        agents_entries: List[DiscussionAgentEntry] = []
        for item in solution_iter:
            agent_id = get_id(item)
            sol = get_sol(item) or {}
            probs = _parse_probs(sol.get("answer", "{}"))
            argmax = _argmax(probs)
            prev = prev_argmax_per_agent.get(agent_id)
            # Resolve role: prefer explicit role from consensus payload,
            # then the caller-provided agent_id->role map (e.g. agent_0
            # -> stats_specialist), and finally fall back to agent_id.
            resolved_role = (
                sol.get("role")
                or (role_map.get(agent_id) if role_map else None)
                or agent_id
            )
            # Pull rationale from the JSON answer first (that's the real
            # agent-authored reasoning); fall back to Solution.reasoning
            # only when the JSON didn't include one. This avoids the
            # "Refined based on peer solutions" boilerplate that
            # MinimalAgent.refine_solution writes there in later rounds.
            jr = _parse_rationale(sol.get("answer", "") or "")
            rationale_text = jr or str(sol.get("reasoning", "") or "")
            agents_entries.append(
                DiscussionAgentEntry(
                    agent_id=agent_id,
                    role=resolved_role,
                    p_home_win=probs.get("home_win", 0.0),
                    p_draw=probs.get("draw", 0.0),
                    p_away_win=probs.get("away_win", 0.0),
                    confidence=float(sol.get("confidence", 0.0)),
                    rationale=rationale_text[:500],
                    changed_position=(prev is not None and prev != argmax),
                    previous_argmax=prev,
                    current_argmax=argmax,
                )
            )
            prev_argmax_per_agent[agent_id] = argmax

        # candidate_answer at the GroupConsensusResult level is raw answer
        # text - convert it to our outcome label by extracting JSON if any
        candidate_answer = raw.get("candidate_answer") or raw.get("candidate_outcome") or ""
        candidate_probs = _parse_probs(candidate_answer) if "{" in str(candidate_answer) else {}
        candidate_outcome = (
            _argmax(candidate_probs) if candidate_probs
            else str(candidate_answer)
        )

        rounds.append(
            DiscussionRound(
                round_number=int(raw.get("round_number", len(rounds) + 1)),
                candidate_outcome=candidate_outcome,
                candidate_confidence=float(raw.get("candidate_confidence") or 0.0),
                quorum_reached=bool(
                    raw.get("consensus_status") in ("reached", "quorum_reached")
                    or raw.get("quorum_reached")
                ),
                agents=agents_entries,
                weighted_votes=_normalize_vote_keys(raw.get("weighted_votes") or {}),
                agreement_points=raw.get("agreement_points") or [],
                disagreement_points=raw.get("disagreement_points") or [],
            )
        )

    final = consensus_response.get("final_solution") or {}
    return DiscussionTrace(
        match_id=match_id,
        runner_id=runner_id,
        enabled=True,
        rounds_used=int(consensus_response.get("rounds_used") or len(rounds)),
        rounds=rounds,
        final_summary=str(final.get("reasoning", "")),
        raw_metadata={
            "weighted_votes": _normalize_vote_keys(consensus_response.get("weighted_votes") or {}),
            "consensus_path": consensus_response.get("consensus_path") or [],
            "consensus_reached": consensus_response.get("consensus_reached", False),
        },
    )


def _parse_answer_blob(answer_blob: str) -> Dict[str, Any]:
    """Best-effort parse of the agent's JSON answer; returns {} on failure."""
    import json
    import re
    try:
        return json.loads(answer_blob)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", answer_blob or "", re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


def _parse_probs(answer_blob: str) -> Dict[str, float]:
    """Robustly extract the 3 probabilities from a JSON answer string."""
    data = _parse_answer_blob(answer_blob)
    if not data:
        return {"home_win": 1/3, "draw": 1/3, "away_win": 1/3}
    return {
        "home_win": float(data.get("p_home_win", 0.0)),
        "draw": float(data.get("p_draw", 0.0)),
        "away_win": float(data.get("p_away_win", 0.0)),
    }


def _parse_rationale(answer_blob: str) -> str:
    """
    Extract the 'rationale' (or 'reasoning') string from the JSON answer.
    Falls back to the raw text trimmed when no JSON is present.
    """
    data = _parse_answer_blob(answer_blob)
    for k in ("rationale", "reasoning", "explanation"):
        v = data.get(k)
        if v:
            return str(v)
    return ""


def _argmax(probs: Dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: kv[1])[0]
