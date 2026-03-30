"""
MockLLMAgent — a lightweight Agent implementation for AegeanBench.

Used by AegeanRunner to plug into the real ConsensusCoordinator
without requiring a live LLM API key.

Behaviour:
  - Non-outlier agents always return correct_answer (or a random variant)
  - Outlier agents return a deterministically wrong answer
  - If a real llm_client is provided AND a task is given, uses the LLM
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

from aegean.core.agent import Agent
from aegean.core.models import Solution


class MockLLMAgent(Agent):
    """
    Mock agent that returns a fixed answer.
    Optionally uses a real LLM client when provided.
    """

    def __init__(
        self,
        agent_id: str,
        correct_answer: str = "correct",
        answer_variants: Optional[List[str]] = None,
        capability_weight: float = 1.0,
        is_outlier: bool = False,
        llm_client: Optional[Any] = None,
        task: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__(
            agent_id=agent_id,
            capability_weight=capability_weight,
        )
        self.correct_answer   = correct_answer
        self.answer_variants  = answer_variants or [correct_answer]
        self.is_outlier       = is_outlier
        self.llm_client       = llm_client
        self.task             = task
        self._rng             = random.Random(seed + hash(agent_id) % 1000)

    async def generate_solution(self, task: str) -> Solution:
        if self.llm_client and task:
            return await self._llm_solution(task)
        answer = self._pick_answer()
        confidence = self._rng.uniform(0.3, 0.6) if self.is_outlier else self._rng.uniform(0.75, 0.99)
        return Solution(
            agent_id=self.agent_id,
            answer=answer,
            reasoning=f"MockLLMAgent {'(outlier) ' if self.is_outlier else ''}selected: {answer}",
            confidence=round(confidence, 3),
        )

    async def refine_solution(self, refinement_set: List[Solution]) -> Solution:
        """In refinement, non-outlier agents adopt the majority answer."""
        if self.is_outlier:
            return await self.generate_solution(self.task or "refine")

        from collections import Counter
        answers = [s.answer for s in refinement_set]
        if answers:
            majority, _ = Counter(answers).most_common(1)[0]
            return Solution(
                agent_id=self.agent_id,
                answer=majority,
                reasoning=f"Refined to majority answer: {majority}",
                confidence=round(self._rng.uniform(0.80, 0.99), 3),
            )
        return await self.generate_solution(self.task or "refine")

    async def _llm_solution(self, task: str) -> Solution:
        """Call real LLM if client is available."""
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer concisely in one line."},
                    {"role": "user", "content": task},
                ],
                max_tokens=64,
            )
            answer = response.choices[0].message.content.strip()
            return Solution(
                agent_id=self.agent_id,
                answer=answer,
                reasoning="LLM response",
                confidence=0.85,
            )
        except Exception as exc:  # noqa: BLE001
            return Solution(
                agent_id=self.agent_id,
                answer=self._pick_answer(),
                reasoning=f"LLM failed ({exc}), using fallback",
                confidence=0.5,
            )

    def _pick_answer(self) -> str:
        if self.is_outlier:
            return "OUTLIER_WRONG_ANSWER"
        return self._rng.choice(self.answer_variants) if self.answer_variants else self.correct_answer

