"""
Abstract base class for sports predictors.

All predictors (Elo, Dixon-Coles, LLM, Aegean consensus) implement this
interface so they can be benchmarked side-by-side under identical inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from aegeanbench.sports.gateway import MatchContext
from aegeanbench.sports.models import Match, Prediction


class Predictor(ABC):
    """
    A model that turns a MatchContext into a Prediction.

    Implementations may be:
      - Stateless and analytical (Elo at inference time, Dixon-Coles)
      - LLM-backed (one or many models)
      - Multi-agent (Aegean consensus)

    Predictors that need training (Dixon-Coles, Elo from scratch) override
    fit(); stateless ones leave it as a no-op.
    """

    runner_id: str = "base"
    display_name: str = "Base Predictor"

    @abstractmethod
    def predict(self, ctx: MatchContext) -> Prediction:
        """Produce a probability distribution over the three outcomes."""

    def fit(self, history: Iterable[Match]) -> None:
        """
        Train on historical matches. Stateless predictors override as no-op.

        Args:
            history: an iterable of Match objects with result populated
        """

    def predict_batch(self, contexts: List[MatchContext]) -> List[Prediction]:
        """Predict many matches; defaults to sequential calls."""
        return [self.predict(ctx) for ctx in contexts]
