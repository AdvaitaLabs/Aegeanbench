"""
Elo-rating based football predictor.

Implements the standard World Football Elo Ratings algorithm (eloratings.net)
with adjustments for:
  - Home advantage (additive bonus to home team rating during prediction)
  - Match importance (K-factor scales with tournament stage)
  - Goal margin (winning by more shifts ratings further)

Draw probability is derived empirically from the rating gap rather than
the classic Elo two-outcome formula.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, Optional

from aegeanbench.sports.gateway import MatchContext
from aegeanbench.sports.models import (
    CompetitionStage,
    Match,
    MatchOutcome,
    Prediction,
)
from aegeanbench.sports.predictors.base import Predictor


# K-factor by tournament stage (eloratings.net convention)
K_FACTOR_BY_STAGE = {
    CompetitionStage.FRIENDLY: 20,
    CompetitionStage.QUALIFIER: 30,
    CompetitionStage.GROUP: 50,
    CompetitionStage.ROUND_OF_16: 55,
    CompetitionStage.QUARTER_FINAL: 60,
    CompetitionStage.SEMI_FINAL: 60,
    CompetitionStage.THIRD_PLACE: 55,
    CompetitionStage.FINAL: 60,
}

DEFAULT_RATING = 1500.0
HOME_ADVANTAGE = 100.0          # Elo points added to home team during prediction
DRAW_TUNING_SIGMA = 200.0       # spread of the draw band in Elo points


def expected_win_prob(rating_diff: float) -> float:
    """
    Classic Elo win probability for the higher-rated side.

    rating_diff = (home_rating + home_advantage) - away_rating
    """
    return 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))


def three_way_probs(rating_diff: float) -> Dict[MatchOutcome, float]:
    """
    Convert a rating differential into (home_win, draw, away_win).

    The draw band uses a Gaussian-shaped function centered on rating_diff=0.
    Larger gaps -> smaller draw probability. Calibrated so that two equally
    rated teams give roughly (0.35, 0.30, 0.35), matching empirical
    international football base rates.
    """
    # Probability of a "decisive" result (not a draw)
    p_decisive_total = 1.0 - 0.30 * math.exp(-(rating_diff ** 2) / (2 * DRAW_TUNING_SIGMA ** 2))
    p_home_win_within_decisive = expected_win_prob(rating_diff)

    p_home = p_decisive_total * p_home_win_within_decisive
    p_away = p_decisive_total * (1 - p_home_win_within_decisive)
    p_draw = 1.0 - p_decisive_total

    # Normalize against any floating point drift
    total = p_home + p_draw + p_away
    return {
        MatchOutcome.HOME_WIN: p_home / total,
        MatchOutcome.DRAW: p_draw / total,
        MatchOutcome.AWAY_WIN: p_away / total,
    }


def goal_margin_multiplier(home_goals: int, away_goals: int) -> float:
    """
    World Football Elo goal-difference multiplier.
      diff 1 -> 1.0
      diff 2 -> 1.5
      diff 3+ -> 1.75 + (diff - 3) / 8
    """
    diff = abs(home_goals - away_goals)
    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    return 1.75 + (diff - 3) / 8.0


class EloPredictor(Predictor):
    """
    Stateful Elo predictor.

    Initial ratings come from either:
      1. Team.elo_rating field on the Team object (preferred, populated from
         clubelo / eloratings scraping)
      2. fit() on a history list, starting from DEFAULT_RATING
      3. DEFAULT_RATING for any unseen team at prediction time
    """

    runner_id = "elo"
    display_name = "Elo Ratings"

    def __init__(
        self,
        home_advantage: float = HOME_ADVANTAGE,
        default_rating: float = DEFAULT_RATING,
    ):
        self.ratings: Dict[str, float] = {}
        self.home_advantage = home_advantage
        self.default_rating = default_rating

    # ---------- training ----------

    def fit(self, history: Iterable[Match]) -> None:
        """
        Walk through historical matches in chronological order, updating
        ratings after each match using the standard Elo update rule.
        """
        ordered = sorted(history, key=lambda m: m.kickoff_at)
        for m in ordered:
            if m.result is None:
                continue
            self._update_ratings(m)

    def _update_ratings(self, m: Match) -> None:
        home_code = m.home_team.fifa_code
        away_code = m.away_team.fifa_code

        # Initialize from Team.elo_rating if present, else default
        if home_code not in self.ratings:
            self.ratings[home_code] = m.home_team.elo_rating or self.default_rating
        if away_code not in self.ratings:
            self.ratings[away_code] = m.away_team.elo_rating or self.default_rating

        rating_diff = (self.ratings[home_code] + self.home_advantage) - self.ratings[away_code]
        expected_home = expected_win_prob(rating_diff)

        # Actual outcome score from home team's perspective
        if m.result.outcome == MatchOutcome.HOME_WIN:
            actual_home = 1.0
        elif m.result.outcome == MatchOutcome.AWAY_WIN:
            actual_home = 0.0
        else:
            actual_home = 0.5

        k = K_FACTOR_BY_STAGE.get(m.stage, 30)
        g = goal_margin_multiplier(m.result.home_goals, m.result.away_goals)
        delta = k * g * (actual_home - expected_home)

        self.ratings[home_code] += delta
        self.ratings[away_code] -= delta

    # ---------- inference ----------

    def get_rating(self, fifa_code: str, fallback: Optional[float] = None) -> float:
        """Look up a team's current Elo, falling back to seed or default."""
        if fifa_code in self.ratings:
            return self.ratings[fifa_code]
        return fallback if fallback is not None else self.default_rating

    def predict(self, ctx: MatchContext) -> Prediction:
        start = time.perf_counter()
        m = ctx.match
        home_rating = self.get_rating(m.home_team.fifa_code, m.home_team.elo_rating)
        away_rating = self.get_rating(m.away_team.fifa_code, m.away_team.elo_rating)

        rating_diff = (home_rating + self.home_advantage) - away_rating
        probs = three_way_probs(rating_diff)
        latency_ms = int((time.perf_counter() - start) * 1000)

        return Prediction(
            match_id=m.match_id,
            runner_id=self.runner_id,
            p_home_win=probs[MatchOutcome.HOME_WIN],
            p_draw=probs[MatchOutcome.DRAW],
            p_away_win=probs[MatchOutcome.AWAY_WIN],
            confidence=max(probs.values()),
            rationale=(
                f"Elo: home {home_rating:.0f} (+{self.home_advantage:.0f} HA) "
                f"vs away {away_rating:.0f}, diff {rating_diff:+.0f}"
            ),
            latency_ms=latency_ms,
            metadata={
                "home_rating": home_rating,
                "away_rating": away_rating,
                "rating_diff": rating_diff,
            },
        )
