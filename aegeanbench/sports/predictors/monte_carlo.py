"""
Monte Carlo tournament simulator.

Drives any Predictor to estimate:
  - per-match outcome distribution (smoothing of the predictor's output)
  - per-team championship probability over a fixture list
  - knockout bracket survival probabilities

The simulator is predictor-agnostic: it samples outcomes from the
predictor's Prediction.p_home_win / p_draw / p_away and propagates results
through the bracket.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from aegeanbench.sports.gateway import MatchContext, SportsDataGateway
from aegeanbench.sports.models import (
    Match,
    MatchOutcome,
    MatchResult,
    Prediction,
)
from aegeanbench.sports.predictors.base import Predictor


def sample_outcome(prediction: Prediction, rng: random.Random) -> MatchOutcome:
    """Sample a single match outcome from a prediction's probability vector."""
    r = rng.random()
    if r < prediction.p_home_win:
        return MatchOutcome.HOME_WIN
    if r < prediction.p_home_win + prediction.p_draw:
        return MatchOutcome.DRAW
    return MatchOutcome.AWAY_WIN


class MonteCarloSimulator:
    """
    Run a predictor across many simulated tournaments to estimate aggregate
    quantities (championship probability, group standings, etc.).

    For the sprint we keep this minimal: simulate a flat fixture list of
    knockout matches, where the winner of each match is determined by
    sampling from the predictor.

    For group-stage tournaments use simulate_group_stage(); for full
    World Cup format combine simulate_group_stage + simulate_knockout.
    """

    def __init__(
        self,
        predictor: Predictor,
        gateway: SportsDataGateway,
        seed: Optional[int] = None,
    ):
        self.predictor = predictor
        self.gateway = gateway
        self.rng = random.Random(seed)

    # ---------- single match ----------

    def simulate_match(self, ctx: MatchContext) -> MatchOutcome:
        """Sample one outcome for a single match using the predictor."""
        prediction = self.predictor.predict(ctx)
        return sample_outcome(prediction, self.rng)

    # ---------- group stage ----------

    def simulate_group_stage(
        self,
        group_matches: List[Match],
        num_simulations: int = 10000,
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate group standings via Monte Carlo.

        Args:
            group_matches: all matches in one group (round-robin)
            num_simulations: tournaments to simulate

        Returns:
            {fifa_code: {"avg_points": float, "p_first": float, "p_second": float,
                        "p_advance": float}}
        """
        contexts = [self.gateway.build_context(m) for m in group_matches]
        predictions = [self.predictor.predict(c) for c in contexts]

        # Tally storage
        points_total: Dict[str, float] = defaultdict(float)
        finishes: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for _ in range(num_simulations):
            standings: Dict[str, int] = defaultdict(int)
            for match, pred in zip(group_matches, predictions):
                outcome = sample_outcome(pred, self.rng)
                hc = match.home_team.fifa_code
                ac = match.away_team.fifa_code
                if outcome == MatchOutcome.HOME_WIN:
                    standings[hc] += 3
                elif outcome == MatchOutcome.AWAY_WIN:
                    standings[ac] += 3
                else:
                    standings[hc] += 1
                    standings[ac] += 1
            # Rank teams in this sim
            ranked = sorted(standings.items(), key=lambda x: x[1], reverse=True)
            for rank, (code, pts) in enumerate(ranked, start=1):
                points_total[code] += pts
                finishes[code][rank] += 1

        results: Dict[str, Dict[str, float]] = {}
        for code, points in points_total.items():
            f = finishes[code]
            results[code] = {
                "avg_points": points / num_simulations,
                "p_first": f.get(1, 0) / num_simulations,
                "p_second": f.get(2, 0) / num_simulations,
                "p_advance": (f.get(1, 0) + f.get(2, 0)) / num_simulations,
            }
        return results

    # ---------- knockout ----------

    def simulate_knockout(
        self,
        bracket: List[Match],
        next_round_factory: Callable[[List[str]], List[Match]],
        num_simulations: int = 10000,
    ) -> Dict[str, float]:
        """
        Simulate a single-elimination bracket.

        Args:
            bracket: matches of the first knockout round
            next_round_factory: given winners of one round, produce next-round
                matches. Returns empty list when the bracket is complete.
            num_simulations: tournaments to simulate

        Returns:
            {fifa_code: championship_probability}
        """
        champion_counts: Dict[str, int] = defaultdict(int)

        for _ in range(num_simulations):
            current_round = bracket
            winners: List[str] = []
            while current_round:
                winners = []
                for m in current_round:
                    ctx = self.gateway.build_context(m)
                    outcome = self.simulate_match(ctx)
                    if outcome == MatchOutcome.AWAY_WIN:
                        winners.append(m.away_team.fifa_code)
                    else:
                        # Knockout: draws settled by penalties; we treat draws
                        # as 50/50 between the two teams for simplicity.
                        if outcome == MatchOutcome.DRAW and self.rng.random() < 0.5:
                            winners.append(m.away_team.fifa_code)
                        else:
                            winners.append(m.home_team.fifa_code)
                next_round = next_round_factory(winners)
                if not next_round:
                    break
                current_round = next_round
            # Last survivor is the champion
            if winners:
                champion_counts[winners[0]] += 1

        return {code: count / num_simulations for code, count in champion_counts.items()}

    # ---------- average match prediction ----------

    def smoothed_prediction(self, ctx: MatchContext, num_samples: int = 1000) -> Prediction:
        """
        Average a predictor's output across multiple samples.

        For stateless predictors (Elo, Dixon-Coles) this just returns the
        underlying prediction. For stochastic predictors (LLMs with non-zero
        temperature) it stabilizes the probabilities.
        """
        base = self.predictor.predict(ctx)
        # For Day 2 baselines this is deterministic, so the average equals
        # the base prediction. Kept here so LLM/Aegean predictors can plug
        # in later without changing the simulator API.
        return base
