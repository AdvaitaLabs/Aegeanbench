"""Predictors for sports outcomes (Elo, Dixon-Coles, Monte Carlo)."""

from aegeanbench.sports.predictors.base import Predictor
from aegeanbench.sports.predictors.dixon_coles import DixonColesPredictor
from aegeanbench.sports.predictors.elo import EloPredictor
from aegeanbench.sports.predictors.monte_carlo import MonteCarloSimulator

__all__ = [
    "Predictor",
    "EloPredictor",
    "DixonColesPredictor",
    "MonteCarloSimulator",
]
