"""Predictors for sports outcomes (Elo, Dixon-Coles, LLM, Aegean, Monte Carlo)."""

from aegeanbench.sports.predictors.aegean import AegeanPredictor
from aegeanbench.sports.predictors.base import Predictor
from aegeanbench.sports.predictors.dixon_coles import DixonColesPredictor
from aegeanbench.sports.predictors.elo import EloPredictor
from aegeanbench.sports.predictors.llm import (
    AnthropicClient,
    LLMClient,
    LLMPredictor,
    MockLLMClient,
    OpenAICompatibleClient,
    make_llm_predictor_from_env,
)
from aegeanbench.sports.predictors.monte_carlo import MonteCarloSimulator

__all__ = [
    "Predictor",
    "EloPredictor",
    "DixonColesPredictor",
    "MonteCarloSimulator",
    "LLMClient",
    "MockLLMClient",
    "OpenAICompatibleClient",
    "AnthropicClient",
    "LLMPredictor",
    "make_llm_predictor_from_env",
    "AegeanPredictor",
]
