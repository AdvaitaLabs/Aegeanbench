"""
Arena: head-to-head model comparison for upcoming matches.

This package is fully additive — it does not modify any existing
endpoint or predictor. It exposes two new read endpoints (wired in the
reporter) that, for each upcoming match, return every configured model's
pre-match prediction side by side:

  * aegean-consensus — OUR model. Runs on the enriched MatchContext
    (odds / xG / lineups / form pulled by the SportsDataGateway). The
    extra data is for our use only.
  * benchmark models — competitors (Gemini / Claude / GPT / Qwen / Kimi
    …). Called DIRECTLY with only the team names + kickoff; they predict
    from their own knowledge. Configured via BENCHMARK_MODELS in .env.

Every model returns the SAME rich schema (win probabilities, score
distribution, full reasoning, analysis sections, lineup, predicted event
timeline, match stats) so the front-end can render them uniformly.
"""

from aegeanbench.sports.arena.config import BenchmarkModel, load_benchmark_models

__all__ = ["BenchmarkModel", "load_benchmark_models"]
