"""
Benchmark model configuration, read from the BENCHMARK_MODELS env var.

Format (comma-separated, one entry per model):

    BENCHMARK_MODELS=model_id:Display Name:cost_usd, model_id2:Name2:cost

  - model_id    : the model name passed to the LLM endpoint (praka /
                  OpenAI-compatible). e.g. "gemini-3.1-pro-preview".
  - Display Name: shown in the arena UI. Optional — defaults to model_id.
  - cost_usd    : estimated $ per prediction, shown as the cost chip.
                  Optional — defaults to 0.0.

Example:
    BENCHMARK_MODELS=gemini-3.1-pro-preview:Gemini 3.1 Pro Preview:0.20,\
claude-opus-4-7:Claude Opus 4.7:0.18,gpt-5:GPT-5.4:0.15,\
deepseek-v4-pro:DeepSeek V4 Pro:0.03,qwen-3.7-max:Qwen3.7 Max:0.10,\
kimi-k2.6:Kimi K2.6:0.09

All benchmark models are called through the same OpenAI-compatible
endpoint as the consensus (OPENAI_BASE_URL / OPENAI_API_KEY), just with a
different `model` name — so a single Praka key serves all of them.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkModel:
    """One competitor model in the arena."""
    model_id: str          # name sent to the LLM endpoint
    display_name: str      # label for the UI
    cost_usd: float        # estimated $ per prediction (display only)
    runner_id: str         # stable id used in payloads / cache keys


def _slug(model_id: str) -> str:
    return model_id.strip().lower().replace(" ", "-")


def load_benchmark_models(env_value: str = None) -> List[BenchmarkModel]:
    """
    Parse BENCHMARK_MODELS into a list of BenchmarkModel. Returns [] when
    unset (the arena then shows only aegean-consensus). Malformed entries
    are skipped with a warning rather than crashing the endpoint.
    """
    raw = env_value if env_value is not None else os.getenv("BENCHMARK_MODELS", "")
    raw = (raw or "").strip()
    if not raw:
        return []

    models: List[BenchmarkModel] = []
    seen = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        model_id = parts[0]
        if not model_id:
            logger.warning("BENCHMARK_MODELS: skipping entry with empty model id: %r", chunk)
            continue
        display = parts[1] if len(parts) > 1 and parts[1] else model_id
        cost = 0.0
        if len(parts) > 2 and parts[2]:
            try:
                cost = float(parts[2])
            except ValueError:
                logger.warning("BENCHMARK_MODELS: bad cost %r for %s; using 0", parts[2], model_id)
        runner_id = _slug(model_id)
        if runner_id in seen:
            continue
        seen.add(runner_id)
        models.append(BenchmarkModel(
            model_id=model_id, display_name=display, cost_usd=cost, runner_id=runner_id,
        ))
    return models
