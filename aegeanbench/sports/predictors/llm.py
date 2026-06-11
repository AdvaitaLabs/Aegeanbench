"""
LLM-backed predictor.

Wraps any chat-completion LLM into the Predictor interface. Supports:
  - MockLLMClient: deterministic offline mock for sprint dev/tests
  - OpenAICompatibleClient: GPT-5, DeepSeek (OpenRouter-style endpoints)
  - AnthropicClient: Claude

Mock mode is the default; switching to real LLMs requires passing the
appropriate client to LLMPredictor.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from aegeanbench.sports.gateway import MatchContext
from aegeanbench.sports.models import Prediction
from aegeanbench.sports.predictors.base import Predictor
from aegeanbench.sports.prompts import build_full_prompt

logger = logging.getLogger(__name__)


# ----------------------------- LLM clients -----------------------------


class LLMClient(ABC):
    """A minimal chat-completion interface."""

    @abstractmethod
    def complete(self, system: str, user: str) -> Tuple[str, int]:
        """
        Return (raw_text, total_tokens).

        raw_text is the model's free-form response. tokens is the sum of
        prompt and completion tokens (0 if unknown).
        """


class MockLLMClient(LLMClient):
    """
    Deterministic offline mock.

    Generates plausible JSON responses keyed by (runner_id, user_prompt hash).
    Different runner_ids produce slightly different probability biases so
    we can demo head-to-head comparisons without real API calls.
    """

    # Per-runner bias toward home_win (in [-0.1, +0.1]). Tweakable.
    RUNNER_BIASES: Dict[str, float] = {
        "gpt-5": 0.02,
        "claude-opus-4-7": -0.01,
        "deepseek-v3": -0.03,
        "grok-4": 0.04,
        "single-llm-mock": 0.0,
    }

    def __init__(self, runner_id: str = "single-llm-mock"):
        self.runner_id = runner_id

    def complete(self, system: str, user: str) -> Tuple[str, int]:
        # Hash user prompt to produce reproducible-but-varying probabilities
        h = hashlib.sha256(user.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)
        # Base probabilities centered around (0.40, 0.28, 0.32)
        # with a small per-match perturbation derived from the hash
        perturb_home = ((seed % 100) / 1000.0) - 0.05    # in [-0.05, +0.05]
        perturb_away = (((seed // 100) % 100) / 1000.0) - 0.05
        bias = self.RUNNER_BIASES.get(self.runner_id, 0.0)

        p_home = max(0.15, min(0.75, 0.40 + perturb_home + bias))
        p_away = max(0.15, min(0.75, 0.32 + perturb_away))
        p_draw = max(0.10, 1.0 - p_home - p_away)
        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total

        confidence = round(max(p_home, p_draw, p_away), 2)
        response = {
            "p_home_win": round(p_home, 4),
            "p_draw": round(p_draw, 4),
            "p_away_win": round(p_away, 4),
            "confidence": confidence,
            "rationale": f"[mock {self.runner_id}] perturbation-based offline response",
            "key_factors": ["mock factor 1", "mock factor 2"],
        }
        return json.dumps(response), 1200  # fake token count


class OpenAICompatibleClient(LLMClient):
    """
    OpenAI-compatible chat completion client.

    Works with:
      - OpenAI (api.openai.com)
      - DeepSeek (api.deepseek.com)
      - OpenRouter (openrouter.ai/api/v1)
      - Any other endpoint that speaks the OpenAI /v1/chat/completions schema
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        temperature: float = 0.3,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, system: str, user: str) -> Tuple[str, int]:
        import requests   # imported lazily so mock-only paths don't need it

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        return text, tokens


class AnthropicClient(LLMClient):
    """Claude / Anthropic chat completion client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-7",
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 30.0,
        max_tokens: int = 1024,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> Tuple[str, int]:
        import requests

        url = f"{self.base_url}/messages"
        payload = {
            "model": self.model,
            "system": system,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        usage = data.get("usage", {})
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return text, tokens


# ----------------------------- Predictor -----------------------------


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly pull a JSON object out of an LLM response.

    Tries direct json.loads first; falls back to regex extraction of the
    first {...} block. Raises ValueError on hard failure.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip Markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort A: find first balanced {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort B: response was TRUNCATED by max_tokens. The JSON
    # never closed, so no brace match. But the numeric fields land at
    # the START of the JSON before the long `rationale`, so we can
    # still pull them out with per-field regexes. Better to surface a
    # half-parsed result than fall all the way back to mock.
    salvaged: Dict[str, Any] = {}
    field_regexes = {
        "p_home_win": r'"p_home_win"\s*:\s*([\d.]+)',
        "p_draw":     r'"p_draw"\s*:\s*([\d.]+)',
        "p_away_win": r'"p_away_win"\s*:\s*([\d.]+)',
        "confidence": r'"confidence"\s*:\s*([\d.]+)',
    }
    for key, pattern in field_regexes.items():
        m = re.search(pattern, text)
        if m:
            try:
                salvaged[key] = float(m.group(1))
            except ValueError:
                pass
    # Salvage rationale (string) too, even if it's truncated
    rat_match = re.search(r'"rationale"\s*:\s*"([^"]*)', text, re.DOTALL)
    if rat_match:
        salvaged["rationale"] = rat_match.group(1).strip()
    if "p_home_win" in salvaged and "p_away_win" in salvaged:
        # Got the essentials. Mark as salvaged so callers can flag it.
        salvaged.setdefault("rationale", "(rationale truncated)")
        salvaged["_salvaged_from_truncation"] = True
        return salvaged

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def _normalize_probs(p_home: float, p_draw: float, p_away: float) -> Tuple[float, float, float]:
    """Clip to non-negative and renormalize so the three sum to exactly 1.0."""
    p_home = max(0.0, p_home)
    p_draw = max(0.0, p_draw)
    p_away = max(0.0, p_away)
    total = p_home + p_draw + p_away
    if total == 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (p_home / total, p_draw / total, p_away / total)


class LLMPredictor(Predictor):
    """
    Generic LLM-backed predictor.

    Stateless: each predict() builds a prompt, calls the client, parses JSON,
    and returns a Prediction. No training step needed.
    """

    def __init__(
        self,
        client: LLMClient,
        runner_id: str,
        display_name: str,
        focus: Optional[str] = None,
    ):
        self.client = client
        self.runner_id = runner_id
        self.display_name = display_name
        self.focus = focus

    def predict(self, ctx: MatchContext) -> Prediction:
        prompts = build_full_prompt(ctx, focus=self.focus)
        start = time.perf_counter()
        try:
            raw, tokens = self.client.complete(prompts["system"], prompts["user"])
            parsed = _extract_json(raw)
        except Exception as e:
            logger.warning("%s LLM call failed (%s); returning uniform fallback", self.runner_id, e)
            latency_ms = int((time.perf_counter() - start) * 1000)
            return Prediction(
                match_id=ctx.match.match_id,
                runner_id=self.runner_id,
                p_home_win=1 / 3,
                p_draw=1 / 3,
                p_away_win=1 / 3,
                confidence=0.0,
                rationale=f"[fallback] LLM error: {e}",
                latency_ms=latency_ms,
                tokens_used=0,
                metadata={"error": str(e)},
            )

        p_home, p_draw, p_away = _normalize_probs(
            float(parsed.get("p_home_win", 0)),
            float(parsed.get("p_draw", 0)),
            float(parsed.get("p_away_win", 0)),
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        return Prediction(
            match_id=ctx.match.match_id,
            runner_id=self.runner_id,
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            confidence=float(parsed.get("confidence", max(p_home, p_draw, p_away))),
            rationale=str(parsed.get("rationale", "")),
            latency_ms=latency_ms,
            tokens_used=tokens,
            metadata={
                "model": self.runner_id,
                "key_factors": parsed.get("key_factors", []),
                "focus": self.focus,
            },
        )


# ----------------------------- Factory -----------------------------


def make_llm_predictor_from_env(
    runner_id: str,
    display_name: Optional[str] = None,
    focus: Optional[str] = None,
) -> LLMPredictor:
    """
    Build an LLMPredictor from environment variables.

    Selects client based on runner_id:
      gpt-5 / gpt-* → OpenAI (OPENAI_API_KEY)
      claude-* → Anthropic (ANTHROPIC_API_KEY)
      deepseek-* → DeepSeek (DEEPSEEK_API_KEY)
      Anything else → MockLLMClient
    """
    display_name = display_name or runner_id

    if runner_id.startswith("gpt-"):
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return LLMPredictor(
                OpenAICompatibleClient(api_key=key, model=runner_id),
                runner_id=runner_id, display_name=display_name, focus=focus,
            )
    elif runner_id.startswith("claude-"):
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            return LLMPredictor(
                AnthropicClient(api_key=key, model=runner_id),
                runner_id=runner_id, display_name=display_name, focus=focus,
            )
    elif runner_id.startswith("deepseek"):
        key = os.getenv("DEEPSEEK_API_KEY")
        if key:
            return LLMPredictor(
                OpenAICompatibleClient(
                    api_key=key,
                    model=runner_id,
                    base_url="https://api.deepseek.com/v1",
                ),
                runner_id=runner_id, display_name=display_name, focus=focus,
            )

    logger.info("No API key found for %s, using MockLLMClient", runner_id)
    return LLMPredictor(
        MockLLMClient(runner_id=runner_id),
        runner_id=runner_id, display_name=display_name, focus=focus,
    )
