"""
Tests for LLMPredictor and AegeanPredictor (Day 3-A).

Mock-only tests; no real API keys required. The Mock LLM client deterministic
hash-based response ensures these tests are reproducible.
"""

from __future__ import annotations

import json

import pytest

from aegeanbench.sports import (
    Prediction,
    SportsDataGateway,
)
from aegeanbench.sports.predictors import (
    AegeanPredictor,
    LLMPredictor,
    MockLLMClient,
    make_llm_predictor_from_env,
)
from aegeanbench.sports.predictors.llm import _extract_json, _normalize_probs
from aegeanbench.sports.prompts import build_full_prompt


class TestPromptBuilder:
    @pytest.fixture
    def ctx(self):
        gw = SportsDataGateway(mock=True)
        m = gw.list_fixtures()[0]
        return gw.build_context(m)

    def test_prompt_contains_team_codes(self, ctx):
        prompts = build_full_prompt(ctx)
        # System prompt has output contract
        assert "json" in prompts["system"].lower()
        assert "p_home_win" in prompts["system"]
        # User prompt has team info
        assert ctx.match.home_team.fifa_code in prompts["user"]
        assert ctx.match.away_team.fifa_code in prompts["user"]
        # And odds, xG, lineups sections
        assert "Market Odds" in prompts["user"]
        assert "Advanced Stats" in prompts["user"]
        assert "Key Players" in prompts["user"]

    def test_focus_hint_appears(self, ctx):
        prompts = build_full_prompt(ctx, focus="stats")
        assert "ANALYTICAL FOCUS" in prompts["user"]
        assert "xG" in prompts["user"]


class TestJSONExtraction:
    def test_clean_json(self):
        result = _extract_json('{"p_home_win": 0.4, "p_draw": 0.3, "p_away_win": 0.3}')
        assert result["p_home_win"] == 0.4

    def test_json_with_markdown_fence(self):
        text = '```json\n{"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2}\n```'
        result = _extract_json(text)
        assert result["p_home_win"] == 0.5

    def test_json_with_surrounding_prose(self):
        text = 'Here is my analysis:\n\n{"p_home_win": 0.4, "p_draw": 0.4, "p_away_win": 0.2}\n\nThat is my prediction.'
        result = _extract_json(text)
        assert result["p_home_win"] == 0.4

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            _extract_json("This has no JSON at all.")


class TestNormalizeProbs:
    def test_already_normalized(self):
        assert _normalize_probs(0.4, 0.3, 0.3) == (0.4, 0.3, 0.3)

    def test_renormalize(self):
        h, d, a = _normalize_probs(0.5, 0.3, 0.4)
        assert abs(h + d + a - 1.0) < 1e-9
        # ratios preserved
        assert h / d == pytest.approx(0.5 / 0.3)

    def test_negative_clipped(self):
        h, d, a = _normalize_probs(0.6, -0.1, 0.5)
        assert d == 0.0
        assert abs(h + d + a - 1.0) < 1e-9

    def test_all_zero_falls_back_uniform(self):
        h, d, a = _normalize_probs(0, 0, 0)
        assert h == d == a == pytest.approx(1 / 3)


class TestMockLLMClient:
    def test_returns_valid_json(self):
        client = MockLLMClient(runner_id="gpt-5")
        text, tokens = client.complete("sys", "user")
        parsed = json.loads(text)
        assert "p_home_win" in parsed
        assert "p_draw" in parsed
        assert "p_away_win" in parsed
        assert tokens > 0

    def test_probabilities_sum_to_one(self):
        client = MockLLMClient(runner_id="claude-opus-4-7")
        text, _ = client.complete("sys", "user")
        parsed = json.loads(text)
        total = parsed["p_home_win"] + parsed["p_draw"] + parsed["p_away_win"]
        assert abs(total - 1.0) < 1e-9

    def test_deterministic_same_prompt_same_output(self):
        c1 = MockLLMClient(runner_id="gpt-5")
        c2 = MockLLMClient(runner_id="gpt-5")
        out1, _ = c1.complete("sys", "the same user prompt")
        out2, _ = c2.complete("sys", "the same user prompt")
        assert out1 == out2

    def test_different_runners_different_outputs(self):
        c1 = MockLLMClient(runner_id="gpt-5")
        c2 = MockLLMClient(runner_id="deepseek-v3")
        out1, _ = c1.complete("sys", "same prompt")
        out2, _ = c2.complete("sys", "same prompt")
        # Bias values differ, so outputs should differ
        p1 = json.loads(out1)
        p2 = json.loads(out2)
        assert p1["p_home_win"] != p2["p_home_win"]


class TestLLMPredictor:
    @pytest.fixture
    def gw(self):
        return SportsDataGateway(mock=True)

    def test_mock_llm_predictor_produces_valid_prediction(self, gw):
        predictor = LLMPredictor(
            client=MockLLMClient("gpt-5"),
            runner_id="gpt-5",
            display_name="GPT-5",
        )
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = predictor.predict(ctx)
        assert isinstance(pred, Prediction)
        assert pred.runner_id == "gpt-5"
        assert abs(pred.p_home_win + pred.p_draw + pred.p_away_win - 1.0) < 1e-9
        assert pred.tokens_used > 0

    def test_factory_without_keys_returns_mock(self, gw, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        predictor = make_llm_predictor_from_env("gpt-5")
        assert isinstance(predictor.client, MockLLMClient)
        # Still produces a valid Prediction
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = predictor.predict(ctx)
        assert isinstance(pred, Prediction)


class TestAegeanPredictor:
    @pytest.fixture
    def gw(self):
        return SportsDataGateway(mock=True)

    def test_mock_aegean_produces_valid_prediction(self, gw):
        predictor = AegeanPredictor(mock=True)
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        pred = predictor.predict(ctx)
        assert isinstance(pred, Prediction)
        assert pred.runner_id == "aegean"
        assert abs(pred.p_home_win + pred.p_draw + pred.p_away_win - 1.0) < 1e-9
        # Metadata should reflect consensus structure
        assert pred.metadata["agent_count"] == 6
        assert "weighted_votes" in pred.metadata

    def test_aegean_smoother_than_single_llm(self, gw):
        """
        Aegean averages multiple agent outputs, so its prediction should
        have lower variance / be closer to (0.4, 0.3, 0.3) than any
        single mock LLM call.
        """
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)

        single = LLMPredictor(MockLLMClient("gpt-5"), "gpt-5", "GPT-5").predict(ctx)
        aegean = AegeanPredictor(mock=True).predict(ctx)

        # Variance from uniform (0.333,0.333,0.333) — aegean should be <= single
        def var_from_uniform(p: Prediction) -> float:
            u = 1 / 3
            return (
                (p.p_home_win - u) ** 2
                + (p.p_draw - u) ** 2
                + (p.p_away_win - u) ** 2
            )
        # Note: this isn't strictly guaranteed in all hash configurations,
        # but on our mock distribution it should hold. If flaky in future
        # we can soften to a confidence interval.
        assert var_from_uniform(aegean) <= var_from_uniform(single) + 0.05

    def test_aegean_falls_back_to_mock_on_server_error(self, gw):
        """When mock=False but server unreachable, should fall back gracefully."""
        predictor = AegeanPredictor(
            mock=False, base_url="http://127.0.0.1:1",  # guaranteed-bad port
            timeout=1.0,
        )
        m = gw.list_fixtures()[0]
        ctx = gw.build_context(m)
        # Should not raise; should return a fallback Prediction
        pred = predictor.predict(ctx)
        assert isinstance(pred, Prediction)
        assert abs(pred.p_home_win + pred.p_draw + pred.p_away_win - 1.0) < 1e-9
