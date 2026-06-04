"""
Investment Backtest Benchmark Cases.

Historical point-in-time investment analysis cases used to evaluate whether
an investment agent can produce useful recommendations using only information
available at the analysis date.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from aegeanbench.core.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkSuite,
    Difficulty,
    HistoricalMarketContext,
    InvestmentGroundTruth,
)


def _suite_from_cases(cases: List[BenchmarkCase], *, name: str, description: str) -> BenchmarkSuite:
    return BenchmarkSuite(
        name=name,
        description=description,
        version="0.1.0",
        cases=cases,
        metadata={"markets": ["US"], "asset_types": ["equity"], "horizon": "20d"},
    )


def load_investment_cases_from_file(path: str) -> BenchmarkSuite:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")

    if file_path.suffix.lower() == ".jsonl":
        payloads = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "cases" in parsed:
            payloads = parsed["cases"]
        elif isinstance(parsed, list):
            payloads = parsed
        else:
            raise ValueError("Investment dataset file must be a JSON array, JSONL stream, or {\"cases\": [...]} object")

    cases = [BenchmarkCase.model_validate(item) for item in payloads]
    for case in cases:
        if case.category != BenchmarkCategory.INVESTMENT:
            raise ValueError(f"Case {case.case_id} is not an investment case")

    return _suite_from_cases(
        cases,
        name=f"Investment Backtest Suite ({file_path.name})",
        description="Investment backtest cases loaded from a file.",
    )


def load_investment_suite() -> BenchmarkSuite:
    cases = [
        BenchmarkCase(
            case_id="INV-US-001",
            name="Apple post-earnings continuation check",
            description="Evaluate whether the system identifies positive momentum with manageable risk.",
            category=BenchmarkCategory.INVESTMENT,
            difficulty=Difficulty.MEDIUM,
            tags=["investment", "us", "equity", "technology", "momentum"],
            investment_request={
                "mode": "auto",
                "asset": {
                    "symbol": "AAPL",
                    "market": "US",
                    "asset_type": "equity",
                    "display_name": "Apple Inc.",
                },
                "timeframe": {
                    "analysis_date": "2024-02-05T00:00:00Z",
                    "lookback_window_days": 90,
                    "horizon": "1m",
                },
                "risk_profile": "balanced",
                "objective": "alpha",
                "market_snapshot": "AAPL is trading near local highs after earnings, with resilient mega-cap sentiment.",
                "public_facts": [
                    "Latest quarter beat consensus revenue expectations.",
                    "Services revenue remains a key margin support.",
                    "Valuation is above long-term average but supported by quality premium.",
                ],
                "constraints": {
                    "allowed_actions": ["buy", "hold", "watch"],
                    "max_single_position_pct": 0.15,
                },
                "portfolio_context": {
                    "current_total_exposure_pct": 0.55,
                    "max_total_exposure_pct": 0.80,
                },
                "metadata": {
                    "benchmark_case_id": "INV-US-001",
                },
            },
            historical_context=HistoricalMarketContext(
                market={
                    "price": 187.68,
                    "change_pct": 0.008,
                    "52w_high": 199.62,
                    "52w_low": 143.90,
                    "volume": 55200000,
                },
                fundamentals={
                    "market_cap": 2890000000000,
                    "pe_ttm": 29.8,
                    "gross_margin": 0.455,
                    "revenue_growth": 0.021,
                },
                news=[
                    {
                        "title": "Apple earnings top estimates as services stay strong",
                        "source": "Reuters",
                        "provider": "benchmark_fixture",
                        "polarity": "supportive",
                    },
                    {
                        "title": "Analysts debate upside after large-cap tech rerating",
                        "source": "Bloomberg",
                        "provider": "benchmark_fixture",
                        "polarity": "neutral",
                    },
                ],
                public_facts=[
                    "Buybacks remain an important support for EPS growth.",
                    "iPhone demand concern persists in some regions.",
                ],
                provider_status={"fixture": {"status": "ok", "signals": []}},
                provider_signals=[],
            ),
            investment_ground_truth=InvestmentGroundTruth(
                forward_return_20d=0.041,
                benchmark_return_20d=0.019,
                max_drawdown_20d=0.032,
                realized_vol_20d=0.184,
                direction_label_20d="bullish",
            ),
            metadata={"benchmark_index": "SPY", "regime": "soft_landing"},
        ),
        BenchmarkCase(
            case_id="INV-US-002",
            name="Tesla valuation stress test",
            description="Check whether the system flags downside risk when valuation and sentiment are fragile.",
            category=BenchmarkCategory.INVESTMENT,
            difficulty=Difficulty.HARD,
            tags=["investment", "us", "equity", "auto", "high-beta"],
            investment_request={
                "mode": "auto",
                "asset": {
                    "symbol": "TSLA",
                    "market": "US",
                    "asset_type": "equity",
                    "display_name": "Tesla Inc.",
                },
                "timeframe": {
                    "analysis_date": "2024-04-10T00:00:00Z",
                    "lookback_window_days": 120,
                    "horizon": "1m",
                },
                "risk_profile": "balanced",
                "objective": "defensive",
                "market_snapshot": "TSLA has broken below medium-term support as delivery concerns pressure sentiment.",
                "public_facts": [
                    "Recent delivery numbers missed market expectations.",
                    "Price cuts are raising margin concerns.",
                    "The stock remains highly volatile relative to the broad market.",
                ],
                "constraints": {
                    "no_short": True,
                    "allowed_actions": ["hold", "sell", "watch"],
                },
                "portfolio_context": {
                    "current_total_exposure_pct": 0.48,
                    "max_total_exposure_pct": 0.75,
                },
                "metadata": {
                    "benchmark_case_id": "INV-US-002",
                },
            },
            historical_context=HistoricalMarketContext(
                market={
                    "price": 171.76,
                    "change_pct": -0.024,
                    "52w_high": 299.29,
                    "52w_low": 138.80,
                    "volume": 103400000,
                },
                fundamentals={
                    "market_cap": 548000000000,
                    "pe_ttm": 41.5,
                    "gross_margin": 0.174,
                    "revenue_growth": 0.031,
                },
                news=[
                    {
                        "title": "Tesla deliveries miss estimates, intensifying margin fears",
                        "source": "Reuters",
                        "provider": "benchmark_fixture",
                        "polarity": "negative",
                    },
                    {
                        "title": "EV competition grows as discounting spreads across the sector",
                        "source": "WSJ",
                        "provider": "benchmark_fixture",
                        "polarity": "negative",
                    },
                ],
                public_facts=[
                    "The name remains retail-owned and sentiment-sensitive.",
                    "Macro rate cuts have not yet repaired demand concerns.",
                ],
                provider_status={"fixture": {"status": "ok", "signals": ["HIGH_VOLATILITY"]}},
                provider_signals=["HIGH_VOLATILITY"],
            ),
            investment_ground_truth=InvestmentGroundTruth(
                forward_return_20d=-0.118,
                benchmark_return_20d=-0.014,
                max_drawdown_20d=0.136,
                realized_vol_20d=0.402,
                direction_label_20d="bearish",
            ),
            metadata={"benchmark_index": "SPY", "regime": "risk_off"},
        ),
        BenchmarkCase(
            case_id="INV-US-003",
            name="Microsoft steady compounder neutral-upside case",
            description="A moderate case where the model should avoid overreacting and prefer a balanced recommendation.",
            category=BenchmarkCategory.INVESTMENT,
            difficulty=Difficulty.EASY,
            tags=["investment", "us", "equity", "quality", "large-cap"],
            investment_request={
                "mode": "auto",
                "asset": {
                    "symbol": "MSFT",
                    "market": "US",
                    "asset_type": "equity",
                    "display_name": "Microsoft Corp.",
                },
                "timeframe": {
                    "analysis_date": "2024-03-15T00:00:00Z",
                    "lookback_window_days": 90,
                    "horizon": "1m",
                },
                "risk_profile": "balanced",
                "objective": "balanced",
                "market_snapshot": "MSFT trends positively but trades at a premium after a strong AI-driven rerating.",
                "public_facts": [
                    "Cloud demand remains robust.",
                    "AI narrative supports sentiment but valuation is no longer cheap.",
                    "Balance sheet quality remains strong.",
                ],
                "constraints": {
                    "allowed_actions": ["buy", "hold", "watch"],
                    "max_single_position_pct": 0.12,
                },
                "portfolio_context": {
                    "current_total_exposure_pct": 0.60,
                    "max_total_exposure_pct": 0.85,
                },
                "metadata": {
                    "benchmark_case_id": "INV-US-003",
                },
            },
            historical_context=HistoricalMarketContext(
                market={
                    "price": 416.42,
                    "change_pct": 0.004,
                    "52w_high": 430.82,
                    "52w_low": 275.37,
                    "volume": 21000000,
                },
                fundamentals={
                    "market_cap": 3090000000000,
                    "pe_ttm": 35.2,
                    "gross_margin": 0.689,
                    "revenue_growth": 0.176,
                },
                news=[
                    {
                        "title": "Microsoft expands Copilot rollout across enterprise products",
                        "source": "CNBC",
                        "provider": "benchmark_fixture",
                        "polarity": "supportive",
                    },
                    {
                        "title": "Analysts debate whether AI upside is fully priced",
                        "source": "Bloomberg",
                        "provider": "benchmark_fixture",
                        "polarity": "neutral",
                    },
                ],
                public_facts=[
                    "Azure remains the key growth driver.",
                    "Premium valuation can limit near-term upside.",
                ],
                provider_status={"fixture": {"status": "ok", "signals": []}},
                provider_signals=[],
            ),
            investment_ground_truth=InvestmentGroundTruth(
                forward_return_20d=0.017,
                benchmark_return_20d=0.014,
                max_drawdown_20d=0.028,
                realized_vol_20d=0.153,
                direction_label_20d="neutral",
            ),
            metadata={"benchmark_index": "SPY", "regime": "quality_bid"},
        ),
    ]

    return _suite_from_cases(
        cases,
        name="AegeanBench Investment Backtest Suite",
        description="Historical point-in-time investment analysis cases for backtesting recommendation quality.",
    )

