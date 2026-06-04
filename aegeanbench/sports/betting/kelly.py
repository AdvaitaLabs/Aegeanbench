"""
Kelly Criterion sizing for binary bets.

Kelly's formula gives the bankroll fraction that maximizes long-run growth
when betting at fixed decimal odds with a known win probability:

    f* = (p * b - q) / b

where
    p = probability of winning (from model)
    b = net decimal odds = decimal_odds - 1  (profit per unit staked on win)
    q = 1 - p

When f* <= 0 the bet has negative edge and should be skipped.

In practice we use FRACTIONAL Kelly (typically 0.25 of full Kelly) because:
  - Model probabilities are uncertain; full Kelly is volatile.
  - The closed-form assumes infinite bets at the same edge, which we won't
    actually have during a 64-match World Cup.
  - Fractional Kelly trades off ~25% expected growth for materially lower
    drawdown risk.

This module is intentionally pure-math: it does not know about Match or
Prediction objects. Higher layers (value.py / portfolio.py) compose these
primitives with domain types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KellyResult:
    """Result of a single Kelly calculation."""
    full_kelly: float        # f* before fractional scaling, capped at [0, 1]
    fractional_kelly: float  # full_kelly * fraction, the recommended stake fraction
    edge: float              # p_win * decimal_odds - 1 (>=0 to be a value bet)
    has_edge: bool           # True iff edge > 0
    growth_rate: float       # Expected log-growth per bet at full Kelly

    @property
    def stake_fraction(self) -> float:
        """Convenience alias for fractional_kelly (the actionable number)."""
        return self.fractional_kelly


def kelly_fraction(
    p_win: float,
    decimal_odds: float,
    fraction: float = 0.25,
    cap: float = 0.20,
) -> KellyResult:
    """
    Compute Kelly-recommended stake fraction.

    Args:
        p_win: Model probability of winning, in [0, 1].
        decimal_odds: Bookmaker decimal odds (e.g., 2.10 means win 1.10 net).
        fraction: Fractional Kelly multiplier. 1.0 = full Kelly, 0.25 = quarter
            Kelly (recommended). Must be in (0, 1].
        cap: Maximum stake fraction allowed regardless of Kelly output.
            Even full Kelly recommendations are clamped to this to bound risk.
            Default 0.20 = never stake more than 20% of bankroll on any single bet.

    Returns:
        KellyResult with both raw and fractional sizes, edge, and a
        has_edge flag for quick filtering.

    Raises:
        ValueError on invalid inputs (probabilities out of range, odds <= 1).
    """
    if not 0.0 <= p_win <= 1.0:
        raise ValueError(f"p_win must be in [0, 1], got {p_win}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if cap <= 0.0 or cap > 1.0:
        raise ValueError(f"cap must be in (0, 1], got {cap}")

    b = decimal_odds - 1.0
    q = 1.0 - p_win
    edge = p_win * decimal_odds - 1.0   # equivalent to p*b - q

    if edge <= 0.0:
        # No edge: do not bet. Growth rate undefined (would be negative).
        return KellyResult(
            full_kelly=0.0,
            fractional_kelly=0.0,
            edge=edge,
            has_edge=False,
            growth_rate=0.0,
        )

    full = (p_win * b - q) / b
    full = max(0.0, min(cap, full))
    fractional = full * fraction
    growth = _expected_growth(p_win, decimal_odds, full)

    return KellyResult(
        full_kelly=full,
        fractional_kelly=fractional,
        edge=edge,
        has_edge=True,
        growth_rate=growth,
    )


def _expected_growth(p_win: float, decimal_odds: float, stake_fraction: float) -> float:
    """
    Expected log-growth rate at a given stake fraction.

    g(f) = p * log(1 + f * b) + q * log(1 - f)

    Returns 0 when stake_fraction is 0 or when log args would be invalid.
    """
    if stake_fraction <= 0.0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - p_win
    # Guard against numerical degeneracies
    if stake_fraction >= 1.0 or 1.0 - stake_fraction <= 0.0:
        return 0.0
    return p_win * math.log(1.0 + stake_fraction * b) + q * math.log(1.0 - stake_fraction)


def stake_amount(
    bankroll: float,
    kelly_result: KellyResult,
    min_stake: float = 0.0,
) -> float:
    """
    Translate a KellyResult into a concrete monetary stake.

    Args:
        bankroll: Current bankroll in the same currency you want the
            stake denominated in.
        kelly_result: Output of kelly_fraction().
        min_stake: Below this threshold, return 0 (avoid micro-stakes
            that lose more to round-trip costs than they could earn).

    Returns:
        Recommended stake amount (>= 0).
    """
    if bankroll <= 0.0:
        return 0.0
    raw = bankroll * kelly_result.fractional_kelly
    if raw < min_stake:
        return 0.0
    return raw
