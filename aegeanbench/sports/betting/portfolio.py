"""
Portfolio-level bet sizing across multiple matches.

Single-match Kelly is well-defined; portfolio Kelly across many parallel
bets is not. This module implements a pragmatic policy:

  1. Rank candidate bets by edge (highest first).
  2. Greedily allocate stake fraction per bet, respecting:
       - per-bet cap (no single bet > max_per_bet_pct of bankroll)
       - per-match cap (no single match consumes > max_per_match_pct,
         in case two outcomes both look like value)
       - total exposure cap (sum of all stakes <= max_total_exposure_pct)
  3. Optionally apply diversification penalty when many bets correlate
     on the same outcome (e.g., all-bullish on home favourites).

This is not provably optimal — provably-optimal portfolio Kelly requires
a covariance matrix of bet outcomes which we don't have. The greedy
caps approach is the standard practitioner heuristic and trades a small
expected-growth loss for a large drawdown protection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from aegeanbench.sports.betting.value import ValueBetCandidate


@dataclass
class PortfolioConfig:
    """Risk parameters for portfolio construction."""
    bankroll: float = 1000.0
    max_per_bet_pct: float = 0.05         # No single bet > 5% bankroll
    max_per_match_pct: float = 0.08       # No single match > 8% bankroll
    max_total_exposure_pct: float = 0.50  # Total simultaneous stake <= 50%
    min_edge: float = 0.05                # Filter weak edges
    min_confidence: float = 0.0           # Optional: skip low-confidence bets
    kelly_fraction: float = 0.25          # Quarter Kelly default

    def __post_init__(self):
        if self.bankroll <= 0:
            raise ValueError("bankroll must be positive")
        for name, value in (
            ("max_per_bet_pct", self.max_per_bet_pct),
            ("max_per_match_pct", self.max_per_match_pct),
            ("max_total_exposure_pct", self.max_total_exposure_pct),
        ):
            if not 0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")


@dataclass
class SizedBet:
    """A value bet candidate with a concrete stake amount attached."""
    candidate: ValueBetCandidate
    stake_amount: float
    stake_fraction: float
    expected_return: float    # stake * edge in absolute money

    def to_dict(self) -> Dict:
        return {
            "candidate": self.candidate.to_dict(),
            "stake_amount": self.stake_amount,
            "stake_fraction": self.stake_fraction,
            "expected_return": self.expected_return,
        }


@dataclass
class BettingPortfolio:
    """Output of portfolio construction: which bets, how much, summary stats."""
    config: PortfolioConfig
    bets: List[SizedBet] = field(default_factory=list)
    skipped_candidates: List[ValueBetCandidate] = field(default_factory=list)
    total_stake: float = 0.0
    total_expected_return: float = 0.0
    exposure_pct: float = 0.0

    def summary(self) -> Dict:
        n_per_match: Dict[str, int] = {}
        for b in self.bets:
            n_per_match[b.candidate.match_id] = n_per_match.get(b.candidate.match_id, 0) + 1
        return {
            "bankroll": self.config.bankroll,
            "n_bets": len(self.bets),
            "n_matches": len(n_per_match),
            "total_stake": self.total_stake,
            "total_expected_return": self.total_expected_return,
            "exposure_pct": self.exposure_pct,
            "skipped": len(self.skipped_candidates),
        }


# ----------------------------- builder -----------------------------


def build_portfolio(
    candidates: Iterable[ValueBetCandidate],
    config: Optional[PortfolioConfig] = None,
) -> BettingPortfolio:
    """
    Greedy portfolio construction with risk caps.

    Args:
        candidates: Iterable of ValueBetCandidate (from find_value_bets across
            many matches). Order does not matter; we re-sort by edge.
        config: PortfolioConfig with risk parameters.

    Returns:
        BettingPortfolio with sized bets, skipped candidates, and summary.
    """
    cfg = config or PortfolioConfig()
    portfolio = BettingPortfolio(config=cfg)

    # Filter: edge floor + confidence floor + must have Kelly attached
    filtered: List[ValueBetCandidate] = []
    for c in candidates:
        if c.edge < cfg.min_edge:
            portfolio.skipped_candidates.append(c)
            continue
        if c.confidence is not None and c.confidence < cfg.min_confidence:
            portfolio.skipped_candidates.append(c)
            continue
        if c.kelly is None or not c.kelly.has_edge:
            portfolio.skipped_candidates.append(c)
            continue
        filtered.append(c)

    # Sort by edge descending so highest-edge bets get priority
    filtered.sort(key=lambda c: c.edge, reverse=True)

    used_per_match: Dict[str, float] = {}
    total_stake_fraction: float = 0.0

    for c in filtered:
        # Compute desired stake fraction from Kelly (already fractional)
        wanted = c.kelly.fractional_kelly

        # Apply per-bet cap
        wanted = min(wanted, cfg.max_per_bet_pct)

        # Apply per-match cap (could already have one bet on this match)
        already_on_match = used_per_match.get(c.match_id, 0.0)
        room_in_match = max(0.0, cfg.max_per_match_pct - already_on_match)
        wanted = min(wanted, room_in_match)

        # Apply total exposure cap
        room_total = max(0.0, cfg.max_total_exposure_pct - total_stake_fraction)
        wanted = min(wanted, room_total)

        if wanted <= 0:
            portfolio.skipped_candidates.append(c)
            continue

        stake_amount = cfg.bankroll * wanted
        expected = stake_amount * c.edge
        portfolio.bets.append(
            SizedBet(
                candidate=c,
                stake_amount=stake_amount,
                stake_fraction=wanted,
                expected_return=expected,
            )
        )
        used_per_match[c.match_id] = already_on_match + wanted
        total_stake_fraction += wanted

    portfolio.total_stake = sum(b.stake_amount for b in portfolio.bets)
    portfolio.total_expected_return = sum(b.expected_return for b in portfolio.bets)
    portfolio.exposure_pct = total_stake_fraction
    return portfolio
