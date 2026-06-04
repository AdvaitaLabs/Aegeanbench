"""
Domain models for sports prediction benchmark.

Designed for World Cup 2026 but generalizable to any league/tournament.
Kept as plain dataclasses to avoid pydantic overhead in tight benchmark loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MatchOutcome(str, Enum):
    """Three-way match outcome from the home team's perspective."""
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


class CompetitionStage(str, Enum):
    """World Cup stages — generalizable to other tournaments."""
    GROUP = "group"
    ROUND_OF_16 = "round_of_16"
    QUARTER_FINAL = "quarter_final"
    SEMI_FINAL = "semi_final"
    THIRD_PLACE = "third_place"
    FINAL = "final"
    FRIENDLY = "friendly"
    QUALIFIER = "qualifier"


@dataclass
class Team:
    """A national team (or club, in extended use)."""
    fifa_code: str               # 3-letter FIFA code: BRA, ARG, GER, ...
    name: str                    # Display name: "Brazil"
    name_zh: Optional[str] = None  # Chinese name: 巴西
    fifa_rank: Optional[int] = None
    elo_rating: Optional[float] = None
    coach: Optional[str] = None
    group: Optional[str] = None  # World Cup group letter

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Player:
    """A football player. Used by PlayerAgent for star/injury analysis."""
    player_id: str
    name: str
    team_fifa_code: str
    position: str                # GK / DF / MF / FW
    age: Optional[int] = None
    club: Optional[str] = None
    matches_played_for_team: int = 0
    goals_for_team: int = 0
    is_injured: bool = False
    is_suspended: bool = False
    # Recent club form: e.g., {"goals_last_10": 5, "minutes_last_10": 850}
    recent_form: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Odds:
    """
    Bookmaker odds for a single match.

    Decimal format (European), e.g., 2.10 means "win 2.10 per 1 staked".
    Implied probability = 1 / decimal_odds (before bookmaker margin removal).
    """
    bookmaker: str               # "Bet365", "Pinnacle", ...
    timestamp: datetime
    home_win: float
    draw: float
    away_win: float
    # Optional additional markets
    over_2_5: Optional[float] = None
    under_2_5: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None

    @property
    def implied_probs(self) -> Dict[str, float]:
        """Raw implied probabilities (sum > 1 due to bookmaker margin)."""
        return {
            "home_win": 1.0 / self.home_win,
            "draw": 1.0 / self.draw,
            "away_win": 1.0 / self.away_win,
        }

    @property
    def fair_probs(self) -> Dict[str, float]:
        """Margin-removed probabilities (sum to 1.0)."""
        raw = self.implied_probs
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    @property
    def margin(self) -> float:
        """Bookmaker overround (margin). e.g., 0.05 = 5% margin."""
        return sum(self.implied_probs.values()) - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bookmaker": self.bookmaker,
            "timestamp": self.timestamp.isoformat(),
            "home_win": self.home_win,
            "draw": self.draw,
            "away_win": self.away_win,
            "over_2_5": self.over_2_5,
            "under_2_5": self.under_2_5,
            "btts_yes": self.btts_yes,
            "btts_no": self.btts_no,
        }


@dataclass
class MatchResult:
    """
    Ground truth for a finished match. None until the match completes.

    Used by AegeanBench to score predictions after the fact.
    """
    home_goals: int
    away_goals: int
    ht_home_goals: Optional[int] = None  # Half-time
    ht_away_goals: Optional[int] = None
    outcome: Optional[MatchOutcome] = None  # Derived in __post_init__
    extra_time: bool = False
    penalty_shootout: bool = False
    pens_home: Optional[int] = None
    pens_away: Optional[int] = None

    def __post_init__(self):
        if self.outcome is None:
            if self.home_goals > self.away_goals:
                self.outcome = MatchOutcome.HOME_WIN
            elif self.home_goals < self.away_goals:
                self.outcome = MatchOutcome.AWAY_WIN
            else:
                self.outcome = MatchOutcome.DRAW

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.outcome is not None:
            d["outcome"] = self.outcome.value
        return d


@dataclass
class Match:
    """
    A football match. The central entity for sports benchmark.

    Lifecycle:
        1. Pre-match: result=None, odds populated, lineups optional
        2. Post-match: result populated → benchmark can score predictions
    """
    match_id: str
    competition: str             # "FIFA World Cup 2026"
    stage: CompetitionStage
    kickoff_at: datetime
    home_team: Team
    away_team: Team
    venue: Optional[str] = None
    # Optional pre-match context
    odds: List[Odds] = field(default_factory=list)
    home_lineup: List[Player] = field(default_factory=list)
    away_lineup: List[Player] = field(default_factory=list)
    weather: Optional[Dict[str, Any]] = None
    referee: Optional[str] = None
    h2h_last5: List[Dict[str, Any]] = field(default_factory=list)
    # Ground truth (None until match completes)
    result: Optional[MatchResult] = None

    @property
    def has_ground_truth(self) -> bool:
        return self.result is not None

    @property
    def best_odds(self) -> Optional[Odds]:
        """Return odds with the lowest bookmaker margin (most efficient)."""
        if not self.odds:
            return None
        return min(self.odds, key=lambda o: o.margin)

    @property
    def consensus_odds(self) -> Optional[Dict[str, float]]:
        """Average implied fair probabilities across all bookmakers."""
        if not self.odds:
            return None
        n = len(self.odds)
        agg = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        for o in self.odds:
            fp = o.fair_probs
            for k in agg:
                agg[k] += fp[k] / n
        return agg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "competition": self.competition,
            "stage": self.stage.value,
            "kickoff_at": self.kickoff_at.isoformat(),
            "home_team": self.home_team.to_dict(),
            "away_team": self.away_team.to_dict(),
            "venue": self.venue,
            "odds": [o.to_dict() for o in self.odds],
            "home_lineup": [p.to_dict() for p in self.home_lineup],
            "away_lineup": [p.to_dict() for p in self.away_lineup],
            "weather": self.weather,
            "referee": self.referee,
            "h2h_last5": self.h2h_last5,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class Prediction:
    """
    A single prediction for a match. Produced by any runner (LLM/Aegean/Elo/DC).

    Three probabilities must sum to ~1.0. Optional confidence reflects the
    model's self-assessment (separate from probabilities).
    """
    match_id: str
    runner_id: str               # "aegean", "gpt-5", "claude-opus-4-7", "elo", ...
    p_home_win: float
    p_draw: float
    p_away_win: float
    confidence: Optional[float] = None
    rationale: str = ""
    latency_ms: int = 0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        total = self.p_home_win + self.p_draw + self.p_away_win
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Probabilities must sum to ~1.0, got {total:.4f} "
                f"({self.p_home_win}/{self.p_draw}/{self.p_away_win})"
            )

    @property
    def predicted_outcome(self) -> MatchOutcome:
        """Outcome with highest probability (argmax)."""
        probs = {
            MatchOutcome.HOME_WIN: self.p_home_win,
            MatchOutcome.DRAW: self.p_draw,
            MatchOutcome.AWAY_WIN: self.p_away_win,
        }
        return max(probs, key=probs.get)

    @property
    def predicted_probability(self) -> float:
        """Probability of the predicted outcome."""
        return max(self.p_home_win, self.p_draw, self.p_away_win)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "runner_id": self.runner_id,
            "p_home_win": self.p_home_win,
            "p_draw": self.p_draw,
            "p_away_win": self.p_away_win,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }
