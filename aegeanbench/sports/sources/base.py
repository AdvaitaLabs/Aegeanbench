"""
Abstract base class for sports data source adapters.

All concrete adapters (FootballData, SoccersAPI, FBref, Kaggle) implement
this interface. The unified shape lets runners and predictors stay agnostic
about which provider served the data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from aegeanbench.sports.models import Match, Odds, Player, Team


@dataclass
class FetchPolicy:
    """
    How an adapter should behave for a single call.

    Sprint uses mock=True everywhere by default; real API gets enabled
    after keys land tomorrow.
    """
    use_cache: bool = True
    mock: bool = False           # if True, return canned data
    timeout_seconds: float = 10.0


class SourceAdapter(ABC):
    """
    Abstract source adapter.

    Each concrete adapter implements only the methods it has data for;
    others raise NotImplementedError. The benchmark's data layer composes
    multiple adapters to fill out a full Match.
    """

    # Subclasses override
    name: str = "base"
    has_fixtures: bool = False
    has_odds: bool = False
    has_lineups: bool = False
    has_xg: bool = False
    has_historical: bool = False

    def __init__(self, api_key: Optional[str] = None, mock_by_default: bool = True):
        self.api_key = api_key
        self.mock_by_default = mock_by_default

    # -------- methods (override only what the adapter supports) --------

    def fetch_teams(self, competition: str, policy: Optional[FetchPolicy] = None) -> List[Team]:
        """Return the list of teams in a competition (e.g., 32 World Cup squads)."""
        raise NotImplementedError(f"{self.name} does not support fetch_teams")

    def fetch_fixtures(
        self,
        competition: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        policy: Optional[FetchPolicy] = None,
    ) -> List[Match]:
        """Return matches in the date range for a competition."""
        raise NotImplementedError(f"{self.name} does not support fetch_fixtures")

    def fetch_match(self, match_id: str, policy: Optional[FetchPolicy] = None) -> Match:
        """Return a single match by ID (pre-match or post-match)."""
        raise NotImplementedError(f"{self.name} does not support fetch_match")

    def fetch_odds(self, match_id: str, policy: Optional[FetchPolicy] = None) -> List[Odds]:
        """Return current odds from all available bookmakers."""
        raise NotImplementedError(f"{self.name} does not support fetch_odds")

    def fetch_lineup(
        self, match_id: str, team_fifa_code: str, policy: Optional[FetchPolicy] = None
    ) -> List[Player]:
        """Return a team's starting lineup for a match."""
        raise NotImplementedError(f"{self.name} does not support fetch_lineup")

    def fetch_team_history(
        self,
        fifa_code: str,
        last_n: int = 10,
        policy: Optional[FetchPolicy] = None,
    ) -> List[Match]:
        """Return a team's last N matches with results."""
        raise NotImplementedError(f"{self.name} does not support fetch_team_history")

    def fetch_h2h(
        self,
        home_fifa: str,
        away_fifa: str,
        last_n: int = 5,
        policy: Optional[FetchPolicy] = None,
    ) -> List[Match]:
        """Return last N head-to-head matches between the two teams."""
        raise NotImplementedError(f"{self.name} does not support fetch_h2h")

    # ----------------- internal helpers -----------------

    def _resolve_policy(self, policy: Optional[FetchPolicy]) -> FetchPolicy:
        """Apply adapter defaults to a partial policy."""
        if policy is None:
            return FetchPolicy(mock=self.mock_by_default)
        # If caller didn't override mock and we default to mock, force mock
        if not policy.mock and self.mock_by_default:
            policy = FetchPolicy(
                use_cache=policy.use_cache,
                mock=True,
                timeout_seconds=policy.timeout_seconds,
            )
        return policy
