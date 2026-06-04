"""
Adapter for FBref (fbref.com).

Free source of advanced football statistics (xG, xA, possession metrics).
Real implementation uses BeautifulSoup to scrape; we ship mock only for
the sprint and switch to live scraping in Day 2 if needed.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from aegeanbench.sports.sources.base import FetchPolicy, SourceAdapter

logger = logging.getLogger(__name__)


# Mock xG profiles for the 8 sample teams (last 10 international matches).
# xg_for: average expected goals scored per match
# xg_against: average expected goals conceded per match
_MOCK_XG_TABLE: Dict[str, Dict[str, float]] = {
    "ARG": {"xg_for": 2.10, "xg_against": 0.85, "possession": 0.58, "ppda": 9.2},
    "BRA": {"xg_for": 2.05, "xg_against": 0.92, "possession": 0.61, "ppda": 8.5},
    "FRA": {"xg_for": 2.00, "xg_against": 0.95, "possession": 0.55, "ppda": 10.1},
    "GER": {"xg_for": 1.85, "xg_against": 1.10, "possession": 0.59, "ppda": 9.8},
    "ESP": {"xg_for": 1.90, "xg_against": 0.90, "possession": 0.65, "ppda": 7.9},
    "ENG": {"xg_for": 1.95, "xg_against": 0.88, "possession": 0.57, "ppda": 9.5},
    "POR": {"xg_for": 1.92, "xg_against": 1.05, "possession": 0.56, "ppda": 10.3},
    "NED": {"xg_for": 1.88, "xg_against": 1.00, "possession": 0.58, "ppda": 9.9},
}


class FBrefAdapter(SourceAdapter):
    """
    FBref does not implement the SourceAdapter abstract methods directly
    because it provides team-level metrics, not matches. Use fetch_xg_profile().
    """

    name = "fbref"
    has_xg = True

    def __init__(self, mock_by_default: bool = True):
        super().__init__(api_key=None, mock_by_default=mock_by_default)

    def fetch_xg_profile(
        self,
        fifa_code: str,
        policy: Optional[FetchPolicy] = None,
    ) -> Dict[str, float]:
        """
        Return xG-based profile for a team.

        Keys:
            xg_for: avg expected goals scored
            xg_against: avg expected goals conceded
            possession: avg possession share (0-1)
            ppda: passes per defensive action (lower = more press)
        """
        policy = self._resolve_policy(policy)
        if policy.mock:
            return _MOCK_XG_TABLE.get(fifa_code, {
                "xg_for": 1.20,
                "xg_against": 1.30,
                "possession": 0.50,
                "ppda": 11.0,
            })
        return self._real_fetch_xg_profile(fifa_code, policy)

    def _real_fetch_xg_profile(
        self, fifa_code: str, policy: FetchPolicy
    ) -> Dict[str, float]:
        logger.warning("fbref real scraper not yet implemented; mock fallback")
        return _MOCK_XG_TABLE.get(fifa_code, {})
