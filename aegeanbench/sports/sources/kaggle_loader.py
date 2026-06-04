"""
Loader for historical Kaggle datasets (offline files).

For Day 2's Dixon-Coles training we need ~5000 historical international
matches. Kaggle hosts the dataset 'International Football Results' which
covers every recorded international match from 1872 to present.

Until we download the real CSV, this loader returns a small synthetic
sample so downstream code can be tested.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from aegeanbench.sports.models import (
    CompetitionStage,
    Match,
    MatchResult,
    Team,
)

logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = Path.home() / ".aegeanbench" / "kaggle" / "international_results.csv"


def _synthetic_history(num_matches: int = 500) -> List[Match]:
    """
    Generate plausible historical matches for testing.

    Teams strength encoded in goal expectation; matches sampled from
    fixed pairings to ensure dataset has variety.
    """
    teams = [
        Team("BRA", "Brazil"), Team("ARG", "Argentina"), Team("FRA", "France"),
        Team("GER", "Germany"), Team("ESP", "Spain"), Team("ENG", "England"),
        Team("ITA", "Italy"), Team("POR", "Portugal"), Team("NED", "Netherlands"),
        Team("BEL", "Belgium"), Team("URU", "Uruguay"), Team("CRO", "Croatia"),
        Team("MEX", "Mexico"), Team("USA", "United States"), Team("JPN", "Japan"),
        Team("KOR", "South Korea"),
    ]

    strength = {
        "BRA": 1.9, "ARG": 1.95, "FRA": 2.0, "GER": 1.8, "ESP": 1.85, "ENG": 1.9,
        "ITA": 1.7, "POR": 1.85, "NED": 1.8, "BEL": 1.75, "URU": 1.6, "CRO": 1.7,
        "MEX": 1.4, "USA": 1.3, "JPN": 1.35, "KOR": 1.4,
    }

    matches: List[Match] = []
    base_date = datetime(2024, 1, 1)
    n_teams = len(teams)
    for i in range(num_matches):
        home_idx = i % n_teams
        away_idx = (i // n_teams + 1 + i) % n_teams
        if home_idx == away_idx:
            away_idx = (away_idx + 1) % n_teams
        home = teams[home_idx]
        away = teams[away_idx]
        home_xg = strength[home.fifa_code] * 1.1  # home advantage
        away_xg = strength[away.fifa_code]
        # Deterministic outcome to keep tests stable
        hg = int(round(home_xg))
        ag = int(round(away_xg))
        # Inject some draws and away wins
        if i % 7 == 0:
            ag = hg
        elif i % 11 == 0 and ag < hg:
            ag = hg + 1
        matches.append(
            Match(
                match_id=f"HIST-{i:04d}",
                competition="International Friendly",
                stage=CompetitionStage.FRIENDLY,
                kickoff_at=base_date - timedelta(days=i * 2),
                home_team=home,
                away_team=away,
                result=MatchResult(home_goals=hg, away_goals=ag),
            )
        )
    return matches


class KaggleHistoryLoader:
    """Loads historical match results, either from local CSV or mock."""

    def __init__(self, csv_path: Optional[Path] = None, mock_by_default: bool = True):
        self.csv_path = Path(csv_path) if csv_path else DEFAULT_DATA_PATH
        self.mock_by_default = mock_by_default

    def load(self, num_matches: int = 500, mock: Optional[bool] = None) -> List[Match]:
        """
        Load historical matches.

        Args:
            num_matches: max number of matches to return
            mock: override default (None → use mock_by_default)

        Returns:
            List of Match with result populated
        """
        use_mock = mock if mock is not None else (self.mock_by_default or not self.csv_path.exists())
        if use_mock:
            logger.info("KaggleLoader: using synthetic history (%d matches)", num_matches)
            return _synthetic_history(num_matches)
        return self._load_csv(num_matches)

    def _load_csv(self, num_matches: int) -> List[Match]:
        logger.warning("kaggle CSV loader not yet implemented; using synthetic")
        return _synthetic_history(num_matches)
