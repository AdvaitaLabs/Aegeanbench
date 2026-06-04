"""Sports data source adapters."""

from aegeanbench.sports.sources.base import FetchPolicy, SourceAdapter
from aegeanbench.sports.sources.clubelo import (
    ClubEloAdapter,
    FIFA_TO_CLUBELO,
    enrich_teams_with_elo,
)
from aegeanbench.sports.sources.fbref import FBrefAdapter
from aegeanbench.sports.sources.football_data import FootballDataAdapter
from aegeanbench.sports.sources.kaggle_loader import (
    KaggleHistoryLoader,
    download_kaggle_dataset,
)
from aegeanbench.sports.sources.soccersapi import SoccersAPIAdapter

__all__ = [
    "FetchPolicy",
    "SourceAdapter",
    "FootballDataAdapter",
    "SoccersAPIAdapter",
    "FBrefAdapter",
    "KaggleHistoryLoader",
    "download_kaggle_dataset",
    "ClubEloAdapter",
    "FIFA_TO_CLUBELO",
    "enrich_teams_with_elo",
]
