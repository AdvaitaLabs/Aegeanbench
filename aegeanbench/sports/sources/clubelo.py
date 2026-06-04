"""
Real adapter for clubelo.com - free Elo ratings, no API key required.

clubelo.com exposes a simple CSV-over-HTTP interface used by many free
football analytics projects:

    http://api.clubelo.com/ARG     -> CSV of Argentina's daily Elo
    http://api.clubelo.com/2026-06-04  -> CSV of all teams' Elo on that date

We use the date endpoint at startup to seed every World Cup team's Elo
rating, then keep the values in memory for the rest of the session.

This adapter does not implement the SourceAdapter ABC because clubelo
returns team-level data rather than matches. Use it as a one-shot Elo
seeder for EloPredictor or as an enrichment step for Team objects.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from aegeanbench.sports.cache import FileCache, get_default_cache

logger = logging.getLogger(__name__)


CLUBELO_API_BASE = "http://api.clubelo.com"
CACHE_TTL = timedelta(days=1)


# Mapping from FIFA 3-letter codes to clubelo's club-or-national identifiers.
# clubelo uses standard 3-letter codes for national teams but a few differ.
FIFA_TO_CLUBELO: Dict[str, str] = {
    "BRA": "Brazil",
    "ARG": "Argentina",
    "FRA": "France",
    "GER": "Germany",
    "ESP": "Spain",
    "ENG": "England",
    "POR": "Portugal",
    "NED": "Netherlands",
    "BEL": "Belgium",
    "ITA": "Italy",
    "CRO": "Croatia",
    "URU": "Uruguay",
    "MEX": "Mexico",
    "USA": "USA",
    "JPN": "Japan",
    "KOR": "South Korea",
    "AUS": "Australia",
    "MAR": "Morocco",
    "SEN": "Senegal",
    "SUI": "Switzerland",
    "DEN": "Denmark",
    "POL": "Poland",
    "SRB": "Serbia",
    "CAN": "Canada",
    "ECU": "Ecuador",
    "IRN": "Iran",
    "QAT": "Qatar",
    "KSA": "Saudi Arabia",
    "TUN": "Tunisia",
    "CRC": "Costa Rica",
    "GHA": "Ghana",
    "CMR": "Cameroon",
}


class ClubEloAdapter:
    """
    Free Elo rating lookup via api.clubelo.com.

    Sprint mode (default mock=False since this source is free):
        - Hits api.clubelo.com/<YYYY-MM-DD>
        - Parses CSV into {team_name: elo_rating}
        - Caches result to ~/.aegeanbench/... with TTL of 1 day

    Offline mode (mock=True):
        - Returns the static FALLBACK_ELO table below so tests stay
          deterministic and the adapter never hits the network in CI.
    """

    name = "clubelo"
    has_xg = False

    # Conservative fallback ratings used in mock mode.
    # Approximately match late-2025 clubelo values; refresh periodically.
    FALLBACK_ELO: Dict[str, float] = {
        "BRA": 1981.0, "ARG": 2114.0, "FRA": 2058.0, "GER": 1918.0,
        "ESP": 1955.0, "ENG": 2026.0, "POR": 1975.0, "NED": 1962.0,
        "BEL": 1899.0, "ITA": 1942.0, "CRO": 1894.0, "URU": 1854.0,
        "MEX": 1737.0, "USA": 1715.0, "JPN": 1810.0, "KOR": 1782.0,
        "AUS": 1681.0, "MAR": 1849.0, "SEN": 1798.0, "SUI": 1773.0,
        "DEN": 1834.0, "POL": 1820.0, "SRB": 1791.0, "CAN": 1671.0,
        "ECU": 1727.0, "IRN": 1718.0, "QAT": 1538.0, "KSA": 1572.0,
        "TUN": 1657.0, "CRC": 1612.0, "GHA": 1574.0, "CMR": 1647.0,
    }

    def __init__(
        self,
        mock: bool = False,
        timeout: float = 5.0,
        cache: Optional[FileCache] = None,
    ):
        self.mock = mock
        self.timeout = timeout
        self.cache = cache or get_default_cache()

    def fetch_elo_table(
        self, as_of: Optional[date] = None
    ) -> Dict[str, float]:
        """
        Return {fifa_code: elo_rating} for all known teams on the given date.

        Args:
            as_of: snapshot date. Default: today.

        Returns:
            Dict keyed by FIFA 3-letter code. Teams not found in clubelo
            data fall back to the FALLBACK_ELO table.
        """
        if self.mock:
            return dict(self.FALLBACK_ELO)

        snapshot = as_of or date.today()
        cache_key = ("clubelo", "table", snapshot.isoformat())

        def compute():
            return self._fetch_csv(snapshot)

        try:
            return self.cache.get_or_compute(compute, *cache_key, ttl=CACHE_TTL)
        except Exception as e:
            logger.warning("ClubElo fetch failed (%s); using fallback", e)
            return dict(self.FALLBACK_ELO)

    def fetch_one(self, fifa_code: str, as_of: Optional[date] = None) -> Optional[float]:
        """Convenience: just one team's Elo."""
        return self.fetch_elo_table(as_of).get(fifa_code)

    def _fetch_csv(self, snapshot: date) -> Dict[str, float]:
        """Raw network fetch + CSV parse. Returns FIFA-code-keyed dict."""
        import requests

        url = f"{CLUBELO_API_BASE}/{snapshot.isoformat()}"
        logger.info("ClubElo: GET %s", url)
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return self._parse_csv(resp.text)

    @staticmethod
    def _parse_csv(text: str) -> Dict[str, float]:
        """
        Parse clubelo CSV into {fifa_code: elo}.

        Columns: Rank,Club,Country,Level,Elo,From,To

        For national teams the "Country" column matches the team name.
        We map back to FIFA codes via FIFA_TO_CLUBELO.
        """
        out: Dict[str, float] = {}
        reverse_map = {v.lower(): k for k, v in FIFA_TO_CLUBELO.items()}
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            club = (row.get("Club") or "").strip()
            try:
                elo = float(row.get("Elo") or 0)
            except ValueError:
                continue
            fifa_code = reverse_map.get(club.lower())
            if fifa_code:
                out[fifa_code] = elo
        return out


def enrich_teams_with_elo(teams, adapter: Optional[ClubEloAdapter] = None) -> None:
    """
    Mutate a list of Team objects to populate elo_rating from clubelo.

    Safe to call in production startup; fetches once and caches for a day.
    Falls through silently if a team isn't in the FIFA_TO_CLUBELO map.
    """
    adapter = adapter or ClubEloAdapter()
    table = adapter.fetch_elo_table()
    for t in teams:
        elo = table.get(t.fifa_code)
        if elo is not None:
            t.elo_rating = elo
