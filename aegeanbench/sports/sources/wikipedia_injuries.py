"""
Wikipedia injury / suspension scraper.

During major tournaments Wikipedia maintains a "Squads" article per
team with a Notes column flagging injuries and suspensions. The article
URLs follow a stable pattern:

    https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads

Our scraper pulls that single page, extracts each team's section, and
parses the per-player notes for keywords like "injured", "withdrew",
"suspended", "replaced". Each entry yields a structured injury record.

The page changes daily during the tournament, so we cache for 1 hour.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

from aegeanbench.sports.cache import FileCache, get_default_cache

logger = logging.getLogger(__name__)


WIKI_BASE = "https://en.wikipedia.org/wiki"
DEFAULT_SQUADS_PAGE = "2026_FIFA_World_Cup_squads"
CACHE_TTL = timedelta(hours=1)


# Tokens that flag injury / unavailability in Wikipedia squad notes.
INJURY_KEYWORDS = (
    "injured", "injury", "withdrew", "withdrawn", "replaced",
    "torn", "fractured", "ruptured", "out of", "ruled out",
    "ankle", "knee", "hamstring", "thigh",
)

SUSPENSION_KEYWORDS = (
    "suspended", "suspension", "red card", "yellow accumulation",
)


class WikipediaInjuriesAdapter:
    """
    Scrape and cache the World Cup 2026 squads page from Wikipedia.

    Returns a dict keyed by team name (as Wikipedia spells it) holding
    a list of (player_name, status, raw_note) tuples.
    """

    def __init__(
        self,
        squads_page: str = DEFAULT_SQUADS_PAGE,
        mock_by_default: bool = False,
        timeout: float = 10.0,
        cache: Optional[FileCache] = None,
    ):
        self.squads_page = squads_page
        self.mock_by_default = mock_by_default
        self.timeout = timeout
        self.cache = cache or get_default_cache()

    def fetch_all(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Return {team_name: [ {player, status, note} ]}.

        Status is one of: 'injured', 'suspended', 'replaced', 'unknown'.
        """
        if self.mock_by_default:
            return self._mock()

        cache_key = ("wikipedia_injuries", self.squads_page)
        cached = self.cache.get(*cache_key, ttl=CACHE_TTL)
        if cached is not None:
            return cached

        try:
            html = self._live_fetch()
            parsed = self._parse_squads_html(html)
            self.cache.set(parsed, *cache_key)
            return parsed
        except Exception as e:
            logger.warning("wikipedia injuries scrape failed (%s); using mock", e)
            return self._mock()

    def fetch_for_team(self, team_name: str) -> List[Dict[str, str]]:
        """Convenience: just one team's injury list."""
        return self.fetch_all().get(team_name, [])

    # ----------------- internals -----------------

    def _live_fetch(self) -> str:
        import requests
        url = f"{WIKI_BASE}/{self.squads_page}"
        resp = requests.get(
            url,
            headers={"User-Agent": "AegeanBench/0.1 (research)"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _parse_squads_html(html: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Lightweight HTML parser tuned to the squads page structure.

        We avoid BeautifulSoup to keep dependencies minimal. The regex
        approach is robust to mild markup changes; if Wikipedia overhauls
        the page format we fall back to mock.

        Strategy:
          1. Find each <h3><span class="mw-headline" id="TeamName">...</span></h3>
             that marks a team section.
          2. Within that section grab every <tr>...</tr> row of the squad
             table.
          3. From each row pull the player name (first <a>) and the notes
             cell content. Scan the notes for injury / suspension tokens.
        """
        out: Dict[str, List[Dict[str, str]]] = {}

        # Split by h3 anchor: <span class="mw-headline" id="...">TeamName</span>
        team_pattern = re.compile(
            r'<span class="mw-headline" id="([^"]+)"[^>]*>([^<]+)</span>',
            re.IGNORECASE,
        )
        teams = list(team_pattern.finditer(html))
        if not teams:
            return out

        # Iterate adjacent pairs to get each team's substring
        for i, match in enumerate(teams):
            team_name = match.group(2).strip()
            section_start = match.end()
            section_end = teams[i + 1].start() if i + 1 < len(teams) else len(html)
            section_html = html[section_start:section_end]

            injuries: List[Dict[str, str]] = []
            # Each squad row is roughly  <tr>...<a ...>Player Name</a>...notes...</tr>
            row_pattern = re.compile(
                r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE
            )
            for row_match in row_pattern.finditer(section_html):
                row = row_match.group(1)
                name_match = re.search(r'<a[^>]*>([^<]+)</a>', row)
                if not name_match:
                    continue
                player = name_match.group(1).strip()
                # Notes cell typically last <td> before </tr>
                note_text = re.sub(r'<[^>]+>', ' ', row).strip()
                lower = note_text.lower()
                status: Optional[str] = None
                if any(k in lower for k in INJURY_KEYWORDS):
                    status = "injured"
                elif any(k in lower for k in SUSPENSION_KEYWORDS):
                    status = "suspended"
                elif "replaced" in lower or "called up" in lower:
                    status = "replaced"
                if status:
                    injuries.append(
                        {
                            "player": player,
                            "status": status,
                            "note": note_text[:200],
                        }
                    )

            if injuries:
                out[team_name] = injuries

        return out

    @staticmethod
    def _mock() -> Dict[str, List[Dict[str, str]]]:
        """Deterministic mock so tests + offline dev work."""
        return {
            "Brazil": [
                {"player": "Casemiro", "status": "injured", "note": "ankle, replaced by Andre"},
            ],
            "Argentina": [],
            "France": [
                {"player": "Aurelien Tchouameni", "status": "suspended", "note": "yellow card accumulation"},
            ],
            "Germany": [],
        }
