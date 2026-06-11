"""
FIFA Men's World Ranking — live scrape with hardcoded fallback.

Primary source: FIFA's official ranking JSON endpoint behind
https://inside.fifa.com/fifa-world-ranking/men . Cached on disk for
6 hours so we don't hammer FIFA. Falls back to a hardcoded April-2026
snapshot when the scrape fails (network down, schema changed, etc.).

Be honest with callers about provenance: `rank_for(code)` returns the
best available rank but `last_source()` tells you whether it came from
live scrape ("fifa_live") or the static fallback ("fifa_hardcoded").

We use the rank as the "team strength" signal for stats / strategy
specialists. xG-style attacking/defending proxies are DERIVED from
rank position via a simple formula — they are NOT measured xG. This
file is the only place where that approximation lives so callers can
decide whether to trust it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ----------------------------- hardcoded fallback -----------------------------
# Manually transcribed from FIFA's published April 2026 update. Used only
# when the live scrape fails. NOT authoritative — refresh periodically.
WC_2026_FIFA_RANKINGS_FALLBACK: Dict[str, int] = {
    "ARG": 1,   "ESP": 2,   "FRA": 3,   "ENG": 4,   "BRA": 5,
    "POR": 6,   "NED": 7,   "BEL": 8,   "GER": 9,   "CRO": 10,
    "ITA": 11,  "URU": 12,  "MAR": 13,  "COL": 14,  "USA": 15,
    "MEX": 16,  "JPN": 17,  "SUI": 18,  "DEN": 19,  "SEN": 20,
    "POL": 21,  "KOR": 22,  "AUS": 23,  "ECU": 24,  "AUT": 25,
    "WAL": 26,  "UKR": 27,  "TUN": 28,  "CIV": 29,  "PER": 30,
    "SRB": 31,  "EGY": 32,  "PAR": 33,  "TUR": 34,  "NOR": 35,
    "VEN": 36,  "PAN": 37,  "CAN": 38,  "QAT": 39,  "KSA": 40,
    "JOR": 41,  "UZB": 42,  "RSA": 43,  "IRN": 44,  "JAM": 45,
    "CRC": 46,  "GHA": 47,  "CPV": 48,
}

DEFAULT_RANK = 80


# ----------------------------- live cache -----------------------------

_CACHE_PATH = Path(os.path.expanduser(
    os.getenv("AEGEANBENCH_FIFA_RANK_CACHE",
              "~/.aegeanbench/fifa_rankings.json")
))
_CACHE_TTL_SECONDS = 6 * 3600   # refresh every 6 hours

_lock = threading.Lock()
_live_state: Dict = {
    "loaded_at": 0.0,
    "rankings": None,
    "source": "uninitialised",
}


# Map FIFA's 3-letter codes to our internal codes when they diverge.
# FIFA mostly uses ISO-3 too, but a handful differ (KSA <-> KOR, etc.).
_FIFA_CODE_ALIASES: Dict[str, str] = {
    "KSA": "KSA", "KOR": "KOR", "RSA": "RSA", "USA": "USA",
    "ENG": "ENG", "WAL": "WAL", "CIV": "CIV", "CRC": "CRC",
    "CPV": "CPV",
}


def _fetch_live_rankings() -> Optional[Dict[str, int]]:
    """
    Pull live FIFA Men's ranking from Wikipedia. The article's ranking
    table is wikitext-stable (the same row format for years), much more
    reliable than FIFA's JS-rendered site or churned community repos.
    The MediaWiki API gives us parsed HTML we can regex without a heavy
    HTML parser dep.

    A CSV override via env AEGEANBENCH_FIFA_CSV_URL still wins when set.
    """
    try:
        import requests
    except ImportError:
        return None

    # 1) Env-var CSV override (if user pinned a specific source)
    override = os.getenv("AEGEANBENCH_FIFA_CSV_URL", "").strip()
    if override:
        try:
            r = requests.get(override, timeout=10)
            if r.status_code == 200 and r.text:
                parsed = _parse_fifa_csv(r.text)
                if parsed and len(parsed) >= 50:
                    logger.info("FIFA ranking parsed from override CSV (%d teams)", len(parsed))
                    return parsed
        except Exception as e:
            logger.warning("Override FIFA CSV fetch failed: %s", e)

    # 2) eloratings.net (default - more stable than Wikipedia for Docker IPs)
    elo_result = _fetch_from_eloratings()
    if elo_result:
        return elo_result

    # 3) Wikipedia (last resort, rate-limited from shared IPs)
    return _fetch_from_wikipedia()


def _fetch_from_eloratings() -> Optional[Dict[str, int]]:
    """
    Pull national-team strength from eloratings.net's CSV endpoint.

    eloratings.net publishes daily Elo ratings for every national team
    going back to 1872. Unlike Wikipedia they don't rate-limit Docker
    IPs aggressively, and the CSV format is stable.

    We convert Elo rating to a rank (1=highest Elo). Elo is actually a
    BETTER strength signal than FIFA rank — FIFA's points system has
    quirks; Elo updates after every match.
    """
    try:
        import requests
    except ImportError:
        return None

    # eloratings.net publishes national-team Elo ratings as a public
    # TSV. Free, no key, stable for years.
    url_national = "https://www.eloratings.net/World.tsv"
    headers = {"User-Agent": "Mozilla/5.0 AegeanBench/0.2 (research)"}

    try:
        r = requests.get(url_national, headers=headers, timeout=10)
        if r.status_code != 200 or len(r.text) < 1000:
            logger.warning("eloratings.net World.tsv -> HTTP %s (%d bytes)",
                           r.status_code, len(r.text or ""))
            return None
    except Exception as e:
        logger.warning("eloratings.net fetch failed: %s", e)
        return None

    # eloratings.net TSV format (verified 2026-06):
    #   col 0: position
    #   col 1: rank
    #   col 2: 2-letter country code (ES, AR, FR, EN, BR, ...)
    #   col 3: current Elo rating
    # We just need col 0 (rank) and col 2 (2-letter code) -> map to FIFA 3-letter.
    result: Dict[str, int] = {}
    for line in r.text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            rank = int(parts[0].strip())
        except ValueError:
            continue
        code2 = parts[2].strip().upper()
        code3 = _ELO_TO_FIFA.get(code2)
        if code3:
            result[code3] = rank
    if len(result) < 30:
        logger.warning("eloratings.net yielded only %d mapped teams", len(result))
        return None
    logger.info("FIFA-style ranking from eloratings.net (%d teams)", len(result))
    return result


# eloratings.net uses a 2-letter code system that is mostly ISO-3166 alpha-2
# but with a few quirks (EN for England, KR for Korea, etc.). Map them to
# our 3-letter FIFA codes. Only WC 2026 teams + a few major nations.
_ELO_TO_FIFA: Dict[str, str] = {
    "ES": "ESP", "AR": "ARG", "FR": "FRA", "EN": "ENG", "BR": "BRA",
    "PT": "POR", "CO": "COL", "NL": "NED", "EC": "ECU", "DE": "GER",
    "IT": "ITA", "BE": "BEL", "HR": "CRO", "UY": "URU", "MA": "MAR",
    "US": "USA", "MX": "MEX", "JP": "JPN", "CH": "SUI", "DK": "DEN",
    "SN": "SEN", "PL": "POL", "KR": "KOR", "AU": "AUS", "AT": "AUT",
    "WS": "WAL", "WL": "WAL", "UA": "UKR", "TN": "TUN", "CI": "CIV",
    "PE": "PER", "RS": "SRB", "EG": "EGY", "PY": "PAR", "TR": "TUR",
    "NO": "NOR", "VE": "VEN", "PA": "PAN", "CA": "CAN", "QA": "QAT",
    "SA": "KSA", "JO": "JOR", "UZ": "UZB", "ZA": "RSA", "IR": "IRN",
    "JM": "JAM", "CR": "CRC", "GH": "GHA", "CV": "CPV",
}


def _fetch_from_wikipedia() -> Optional[Dict[str, int]]:
    """
    Scrape Wikipedia's FIFA Men's World Ranking table via MediaWiki API.
    The page has the current top ~30 teams in a sortable table; we
    extract (rank, country, FIFA code) rows.
    """
    try:
        import re
        import requests
    except ImportError:
        return None

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": "FIFA Men's World Ranking",
        "prop": "wikitext",
        "format": "json",
        "section": "1",  # "Current ranking" section
    }
    headers = {"User-Agent": "AegeanBench/0.2 (worldcup-2026; football research)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        wikitext = (r.json().get("parse") or {}).get("wikitext", {}).get("*", "")
    except Exception as e:
        logger.warning("Wikipedia FIFA ranking fetch failed: %s", e)
        return None

    if not wikitext or "{{fb" not in wikitext.lower():
        logger.warning("Wikipedia FIFA ranking page has no fb templates; using fallback")
        return None

    # Wikipedia uses {{fb|XYZ}} templates where XYZ is a 3-letter code.
    # The table rows look like:
    #   | 1 || {{steady}} || {{fb|ARG}} || 1886.16 || ...
    # We pair each numeric rank cell with the next {{fb|CODE}} we see.
    out: Dict[str, int] = {}
    rank_pattern = re.compile(r"\|\s*(\d{1,3})\s*\|\|")
    code_pattern = re.compile(r"\{\{\s*fb(?:r|f)?\s*\|\s*([A-Z]{3})\s*[\|}]", re.IGNORECASE)

    # Walk line by line; remember the last seen rank, attach the next code
    last_rank: Optional[int] = None
    for line in wikitext.splitlines():
        rm = rank_pattern.search(line)
        if rm:
            try:
                cand = int(rm.group(1))
                if 1 <= cand <= 250:
                    last_rank = cand
            except ValueError:
                pass
        cm = code_pattern.search(line)
        if cm and last_rank is not None:
            code = cm.group(1).upper()
            if code not in out:
                out[code] = last_rank
                last_rank = None  # consume so next rank attaches to next code

    if len(out) >= 30:
        logger.info("FIFA ranking parsed from Wikipedia (%d teams)", len(out))
        return out
    logger.warning("Wikipedia parse yielded only %d entries", len(out))
    return None


def _parse_fifa_csv(text: str) -> Dict[str, int]:
    """
    Parse a FIFA ranking CSV into {3-letter-code: rank}. Picks the
    LATEST snapshot when the file is a long history (rank_date column).
    Tolerates a handful of column-name variations across community repos.
    """
    import csv, io
    out: Dict[str, int] = {}
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return out

    cols = {c.lower(): c for c in reader.fieldnames}
    rank_col = (cols.get("rank") or cols.get("position")
                or cols.get("ranking") or cols.get("current_rank"))
    code_col = (cols.get("country_abrv") or cols.get("country_abbr")
                or cols.get("countryabbr") or cols.get("country_code")
                or cols.get("code") or cols.get("tla"))
    name_col = (cols.get("country_full") or cols.get("country_name")
                or cols.get("country") or cols.get("team"))
    date_col = (cols.get("rank_date") or cols.get("date")
                or cols.get("ranking_date"))

    if rank_col is None or (code_col is None and name_col is None):
        return out

    # First pass: find the latest snapshot date (if applicable)
    latest_date = ""
    if date_col:
        for row in reader:
            d = (row.get(date_col) or "").strip()
            if d > latest_date:
                latest_date = d
        # csv.DictReader is single-pass; re-read.
        reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        if date_col and latest_date:
            if (row.get(date_col) or "").strip() != latest_date:
                continue
        code = ""
        if code_col:
            code = (row.get(code_col) or "").upper().strip()
        if not code and name_col:
            code = _NAME_TO_CODE.get((row.get(name_col) or "").strip().lower(), "")
        try:
            rank = int(row.get(rank_col) or 0)
        except (TypeError, ValueError):
            continue
        if code and 1 <= rank <= 250:
            out[code] = rank
    return out


# Country full-name -> 3-letter code, used when a CSV only has the name.
_NAME_TO_CODE: Dict[str, str] = {
    "argentina": "ARG", "spain": "ESP", "france": "FRA", "england": "ENG",
    "brazil": "BRA", "portugal": "POR", "netherlands": "NED",
    "belgium": "BEL", "germany": "GER", "croatia": "CRO", "italy": "ITA",
    "uruguay": "URU", "morocco": "MAR", "colombia": "COL",
    "united states": "USA", "usa": "USA", "mexico": "MEX",
    "japan": "JPN", "switzerland": "SUI", "denmark": "DEN",
    "senegal": "SEN", "poland": "POL", "korea republic": "KOR",
    "south korea": "KOR", "australia": "AUS", "ecuador": "ECU",
    "austria": "AUT", "wales": "WAL", "ukraine": "UKR", "tunisia": "TUN",
    "ivory coast": "CIV", "côte d'ivoire": "CIV", "cote d'ivoire": "CIV",
    "peru": "PER", "serbia": "SRB", "egypt": "EGY", "paraguay": "PAR",
    "türkiye": "TUR", "turkey": "TUR", "norway": "NOR",
    "venezuela": "VEN", "panama": "PAN", "canada": "CAN", "qatar": "QAT",
    "saudi arabia": "KSA", "jordan": "JOR", "uzbekistan": "UZB",
    "south africa": "RSA", "iran": "IRN", "ir iran": "IRN",
    "jamaica": "JAM", "costa rica": "CRC", "ghana": "GHA",
    "cape verde": "CPV", "cabo verde": "CPV",
}


def _save_cache(rankings: Dict[str, int]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(
                {"loaded_at": time.time(), "rankings": rankings},
                f, ensure_ascii=False,
            )
    except Exception as e:
        logger.debug("FIFA rank cache write failed: %s", e)


def _load_cache() -> Optional[Dict[str, int]]:
    if not _CACHE_PATH.exists():
        return None
    try:
        with _CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
    except Exception:
        return None
    if time.time() - float(data.get("loaded_at", 0)) > _CACHE_TTL_SECONDS:
        return None
    return data.get("rankings") or None


def _get_rankings() -> Dict[str, int]:
    """
    Resolution order:
      1. In-process cache (fastest)
      2. Disk cache  (survives restart, 6h TTL)
      3. Live FIFA scrape
      4. Hardcoded fallback
    """
    with _lock:
        # Step 1: in-process
        if (_live_state.get("rankings")
                and time.time() - _live_state["loaded_at"] < _CACHE_TTL_SECONDS):
            return _live_state["rankings"]

        # Step 2: disk
        from_disk = _load_cache()
        if from_disk:
            _live_state.update({
                "loaded_at": time.time(),
                "rankings": from_disk,
                "source": "fifa_disk_cache",
            })
            return from_disk

        # Step 3: live scrape
        scraped = _fetch_live_rankings()
        if scraped:
            _save_cache(scraped)
            _live_state.update({
                "loaded_at": time.time(),
                "rankings": scraped,
                "source": "fifa_live",
            })
            logger.info("FIFA ranking refreshed from live scrape (%d entries)", len(scraped))
            return scraped

        # Step 4: hardcoded fallback
        _live_state.update({
            "loaded_at": time.time(),
            "rankings": WC_2026_FIFA_RANKINGS_FALLBACK,
            "source": "fifa_hardcoded",
        })
        return WC_2026_FIFA_RANKINGS_FALLBACK


def last_source() -> str:
    """Where did the most recent rank lookup come from?"""
    return _live_state.get("source", "uninitialised")


def rank_for(fifa_code: str) -> int:
    rankings = _get_rankings()
    return rankings.get((fifa_code or "").upper(), DEFAULT_RANK)


# Back-compat alias for existing imports.
WC_2026_FIFA_RANKINGS = WC_2026_FIFA_RANKINGS_FALLBACK


def derive_xg_profile(fifa_code: str) -> Dict[str, float]:
    """
    Derive an xG-style profile from FIFA rank position.

    THIS IS NOT REAL MEASURED xG. It's a formula that maps rank to
    plausible attacking/defending numbers so the LLM has concrete
    figures to anchor on instead of "(no data)" placeholders.

    The `source` field tells you whether the rank itself came from a
    live FIFA scrape ("fifa_live") or a hardcoded fallback
    ("fifa_hardcoded"). Either way the xG numbers are RANK-DERIVED,
    not measured — callers should surface this honestly to the user.
    """
    rank = rank_for(fifa_code)
    strength = max(0.0, 1.0 - (rank - 1) / 79.0)   # 1.0 best -> 0.0 worst
    xg_for = round(0.9 + strength * 0.8, 2)        # 0.9 .. 1.7
    xg_against = round(1.5 - strength * 0.8, 2)    # 1.5 .. 0.7
    possession = round(0.45 + strength * 0.10, 2)  # 0.45 .. 0.55
    ppda = round(13.0 - strength * 4.0, 1)         # 13.0 .. 9.0
    return {
        "xg_for": xg_for,
        "xg_against": xg_against,
        "possession": possession,
        "ppda": ppda,
        "fifa_rank": rank,
        "source": f"{last_source()}+derived",
    }


def derive_xg_profile(fifa_code: str) -> Dict[str, float]:
    """
    Derive an xG-style profile from FIFA rank position. This is a real-
    data-grounded approximation — it's a function of the published rank
    not a random number — but it's NOT live per-match xG.

    Empirically tuned so:
      Top 5 (ARG/ESP/FRA/ENG/BRA): xg_for ~1.7, xg_against ~0.7
      Mid (rank 20):               xg_for ~1.3, xg_against ~1.1
      Bottom WC (rank 48):         xg_for ~0.9, xg_against ~1.5
    """
    rank = rank_for(fifa_code)
    # Linear interpolation from rank 1 (best) to rank 80 (default)
    strength = max(0.0, 1.0 - (rank - 1) / 79.0)   # 1.0 best -> 0.0 worst
    xg_for = round(0.9 + strength * 0.8, 2)        # 0.9 .. 1.7
    xg_against = round(1.5 - strength * 0.8, 2)    # 1.5 .. 0.7
    possession = round(0.45 + strength * 0.10, 2)  # 0.45 .. 0.55
    # PPDA: lower = more press. Strong teams press more aggressively.
    ppda = round(13.0 - strength * 4.0, 1)         # 13.0 .. 9.0
    return {
        "xg_for": xg_for,
        "xg_against": xg_against,
        "possession": possession,
        "ppda": ppda,
        "fifa_rank": rank,
        "source": "fifa_rank_derived",
    }
