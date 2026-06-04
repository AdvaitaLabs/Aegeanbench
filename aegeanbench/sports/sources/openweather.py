"""
OpenWeatherMap free-tier adapter.

Used for knockout-stage predictions where weather (rain, heat, wind)
can materially shift outcomes. Free tier allows 60 calls/minute and
1M calls/month - far more than the 64 World Cup matches need.

API: https://openweathermap.org/api/one-call-3
Auth: ?appid=<key>

Defaults to mock when no key configured.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from aegeanbench.sports.cache import FileCache, get_default_cache

logger = logging.getLogger(__name__)


OWM_BASE = "https://api.openweathermap.org/data/2.5"
CACHE_TTL = timedelta(hours=3)


# World Cup 2026 host cities (US/Mexico/Canada) with rough lat/lon.
# Used so callers can look up by venue name without geocoding each time.
HOST_CITIES: Dict[str, Dict[str, float]] = {
    "New York":     {"lat": 40.71, "lon": -74.01},
    "Los Angeles":  {"lat": 34.05, "lon": -118.24},
    "Dallas":       {"lat": 32.78, "lon": -96.80},
    "Kansas City":  {"lat": 39.10, "lon": -94.58},
    "Atlanta":      {"lat": 33.75, "lon": -84.39},
    "Boston":       {"lat": 42.36, "lon": -71.06},
    "Houston":      {"lat": 29.76, "lon": -95.37},
    "Miami":        {"lat": 25.76, "lon": -80.19},
    "Philadelphia": {"lat": 39.95, "lon": -75.16},
    "San Francisco":{"lat": 37.77, "lon": -122.42},
    "Seattle":      {"lat": 47.61, "lon": -122.33},
    "Toronto":      {"lat": 43.65, "lon": -79.38},
    "Vancouver":    {"lat": 49.28, "lon": -123.12},
    "Mexico City":  {"lat": 19.43, "lon": -99.13},
    "Guadalajara":  {"lat": 20.66, "lon": -103.34},
    "Monterrey":    {"lat": 25.69, "lon": -100.31},
}


class OpenWeatherAdapter:
    """
    Pull a brief weather summary for a given match kickoff time + city.

    Returns a dict shaped for the prompt template:
        {
          "city": "Mexico City",
          "temperature_c": 22.4,
          "humidity_pct": 58,
          "wind_kph": 14.3,
          "precipitation_mm_h": 0.0,
          "conditions": "Clear",
          "kickoff_at": "2026-06-12T18:00:00Z",
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock: Optional[bool] = None,
        timeout: float = 5.0,
        cache: Optional[FileCache] = None,
    ):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if mock is None:
            mock = self.api_key is None
        self.mock = mock
        self.timeout = timeout
        self.cache = cache or get_default_cache()

    def fetch_for_match(
        self,
        city: str,
        kickoff_at: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Return weather summary near kickoff, or None on failure."""
        if self.mock:
            return self._mock_summary(city, kickoff_at)

        coords = HOST_CITIES.get(city)
        if coords is None:
            logger.info("openweather: unknown city %r; using mock", city)
            return self._mock_summary(city, kickoff_at)

        cache_key = ("openweather", "current", city, kickoff_at.isoformat())
        cached = self.cache.get(*cache_key, ttl=CACHE_TTL)
        if cached is not None:
            return cached

        try:
            raw = self._live_fetch(coords["lat"], coords["lon"])
            summary = self._parse(raw, city, kickoff_at)
            self.cache.set(summary, *cache_key)
            return summary
        except Exception as e:
            logger.warning("openweather live fetch failed (%s); using mock", e)
            return self._mock_summary(city, kickoff_at)

    # ----------------- internals -----------------

    def _live_fetch(self, lat: float, lon: float) -> Dict[str, Any]:
        import requests
        params = {
            "lat": lat, "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }
        resp = requests.get(f"{OWM_BASE}/weather", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse(raw: Dict[str, Any], city: str, kickoff_at: datetime) -> Dict[str, Any]:
        main = raw.get("main", {}) or {}
        wind = raw.get("wind", {}) or {}
        rain = (raw.get("rain", {}) or {}).get("1h", 0.0)
        conds_list = raw.get("weather", [{}])
        conds = conds_list[0].get("main", "Unknown") if conds_list else "Unknown"
        return {
            "city": city,
            "temperature_c": float(main.get("temp", 22.0)),
            "humidity_pct": int(main.get("humidity", 60)),
            "wind_kph": float(wind.get("speed", 0.0)) * 3.6,  # m/s -> kph
            "precipitation_mm_h": float(rain),
            "conditions": conds,
            "kickoff_at": kickoff_at.isoformat(),
        }

    @staticmethod
    def _mock_summary(city: str, kickoff_at: datetime) -> Dict[str, Any]:
        """Deterministic mock keyed by city name length so tests are stable."""
        base_temp = 22.0 + (len(city) % 8)
        return {
            "city": city,
            "temperature_c": base_temp,
            "humidity_pct": 60,
            "wind_kph": 12.0,
            "precipitation_mm_h": 0.0,
            "conditions": "Clear",
            "kickoff_at": kickoff_at.isoformat(),
            "_mock": True,
        }
