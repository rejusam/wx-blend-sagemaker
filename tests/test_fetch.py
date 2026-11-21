from datetime import date
from urllib.parse import parse_qs, urlparse

import fetch
from places import MODELS, PLACES, TIMEZONE


def query(url):
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_forecast_url_has_all_models_and_timezone():
    lat, lon = PLACES["auckland"]
    url = fetch.forecast_url(lat, lon, date(2024, 1, 1), date(2024, 1, 3))
    q = query(url)
    assert url.startswith("https://previous-runs-api.open-meteo.com/v1/forecast?")
    assert q["hourly"] == "temperature_2m_previous_day1"
    assert q["models"].split(",") == MODELS
    assert q["timezone"] == TIMEZONE
    assert q["start_date"] == "2024-01-01" and q["end_date"] == "2024-01-03"
    assert q["latitude"] == "-37.0" and q["longitude"] == "174.8"


def test_actuals_url_daily_max():
    lat, lon = PLACES["wellington"]
    q = query(fetch.actuals_url(lat, lon, date(2024, 1, 1), date(2024, 1, 3)))
    assert q["daily"] == "temperature_2m_max"
    assert q["timezone"] == TIMEZONE
    assert "models" not in q
