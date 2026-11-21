"""Download past forecasts and actual temperatures from Open-Meteo and save them unchanged.

Forecasts: each model's hourly 2 m temperature as it was forecast one day ahead
(Previous Runs API). Actuals: daily maximum temperature from the Historical Weather API,
which is a reanalysis (a best estimate from many observations), not one station's reading.

Usage:
    uv run python scripts/fetch.py --out build/raw            # save to a local folder
    uv run python scripts/fetch.py --out build/raw --s3       # and upload to S3 raw/<run date>/
"""

import argparse
import json
import os
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

from places import BUCKET, FIRST_DAY, MODELS, PLACES, TIMEZONE, require_account

FORECAST_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"


def forecast_url(lat: float, lon: float, start: date, end: date) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m_previous_day1",
        "models": ",".join(MODELS),
        "timezone": TIMEZONE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    return f"{FORECAST_API}?{urlencode(params, safe=',/')}"


def actuals_url(lat: float, lon: float, start: date, end: date) -> str:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max",
        "timezone": TIMEZONE,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    return f"{ARCHIVE_API}?{urlencode(params, safe=',/')}"


def download(url: str) -> bytes:
    """GET a URL, retrying twice on a network error or a 5xx/429 reply."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("unreachable")


def main(out_dir: Path, end: date, upload: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(FIRST_DAY)
    for place, (lat, lon) in PLACES.items():
        for kind, url in [
            ("forecast", forecast_url(lat, lon, start, end)),
            ("actuals", actuals_url(lat, lon, start, end)),
        ]:
            body = download(url)
            json.loads(body)  # fail now, not inside SageMaker, if the reply is not JSON
            path = out_dir / f"{kind}_{place}.json"
            path.write_bytes(body)
            print(f"{path}  {len(body):,} bytes")

    if upload:
        require_account()
        import boto3

        s3 = boto3.client("s3")
        prefix = f"raw/{date.today().isoformat()}"
        for path in sorted(out_dir.glob("*.json")):
            s3.upload_file(str(path), BUCKET, f"{prefix}/{path.name}")
        uri = f"s3://{BUCKET}/{prefix}/"
        print(uri)
        if "GITHUB_OUTPUT" in os.environ:  # hand the exact folder to the next workflow step
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"raw_uri={uri}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("build/raw"))
    # The archive lags a few days behind today, so stop a week back.
    parser.add_argument("--end", type=date.fromisoformat, default=date.today() - timedelta(days=7))
    parser.add_argument("--s3", action="store_true", help="also upload to S3")
    args = parser.parse_args()
    main(args.out, args.end, args.s3)
