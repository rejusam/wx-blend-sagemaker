"""Pipeline step 1: turn raw Open-Meteo files into one clean table.

Runs inside a SageMaker processing job (scikit-learn container). Reads the raw JSON files
saved by scripts/fetch.py, writes:

    table.csv        place, date, one column per kept model, actual, is_test
    data_check.json  date range, row counts, which models were kept or dropped and why

Fails (exit code 1) if the data checks find a problem, which stops the pipeline.
This file imports nothing from the rest of the repo, because SageMaker copies only this file.
"""

from __future__ import annotations  # type hints below also work on the container's Python

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PREFIX = "temperature_2m_previous_day1_"
LOWEST, HIGHEST = -30.0, 50.0  # °C; anything outside is a data error at these places


def daily_max(forecast: dict) -> pd.DataFrame:
    """Each model's highest hourly forecast per local date.

    A day with any missing hour is missing (NaN), not the max of the hours that are there.
    """
    hourly = pd.DataFrame(forecast["hourly"])
    hourly["date"] = hourly["time"].str[:10]  # times are already local (timezone in the request)
    hourly = hourly.drop(columns="time").rename(columns=lambda c: c.replace(PREFIX, ""))
    hourly = hourly.set_index("date").astype(float)
    highest = hourly.groupby(level="date").max()
    complete = hourly.notna().groupby(level="date").all()
    return highest.where(complete).reset_index()


def build_table(raw_dir: Path) -> pd.DataFrame:
    """One row per place and date: each model's forecast maximum, and the actual maximum."""
    parts = []
    for forecast_file in sorted(raw_dir.glob("forecast_*.json")):
        place = forecast_file.stem[len("forecast_") :]
        forecasts = daily_max(json.loads(forecast_file.read_text()))
        actuals = json.loads((raw_dir / f"actuals_{place}.json").read_text())["daily"]
        actual = pd.DataFrame({"date": actuals["time"], "actual": actuals["temperature_2m_max"]})
        part = forecasts.merge(actual, on="date", how="inner")
        part.insert(0, "place", place)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def select_models(
    table: pd.DataFrame, models: list[str], min_coverage: float
) -> tuple[list[str], dict[str, str]]:
    """Keep models that have a forecast on at least `min_coverage` of all rows."""
    kept, dropped = [], {}
    for model in models:
        coverage = table[model].notna().mean()
        if coverage >= min_coverage:
            kept.append(model)
        else:
            dropped[model] = f"forecast present on {coverage:.1%} of days, below {min_coverage:.0%}"
    return kept, dropped


def check(table: pd.DataFrame, models: list[str], rows_before: int) -> list[str]:
    """Return a list of problems; an empty list means the table is fit to train on."""
    problems = []
    duplicates = table.duplicated(["place", "date"]).sum()
    if duplicates:
        problems.append(f"{duplicates} duplicate place/date rows")
    values = table[models + ["actual"]]
    outside = ((values < LOWEST) | (values > HIGHEST)).sum().sum()
    if outside:
        problems.append(f"{outside} temperatures outside {LOWEST} to {HIGHEST} °C")
    if values.isna().any().any():
        problems.append("missing values left after dropping incomplete rows")
    dropped_share = 1 - len(table) / rows_before
    if dropped_share > 0.10:
        problems.append(f"{dropped_share:.1%} of rows dropped for missing data, more than 10%")
    return problems


def mark_test(table: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Hold back the last `test_days` dates as the test set. Split by time, never at random."""
    dates = sorted(table["date"].unique())
    first_test_date = dates[-test_days]
    table = table.sort_values(["place", "date"]).reset_index(drop=True)
    table["is_test"] = table["date"] >= first_test_date
    return table


def main(raw_dir: Path, out_dir: Path, test_days: int, min_coverage: float) -> int:
    table = build_table(raw_dir)
    all_models = [c for c in table.columns if c not in ("place", "date", "actual")]
    models, dropped = select_models(table, all_models, min_coverage)

    rows_before = len(table)
    table = table[["place", "date"] + models + ["actual"]].dropna()
    problems = check(table, models, rows_before)
    if problems:
        print("Data check failed:", *problems, sep="\n  ")
        return 1

    table = mark_test(table, test_days)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table.csv", index=False)
    summary = {
        "start": table["date"].min(),
        "end": table["date"].max(),
        "places": sorted(table["place"].unique()),
        "rows_before": rows_before,
        "rows": len(table),
        "rows_dropped": rows_before - len(table),
        "models_kept": models,
        "models_dropped": dropped,
        "test_days": test_days,
        "test_start": table.loc[table["is_test"], "date"].min(),
    }
    (out_dir / "data_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/opt/ml/processing/raw"))
    parser.add_argument("--output", type=Path, default=Path("/opt/ml/processing/table"))
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    args = parser.parse_args()
    sys.exit(main(args.input, args.output, args.test_days, args.min_coverage))
