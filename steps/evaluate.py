"""Pipeline step 3: score the blend on the held-back days and decide whether it passes the gate.

Runs inside a SageMaker processing job (scikit-learn container). On the test days it computes the
mean absolute error (MAE, in °C) of:

    each raw model
    mean             the plain average of all models (wx itself shows the median)
    inverse_mae_45d  each model weighted by 1 / its MAE over the previous 45 days at that place
                     (a published multi-model method; see README)
    blend            the Ridge blend from train.py

Gate: the blend's overall MAE must be at least `margin` (2%) lower than inverse_mae_45d, and the
blend must not be worse than inverse_mae_45d at any place. The result goes in evaluation.json,
which the pipeline's condition step reads.

This file imports nothing from the rest of the repo, because SageMaker copies only this file.
"""

from __future__ import annotations  # type hints below also work on the container's Python

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

NOT_MODELS = ("place", "date", "actual", "is_test")


def predict(model: dict, rows: pd.DataFrame) -> pd.Series:
    """Intercept plus weighted sum of the model forecasts, with each place's own numbers."""
    values = []
    for _, row in rows.iterrows():
        place = model["places"][row["place"]]
        values.append(place["intercept"] + sum(w * row[m] for m, w in place["weights"].items()))
    return pd.Series(values, index=rows.index)


def inverse_mae_baseline(table: pd.DataFrame, models: list[str], window: int) -> pd.Series:
    """For each row, weight each model by 1 / its MAE over the previous `window` rows at that place.

    Rows with fewer than `window` earlier rows get the plain mean instead.
    """
    result = pd.Series(index=table.index, dtype=float)
    for _, rows in table.sort_values("date").groupby("place"):
        errors = rows[models].sub(rows["actual"], axis=0).abs()
        past_mae = errors.rolling(window).mean().shift(1)  # shift: never use the day itself
        weights = 1 / past_mae
        weights = weights.div(weights.sum(axis=1), axis=0)
        weighted = (weights * rows[models]).sum(axis=1, min_count=1)
        result[rows.index] = weighted.fillna(rows[models].mean(axis=1))
    return result


def champion_weights(table: pd.DataFrame, models: list[str], window: int) -> dict:
    """Today's inverse-MAE weights per place, from the last `window` days of data.

    This is what wx serves while no blend has passed the gate: the baseline stays the champion.
    """
    places = {}
    for place, rows in table.sort_values("date").groupby("place"):
        recent = rows.tail(window)
        errors = recent[models].sub(recent["actual"], axis=0).abs().mean()
        weights = (1 / errors) / (1 / errors).sum()
        places[place] = {
            "intercept": 0.0,
            "weights": {m: round(float(weights[m]), 4) for m in models},
        }
    return {"method": "inverse_mae_45d", "window_end": table["date"].max(), "places": places}


def mae(predicted: pd.Series, actual: pd.Series) -> float:
    return round(float((predicted - actual).abs().mean()), 3)


MIN_MARGIN = (
    0.02  # fixed before the first run; a run may ask for a stricter gate, never a looser one
)


def check_margin(margin: float) -> None:
    if not MIN_MARGIN <= margin < 1:
        raise ValueError(f"gate margin must be at least {MIN_MARGIN} and below 1, got {margin}")


def gate(result: dict, margin: float) -> bool:
    overall = result["overall"]
    if overall["blend"] > (1 - margin) * overall["inverse_mae_45d"]:
        return False
    return all(p["blend"] <= p["inverse_mae_45d"] for p in result["by_place"].values())


def evaluate(table: pd.DataFrame, model: dict, margin: float) -> dict:
    models = model["models"]
    table = table.copy()
    table["mean"] = table[models].mean(axis=1)
    table["inverse_mae_45d"] = inverse_mae_baseline(table, models, window=45)
    test = table[table["is_test"]].copy()
    test["blend"] = predict(model, test)

    names = models + ["mean", "inverse_mae_45d", "blend"]
    result = {
        "test_days": int(test["date"].nunique()),
        "test_start": test["date"].min(),
        "test_end": test["date"].max(),
        "overall": {n: mae(test[n], test["actual"]) for n in names},
        "by_place": {
            place: {n: mae(rows[n], rows["actual"]) for n in names}
            for place, rows in test.groupby("place")
        },
    }
    result["gate"] = {"margin": margin, "passed": int(gate(result, margin))}
    result["champion"] = champion_weights(table, models, window=45)
    # The format SageMaker Model Registry shows as model quality.
    result["regression_metrics"] = {"mae": {"value": result["overall"]["blend"]}}
    return result


def main(table_dir: Path, model_dir: Path, out_dir: Path, margin: float) -> None:
    check_margin(margin)
    archive = model_dir / "model.tar.gz"
    if archive.exists():  # inside SageMaker the training output arrives packed
        with tarfile.open(archive) as tar:
            tar.extract("model.json", model_dir)  # the one file train.py writes
    model = json.loads((model_dir / "model.json").read_text())
    table = pd.read_csv(table_dir / "table.csv")

    result = evaluate(table, model, margin)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    if (table_dir / "data_check.json").exists():
        shutil.copy(table_dir / "data_check.json", out_dir / "data_check.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, default=Path("/opt/ml/processing/table"))
    parser.add_argument("--model", type=Path, default=Path("/opt/ml/processing/model"))
    parser.add_argument("--output", type=Path, default=Path("/opt/ml/processing/evaluation"))
    parser.add_argument("--gate-margin", type=float, default=0.02)
    args = parser.parse_args()
    main(args.table, args.model, args.output, args.gate_margin)
