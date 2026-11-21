"""Pipeline step 2: learn the blend.

Runs as a SageMaker training job (scikit-learn container). For each place it fits a Ridge
regression: inputs are the models' forecast maxima, target is the actual maximum. The result
is a few numbers per place (an intercept and one weight per model), saved as model.json, so
any service can apply the blend with plain arithmetic. wx does exactly that in its Worker.

This file imports nothing from the rest of the repo, because SageMaker copies only this file.
"""

from __future__ import annotations  # type hints below also work on the container's Python

import argparse
import json
import os
import tarfile
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge

NOT_MODELS = ("place", "date", "actual", "is_test")


def fit(train: pd.DataFrame, models: list[str]) -> dict:
    places = {}
    for place, rows in train.groupby("place"):
        ridge = Ridge(alpha=1.0).fit(rows[models], rows["actual"])
        weights = dict(zip(models, ridge.coef_))  # noqa: B905 (strict= needs Python 3.10)
        places[place] = {
            "intercept": round(float(ridge.intercept_), 4),
            "weights": {m: round(float(w), 4) for m, w in weights.items()},
        }
    return {
        "models": models,
        "places": places,
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --pack: also write model.tar.gz. A training job packs the model folder itself; when this
    # script runs as a processing job instead (see pipeline.py --train-in-processing), it must.
    parser.add_argument("--pack", action="store_true")
    args = parser.parse_args()

    # SageMaker puts the input channel and the model folder in these environment variables.
    data_dir = Path(os.environ.get("SM_CHANNEL_TRAIN", "build/table"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "build/model"))

    table = pd.read_csv(data_dir / "table.csv")
    train = table[~table["is_test"]]
    models = [c for c in table.columns if c not in NOT_MODELS]
    model = fit(train, models)

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    if args.pack:
        with tarfile.open(model_dir / "model.tar.gz", "w:gz") as tar:
            tar.add(model_dir / "model.json", arcname="model.json")
    print(json.dumps(model, indent=2))
