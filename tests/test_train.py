import numpy as np
import pandas as pd
import train


def synthetic(place="auckland", days=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(18, 4, days)
    b = rng.normal(18, 4, days)
    dates = pd.date_range("2024-01-01", periods=days).strftime("%Y-%m-%d")
    return pd.DataFrame(
        {"place": place, "date": dates, "a": a, "b": b, "actual": 0.7 * a + 0.3 * b + 1.0}
    )


def test_fit_recovers_known_weights():
    model = train.fit(synthetic(), ["a", "b"])
    auckland = model["places"]["auckland"]
    assert model["models"] == ["a", "b"]
    assert abs(auckland["weights"]["a"] - 0.7) < 0.01
    assert abs(auckland["weights"]["b"] - 0.3) < 0.01
    assert abs(auckland["intercept"] - 1.0) < 0.2
    assert model["train_start"] == "2024-01-01" and model["train_end"] == "2024-07-18"


def test_fit_one_model_per_place():
    both = pd.concat([synthetic("auckland"), synthetic("wellington", seed=1)])
    model = train.fit(both, ["a", "b"])
    assert sorted(model["places"]) == ["auckland", "wellington"]


def test_pack_writes_model_archive(tmp_path, monkeypatch):
    import subprocess
    import sys
    import tarfile
    from pathlib import Path

    table = synthetic()
    table["is_test"] = False
    (tmp_path / "data").mkdir()
    table.to_csv(tmp_path / "data" / "table.csv", index=False)
    env = {
        "SM_CHANNEL_TRAIN": str(tmp_path / "data"),
        "SM_MODEL_DIR": str(tmp_path / "model"),
        "PATH": "/usr/bin:/bin",
    }
    script = Path(__file__).parents[1] / "steps" / "train.py"
    subprocess.run(
        [sys.executable, str(script), "--pack"], env=env, check=True, capture_output=True
    )
    with tarfile.open(tmp_path / "model" / "model.tar.gz") as tar:
        assert tar.getnames() == ["model.json"]
