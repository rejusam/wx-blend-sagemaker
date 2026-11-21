import evaluate
import numpy as np
import pandas as pd


def test_predict_uses_place_weights():
    model = {
        "models": ["a", "b"],
        "places": {
            "auckland": {"intercept": 1.0, "weights": {"a": 0.5, "b": 0.5}},
            "wellington": {"intercept": 0.0, "weights": {"a": 1.0, "b": 0.0}},
        },
    }
    rows = pd.DataFrame({"place": ["auckland", "wellington"], "a": [10.0, 10.0], "b": [20.0, 20.0]})
    assert list(evaluate.predict(model, rows)) == [16.0, 10.0]


def test_inverse_mae_baseline_prefers_better_model():
    days = 60
    dates = pd.date_range("2024-01-01", periods=days).strftime("%Y-%m-%d")
    actual = np.full(days, 20.0)
    table = pd.DataFrame(
        {
            "place": "auckland",
            "date": dates,
            "good": actual + 1.0,
            "bad": actual + 3.0,
            "actual": actual,
        }
    )
    blend = evaluate.inverse_mae_baseline(table, ["good", "bad"], window=45)
    # Day 46 on: weights 1/1 and 1/3, normalised to 0.75 and 0.25 -> 21.5
    assert abs(blend.iloc[50] - 21.5) < 1e-9
    # Before 45 days of history exist, fall back to the plain mean
    assert abs(blend.iloc[10] - 22.0) < 1e-9


def result(blend, baseline, by_place):
    return {
        "overall": {"blend": blend, "inverse_mae_45d": baseline},
        "by_place": {p: {"blend": b, "inverse_mae_45d": base} for p, (b, base) in by_place.items()},
    }


def test_gate_needs_margin_overall():
    places = {"auckland": (0.97, 1.0)}
    assert evaluate.gate(result(0.97, 1.0, places), margin=0.02)
    assert not evaluate.gate(result(0.99, 1.0, places), margin=0.02)


def test_gate_fails_if_one_place_worse():
    places = {"auckland": (0.80, 1.0), "wellington": (1.01, 1.0)}
    assert not evaluate.gate(result(0.90, 1.0, places), margin=0.02)


def test_gate_margin_cannot_be_loosened():
    import pytest

    for margin in [-0.5, 0.0, 0.019, 1.0]:
        with pytest.raises(ValueError):
            evaluate.check_margin(margin)
    evaluate.check_margin(0.02)
    evaluate.check_margin(0.5)


def test_champion_weights_use_the_last_45_days():
    days = 100
    dates = pd.date_range("2024-01-01", periods=days).strftime("%Y-%m-%d")
    actual = np.full(days, 20.0)
    good = actual + 1.0
    good[:55] = actual[:55] + 9.0  # bad early on, good in the last 45 days
    table = pd.DataFrame(
        {"place": "auckland", "date": dates, "good": good, "bad": actual + 3.0, "actual": actual}
    )
    champion = evaluate.champion_weights(table, ["good", "bad"], window=45)
    assert champion["method"] == "inverse_mae_45d"
    assert champion["window_end"] == "2024-04-09"
    auckland = champion["places"]["auckland"]
    assert auckland["intercept"] == 0.0
    assert auckland["weights"] == {"good": 0.75, "bad": 0.25}
