import export
import pytest

MODEL = {
    "models": ["a", "b"],
    "places": {"auckland": {"intercept": 0.5, "weights": {"a": 0.6, "b": 0.4}}},
    "train_start": "2024-02-05",
    "train_end": "2026-08-16",
}
EVALUATION = {"overall": {"blend": 0.5, "inverse_mae_45d": 0.6, "a": 0.7}}


def test_to_weights_shape():
    weights = export.to_weights(MODEL, EVALUATION, version=3, method="ridge_blend")
    assert weights["version"] == 3
    assert weights["method"] == "ridge_blend"
    assert weights["test_mae"] == {"blend": 0.5, "inverse_mae_45d": 0.6}
    auckland = weights["places"]["auckland"]
    assert (auckland["lat"], auckland["lon"]) == (-37.0, 174.8)
    assert auckland["weights"] == {"a": 0.6, "b": 0.4} and auckland["intercept"] == 0.5
    assert weights["models"] == ["a", "b"]


class EmptyGroup:
    def list_model_packages(self, **kwargs):
        return {"ModelPackageSummaryList": []}


def test_main_exits_when_group_is_empty(tmp_path, capsys):
    with pytest.raises(SystemExit) as stop:
        export.main(version=None, client=EmptyGroup(), s3=None, reports=tmp_path)
    assert stop.value.code == 1
    assert "No model in wx-blend; nothing exported." in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


CHAMPION = {
    "method": "inverse_mae_45d",
    "window_end": "2026-11-14",
    "places": {"auckland": {"intercept": 0.0, "weights": {"a": 0.5, "b": 0.5}}},
}


def test_rejected_version_ships_the_champion():
    evaluation = {**EVALUATION, "champion": CHAMPION}
    weights = export.serving_weights(MODEL, evaluation, version=4, status="Rejected")
    assert weights["method"] == "inverse_mae_45d"
    assert weights["places"]["auckland"]["weights"] == {"a": 0.5, "b": 0.5}
    assert weights["weights_as_of"] == "2026-11-14"


def test_approved_version_ships_the_blend():
    evaluation = {**EVALUATION, "champion": CHAMPION}
    weights = export.serving_weights(MODEL, evaluation, version=5, status="Approved")
    assert weights["method"] == "ridge_blend"
    assert weights["places"]["auckland"]["weights"] == {"a": 0.6, "b": 0.4}


def test_evaluation_uri_from_model_quality():
    package = {
        "ModelMetrics": {"ModelQuality": {"Statistics": {"S3Uri": "s3://b/e/evaluation.json"}}}
    }
    assert export.evaluation_uri(package) == "s3://b/e/evaluation.json"
