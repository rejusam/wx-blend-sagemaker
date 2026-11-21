"""Copy a registered model's evidence into reports/, and write the weights wx should serve.

    uv run python scripts/export.py               # latest version in the group
    uv run python scripts/export.py --version 1   # a given version

Always writes reports/evaluation_v<N>.json and reports/data_check_v<N>.json for that version.
Writes build/blend-weights.json (the file wx reads): the Ridge blend if the version passed the
gate (Approved); otherwise the champion, the 45-day inverse-MAE weights from the same run.
With an empty model group it writes nothing and exits with code 1.
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path
from urllib.parse import urlparse

from places import ACCOUNT, MODEL_GROUP, PLACES, REGION, require_account

WEIGHTS_FILE = Path("build/blend-weights.json")


def to_weights(model: dict, evaluation: dict, version: int, method: str) -> dict:
    """The small file wx bundles: per place, where it is and how to blend."""
    return {
        "version": version,
        "method": method,
        "models": model["models"],
        "train_start": model["train_start"],
        "train_end": model["train_end"],
        "test_mae": {k: evaluation["overall"][k] for k in ("blend", "inverse_mae_45d")},
        "places": {
            place: {
                "lat": PLACES[place][0],
                "lon": PLACES[place][1],
                "intercept": numbers["intercept"],
                "weights": numbers["weights"],
            }
            for place, numbers in model["places"].items()
        },
    }


def evaluation_uri(package: dict) -> str:
    """Where the pipeline's evaluation.json is; attached to the package as model quality."""
    return package["ModelMetrics"]["ModelQuality"]["Statistics"]["S3Uri"]


def serving_weights(model: dict, evaluation: dict, version: int, status: str) -> dict:
    """What wx should serve after this version: the blend if it passed the gate, else the champion.

    The champion is the 45-day inverse-MAE weighting, recomputed on the latest data each run.
    """
    if status == "Approved":
        weights = to_weights(model, evaluation, version, method="ridge_blend")
        weights["weights_as_of"] = model["train_end"]
        return weights
    champion = evaluation["champion"]
    weights = to_weights(
        {**model, "places": champion["places"]}, evaluation, version, method=champion["method"]
    )
    weights["weights_as_of"] = champion["window_end"]
    return weights


def read_s3(s3, uri: str) -> bytes:
    parsed = urlparse(uri)
    return s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()


def package_arn(client, version: int | None) -> str | None:
    if version is not None:
        return f"arn:aws:sagemaker:{REGION}:{ACCOUNT}:model-package/{MODEL_GROUP}/{version}"
    latest = client.list_model_packages(
        ModelPackageGroupName=MODEL_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )["ModelPackageSummaryList"]
    return latest[0]["ModelPackageArn"] if latest else None


def main(version: int | None, client, s3, reports: Path) -> None:
    arn = package_arn(client, version)
    if arn is None:
        print(f"No model in {MODEL_GROUP}; nothing exported.")
        sys.exit(1)

    package = client.describe_model_package(ModelPackageName=arn)
    number = package["ModelPackageVersion"]
    status = package["ModelApprovalStatus"]
    uri = evaluation_uri(package)
    evaluation = json.loads(read_s3(s3, uri))
    data_check = json.loads(read_s3(s3, uri.replace("evaluation.json", "data_check.json")))

    reports.mkdir(exist_ok=True)
    for name, content in [("evaluation", evaluation), ("data_check", data_check)]:
        content = {
            "model_package_version": number,
            "approval_status": status,
            # The S3 path names the pipeline run (execution id) that produced this evaluation.
            "evaluation_s3_uri": uri,
            **content,
        }
        (reports / f"{name}_v{number}.json").write_text(json.dumps(content, indent=2) + "\n")
    print(f"Version {number} ({status}): evidence saved in {reports}/")

    archive = read_s3(s3, package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        model = json.load(tar.extractfile("model.json"))
    if status != "Approved" and "champion" not in evaluation:
        print("Not approved, and this evaluation has no champion weights; no weights written.")
        return
    weights = serving_weights(model, evaluation, number, status)
    WEIGHTS_FILE.parent.mkdir(exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps(weights, indent=2) + "\n")
    print(f"Weights for wx ({weights['method']}) written to {WEIGHTS_FILE}")


if __name__ == "__main__":
    import boto3

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", type=int, help="model package version; default: latest")
    args = parser.parse_args()
    require_account()
    main(
        args.version,
        client=boto3.client("sagemaker", region_name=REGION),
        s3=boto3.client("s3", region_name=REGION),
        reports=Path("reports"),
    )
