"""Define the SageMaker Pipeline, and with --run, start it, wait, and record the result.

    Prepare (processing) -> Train (training) -> Evaluate (processing) -> CheckGate
                                                         gate passed:  RegisterApproved
                                                         gate failed:  RegisterRejected

Both branches put the model in the Model Registry group wx-blend, so every run leaves a
numbered version with its evaluation attached, approved or not.

Usage:
    uv run python scripts/pipeline.py --run --raw s3://wx-blend-.../raw/2026-11-21/
    uv run python scripts/pipeline.py --run --raw ... --gate-margin 0.5   # rejected on purpose
"""

import argparse
import json
import os
from pathlib import Path

# Load this repo's SDK settings (telemetry off) before the SDK is imported.
os.environ.setdefault(
    "SAGEMAKER_USER_CONFIG_OVERRIDE", str(Path(__file__).with_name("sagemaker.yaml"))
)

import boto3
from places import ACCOUNT, BUCKET, MODEL_GROUP, PIPELINE_NAME, REGION, require_account
from sagemaker import image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# The newest scikit-learn container this SDK version has for both training and serving.
SKLEARN_VERSION = "1.2-1"
# Smallest instances this account has a quota for: processing on t3.medium, training on m5.large.
PROCESSING_INSTANCE = "ml.t3.medium"
TRAINING_INSTANCE = "ml.m5.large"
MAX_RUNTIME = 1800  # seconds; a job that runs longer is stopped (the AWS default is 24 hours)
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/wx-blend-sagemaker"
RUNS_FILE = Path("reports/pipeline_runs.json")


def build_pipeline(
    role_arn: str, session: PipelineSession, train_in_processing: bool = False
) -> Pipeline:
    raw_uri = ParameterString("RawDataUri")
    test_days = ParameterInteger("TestDays", default_value=90)
    gate_margin = ParameterFloat("GateMargin", default_value=0.02)

    processor = SKLearnProcessor(
        framework_version=SKLEARN_VERSION,
        role=role_arn,
        instance_type=PROCESSING_INSTANCE,
        instance_count=1,
        max_runtime_in_seconds=MAX_RUNTIME,
        sagemaker_session=session,
    )

    prepare = ProcessingStep(
        name="Prepare",
        step_args=processor.run(
            code="steps/prepare.py",
            inputs=[ProcessingInput(source=raw_uri, destination="/opt/ml/processing/raw")],
            outputs=[ProcessingOutput(output_name="table", source="/opt/ml/processing/table")],
            arguments=["--test-days", test_days.to_string()],
        ),
    )
    table_uri = prepare.properties.ProcessingOutputConfig.Outputs["table"].S3Output.S3Uri

    if train_in_processing:
        # Fallback for an account with no training-job quota: same script, run as a processing
        # job, with the folders a training job would give it.
        train_processor = SKLearnProcessor(
            framework_version=SKLEARN_VERSION,
            role=role_arn,
            instance_type=PROCESSING_INSTANCE,
            instance_count=1,
            max_runtime_in_seconds=MAX_RUNTIME,
            env={
                "SM_CHANNEL_TRAIN": "/opt/ml/processing/table",
                "SM_MODEL_DIR": "/opt/ml/processing/model",
            },
            sagemaker_session=session,
        )
        train = ProcessingStep(
            name="Train",
            step_args=train_processor.run(
                code="steps/train.py",
                inputs=[ProcessingInput(source=table_uri, destination="/opt/ml/processing/table")],
                outputs=[ProcessingOutput(output_name="model", source="/opt/ml/processing/model")],
                arguments=["--pack"],
            ),
        )
        model_folder = train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri
        model_uri = Join(on="/", values=[model_folder, "model.tar.gz"])
    else:
        estimator = SKLearn(
            entry_point="steps/train.py",
            framework_version=SKLEARN_VERSION,
            role=role_arn,
            instance_type=TRAINING_INSTANCE,
            instance_count=1,
            max_runtime_in_seconds=MAX_RUNTIME,
            sagemaker_session=session,
        )
        train = TrainingStep(
            name="Train",
            step_args=estimator.fit(inputs={"train": TrainingInput(s3_data=table_uri)}),
        )
        model_uri = train.properties.ModelArtifacts.S3ModelArtifacts

    evaluation_file = PropertyFile(
        name="Evaluation", output_name="evaluation", path="evaluation.json"
    )
    evaluate = ProcessingStep(
        name="Evaluate",
        step_args=processor.run(
            code="steps/evaluate.py",
            inputs=[
                ProcessingInput(source=table_uri, destination="/opt/ml/processing/table"),
                ProcessingInput(source=model_uri, destination="/opt/ml/processing/model"),
            ],
            outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
            ],
            arguments=["--gate-margin", gate_margin.to_string()],
        ),
        property_files=[evaluation_file],
    )
    evaluation_uri = evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri

    # The registered model is the training output (model.json inside model.tar.gz) plus the
    # standard scikit-learn serving image, so it could be hosted later without repacking.
    model = Model(
        image_uri=image_uris.retrieve(
            "sklearn",
            REGION,
            version=SKLEARN_VERSION,
            image_scope="inference",
            instance_type=TRAINING_INSTANCE,
        ),
        model_data=model_uri,
        role=role_arn,
        sagemaker_session=session,
    )
    metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(on="/", values=[evaluation_uri, "evaluation.json"]),
            content_type="application/json",
        )
    )

    def register(status: str) -> ModelStep:
        return ModelStep(
            name=f"Register{status}",
            step_args=model.register(
                content_types=["application/json"],
                response_types=["application/json"],
                inference_instances=[TRAINING_INSTANCE],
                transform_instances=[TRAINING_INSTANCE],
                model_package_group_name=MODEL_GROUP,
                approval_status=status,
                model_metrics=metrics,
            ),
        )

    check_gate = ConditionStep(
        name="CheckGate",
        conditions=[
            ConditionEquals(
                left=JsonGet(
                    step_name=evaluate.name, property_file=evaluation_file, json_path="gate.passed"
                ),
                right=1,
            )
        ],
        if_steps=[register("Approved")],
        else_steps=[register("Rejected")],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[raw_uri, test_days, gate_margin],
        steps=[prepare, train, evaluate, check_gate],
        sagemaker_session=session,
    )


def run_record(execution: dict, steps: list[dict], parameters: dict, package: dict) -> dict:
    """The facts about one run that go in reports/pipeline_runs.json."""
    train = next((s for s in steps if s["StepName"] == "Train"), {})
    train_step = "training job" if "TrainingJob" in train.get("Metadata", {}) else "processing job"
    return {
        "execution_arn": execution["PipelineExecutionArn"],
        "started": execution["CreationTime"].isoformat(),
        "ended": execution["LastModifiedTime"].isoformat(),
        "status": execution["PipelineExecutionStatus"],
        # The IAM identity that started the run, as AWS recorded it (laptop user or GitHub role).
        "started_by": execution.get("CreatedBy", {}).get("IamIdentity", {}).get("Arn"),
        "gate_margin": float(parameters.get("GateMargin", 0.02)),
        "train_step": train_step if train else None,
        "steps": {s["StepName"]: s["StepStatus"] for s in reversed(steps)},
        "model_package_arn": package.get("ModelPackageArn"),
        "model_package_version": package.get("ModelPackageVersion"),
        "approval_status": package.get("ModelApprovalStatus"),
    }


def registered_package(steps: list[dict], client, execution_id: str = "") -> dict:
    """Find the model package the run registered, whichever branch it took.

    If the register step failed after creating the package, its metadata has no ARN; then the
    package is found by its model file, whose S3 path contains the run's execution id.
    """
    for step in steps:
        arn = step.get("Metadata", {}).get("RegisterModel", {}).get("Arn")
        if arn:
            return client.describe_model_package(ModelPackageName=arn)
    if execution_id:
        summaries = client.list_model_packages(ModelPackageGroupName=MODEL_GROUP, MaxResults=100)
        for summary in summaries["ModelPackageSummaryList"]:
            package = client.describe_model_package(ModelPackageName=summary["ModelPackageArn"])
            model_url = package["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
            if f"/{execution_id}/" in model_url:
                return package
    return {}


def record(execution_arn: str) -> None:
    """Append one finished run to reports/pipeline_runs.json."""
    client = boto3.client("sagemaker", region_name=REGION)
    execution = client.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    steps = client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)[
        "PipelineExecutionSteps"
    ]
    parameters = client.list_pipeline_parameters_for_execution(PipelineExecutionArn=execution_arn)[
        "PipelineParameters"
    ]
    parameters = {p["Name"]: p["Value"] for p in parameters}
    package = registered_package(steps, client, execution_id=execution_arn.rsplit("/", 1)[-1])
    entry = run_record(execution, steps, parameters, package)

    runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else []
    runs = [r for r in runs if r["execution_arn"] != execution_arn]  # recording twice replaces
    runs.append(entry)
    RUNS_FILE.parent.mkdir(exist_ok=True)
    RUNS_FILE.write_text(json.dumps(runs, indent=2) + "\n")
    print(json.dumps(entry, indent=2))


def start(raw: str, gate_margin: float, role_arn: str, train_in_processing: bool) -> str:
    """Create or update the pipeline, start a run, wait for it to finish, return its ARN."""
    session = PipelineSession(default_bucket=BUCKET)  # never let the SDK create its own bucket
    pipeline = build_pipeline(role_arn, session, train_in_processing)
    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start(parameters={"RawDataUri": raw, "GateMargin": gate_margin})
    print("Started", execution.arn)
    print("If this script stops before the run ends: pipeline.py --record", execution.arn)
    try:
        execution.wait(delay=30, max_attempts=120)
    except Exception as error:  # a failed step raises here; the record says which one
        print("Pipeline did not succeed:", error)
    return execution.arn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="upsert and start the pipeline")
    parser.add_argument("--record", metavar="EXECUTION_ARN", help="record a finished run")
    parser.add_argument("--raw", help="S3 folder with the raw files from fetch.py")
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--role-arn", default=ROLE_ARN)
    parser.add_argument(
        "--train-in-processing",
        action="store_true",
        help="run Train as a processing job (for an account with no training-job quota)",
    )
    args = parser.parse_args()
    require_account()
    if args.record:
        record(args.record)
    elif args.run:
        if not args.raw:
            parser.error("--raw is required with --run")
        record(start(args.raw, args.gate_margin, args.role_arn, args.train_in_processing))
    else:
        session = PipelineSession(default_bucket=BUCKET)
        print(json.dumps(json.loads(build_pipeline(args.role_arn, session).definition()), indent=2))
