from datetime import UTC, datetime

import pipeline

STEPS = [
    {
        "StepName": "RegisterRejected",
        "StepStatus": "Succeeded",
        "Metadata": {"RegisterModel": {"Arn": "arn:pkg/wx-blend/2"}},
    },
    {"StepName": "CheckGate", "StepStatus": "Succeeded", "Metadata": {}},
    {"StepName": "Evaluate", "StepStatus": "Succeeded", "Metadata": {}},
    {
        "StepName": "Train",
        "StepStatus": "Succeeded",
        "Metadata": {"TrainingJob": {"Arn": "arn:training-job/x"}},
    },
]


class FakeClient:
    def describe_model_package(self, ModelPackageName):
        return {
            "ModelPackageArn": ModelPackageName,
            "ModelPackageVersion": 2,
            "ModelApprovalStatus": "Rejected",
        }


def test_registered_package_found_on_either_branch():
    package = pipeline.registered_package(STEPS, FakeClient())
    assert package["ModelPackageVersion"] == 2
    assert pipeline.registered_package(STEPS[1:], FakeClient()) == {}


def test_run_record_has_the_facts_for_reports():
    t = datetime(2026, 11, 21, 8, 0, tzinfo=UTC)
    execution = {
        "PipelineExecutionArn": "arn:exec/1",
        "CreationTime": t,
        "LastModifiedTime": t,
        "PipelineExecutionStatus": "Succeeded",
    }
    package = pipeline.registered_package(STEPS, FakeClient())
    record = pipeline.run_record(execution, STEPS, {"GateMargin": "0.5"}, package)
    assert record["approval_status"] == "Rejected"
    assert record["model_package_version"] == 2
    assert record["gate_margin"] == 0.5
    assert record["train_step"] == "training job"
    assert record["steps"]["CheckGate"] == "Succeeded"
    assert record["started"] == "2026-11-21T08:00:00+00:00"


def test_run_record_spots_train_as_processing_job():
    t = datetime(2026, 11, 21, 8, 0, tzinfo=UTC)
    execution = {
        "PipelineExecutionArn": "arn:exec/2",
        "CreationTime": t,
        "LastModifiedTime": t,
        "PipelineExecutionStatus": "Failed",
    }
    steps = [
        {
            "StepName": "Train",
            "StepStatus": "Succeeded",
            "Metadata": {"ProcessingJob": {"Arn": "arn:processing-job/y"}},
        }
    ]
    record = pipeline.run_record(execution, steps, {}, {})
    assert record["train_step"] == "processing job"
    assert record["gate_margin"] == 0.02 and record["approval_status"] is None


class GroupClient:
    """A group with two packages; the register step left no ARN in the run's metadata."""

    def list_model_packages(self, **kwargs):
        return {
            "ModelPackageSummaryList": [
                {"ModelPackageArn": "arn:pkg/wx-blend/1"},
                {"ModelPackageArn": "arn:pkg/wx-blend/2"},
            ]
        }

    def describe_model_package(self, ModelPackageName):
        execution = {"arn:pkg/wx-blend/1": "aaa111", "arn:pkg/wx-blend/2": "bbb222"}
        url = f"s3://b/wx-blend/{execution[ModelPackageName]}/Train/output/model/model.tar.gz"
        return {
            "ModelPackageArn": ModelPackageName,
            "ModelPackageVersion": int(ModelPackageName[-1]),
            "ModelApprovalStatus": "Rejected",
            "InferenceSpecification": {"Containers": [{"ModelDataUrl": url}]},
        }


def test_registered_package_found_by_model_path_when_step_failed():
    steps = [{"StepName": "RegisterRejected-RegisterModel", "StepStatus": "Failed", "Metadata": {}}]
    package = pipeline.registered_package(steps, GroupClient(), execution_id="aaa111")
    assert package["ModelPackageVersion"] == 1


GITHUB_ROLE = "arn:aws:sts::1:assumed-role/wx-blend-github/GitHubActions"


def test_run_record_says_who_started_it():
    t = datetime(2026, 11, 21, 8, 0, tzinfo=UTC)
    execution = {
        "PipelineExecutionArn": "arn:exec/3",
        "CreationTime": t,
        "LastModifiedTime": t,
        "PipelineExecutionStatus": "Succeeded",
        "CreatedBy": {
            "IamIdentity": {"Arn": "arn:aws:sts::1:assumed-role/wx-blend-github/GitHubActions"}
        },
    }
    record = pipeline.run_record(execution, [], {}, {})
    assert record["started_by"].endswith("wx-blend-github/GitHubActions")
