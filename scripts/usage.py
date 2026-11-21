"""Write reports/usage.json: SageMaker jobs run for this project, their billed minutes, and
whether anything is still running. Cost Explorer had no data for this new account, so this
records usage from the SageMaker API instead of a dollar figure.

    uv run python scripts/usage.py --since 2026-11-21
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from places import REGION


def main(since: datetime) -> None:
    sm = boto3.client("sagemaker", region_name=REGION)
    jobs = sm.list_processing_jobs(CreationTimeAfter=since, MaxResults=100)
    rows = []
    for summary in jobs["ProcessingJobSummaries"]:
        job = sm.describe_processing_job(ProcessingJobName=summary["ProcessingJobName"])
        start, end = job.get("ProcessingStartTime"), job.get("ProcessingEndTime")
        rows.append(
            {
                "status": job["ProcessingJobStatus"],
                "instance": job["ProcessingResources"]["ClusterConfig"]["InstanceType"],
                "seconds": (end - start).total_seconds() if start and end else 0,
            }
        )
    training = sm.list_training_jobs(CreationTimeAfter=since)["TrainingJobSummaries"]
    usage = {
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
        "since": since.date().isoformat(),
        "processing_jobs": len(rows),
        "processing_job_minutes_total": round(sum(r["seconds"] for r in rows) / 60, 1),
        "processing_instance_types": sorted({r["instance"] for r in rows}),
        "training_jobs": len(training),
        "jobs_in_progress": sum(r["status"] == "InProgress" for r in rows),
        "endpoints": len(sm.list_endpoints()["Endpoints"]),
        "notebook_instances": len(sm.list_notebook_instances()["NotebookInstances"]),
    }
    Path("reports/usage.json").write_text(json.dumps(usage, indent=2) + "\n")
    print(json.dumps(usage, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", required=True, type=datetime.fromisoformat)
    args = parser.parse_args()
    main(args.since.replace(tzinfo=UTC))
