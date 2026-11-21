# wx-blend-sagemaker

A learned blend of weather models for next-day maximum temperature at three New Zealand places,
trained, evaluated and registered by an AWS SageMaker Pipeline.

[wx](https://wx.rejusamjohn.workers.dev) shows eight weather models side by side and uses their
median as the headline. This project asks whether a blend learned from past forecasts does better,
and puts that question through the steps a production model goes through: a repeatable pipeline,
a fixed evaluation gate, a model registry, and a deploy step that only runs for an approved model.

The same loop, built by hand with GitHub Actions, MLflow and AWS Lambda, is in
[student-success-early-warning](https://github.com/rejusam/student-success-early-warning).
This repo does it on a managed platform.

## The pipeline

```
fetch.py (laptop or GitHub Actions)  ->  S3 raw/<date>/
                                              |
SageMaker Pipeline "wx-blend"                 v
  Prepare   (processing)   join forecasts and actuals, data checks, last 90 days held back
  Train     (training)     one Ridge regression per place -> model.json
  Evaluate  (processing)   MAE of each model, two baselines and the blend on the 90 test days
  CheckGate                blend must beat the 45-day inverse-MAE baseline by 2% overall,
                           and must not be worse than it at any place
    passed -> RegisterApproved   (Model Registry group wx-blend)
    failed -> RegisterRejected   (same group; kept with its evaluation)
                                              |
export.py  ->  reports/ (evidence)  and blend-weights.json for wx:
               the blend if Approved, otherwise the champion (45-day inverse-MAE weights)
```

| File | What it does |
|---|---|
| `scripts/fetch.py` | Downloads past forecasts and actuals from Open-Meteo, saves them unchanged |
| `steps/prepare.py` | Daily maximum per model, model coverage rule, data checks, time split |
| `steps/train.py` | Fits the blend; the result is a few numbers per place |
| `steps/evaluate.py` | Baselines, MAE, the gate |
| `scripts/pipeline.py` | Defines, starts and records the pipeline |
| `scripts/export.py` | Copies a version's evaluation into `reports/`; writes the weights wx should serve |
| `scripts/usage.py` | Records SageMaker jobs, billed minutes and anything still running |
| `infra/` | Terraform: bucket, two least-privilege roles, model group, budget with an automatic stop |

The step scripts import nothing from each other, because SageMaker copies each one into its own
container.

## Data

- Forecasts: each model's hourly 2 m temperature as forecast one day ahead, from the Open-Meteo
  [Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api). The daily maximum is taken
  per local date.
- Actuals: daily maximum from the Open-Meteo
  [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). This is a
  reanalysis (a best estimate built from many observations), not a reading from one weather station.
- Places: Auckland, Wellington and Tara Hills, at the coordinates wx already uses for its station
  checks.
- Range used (`reports/data_check_v1.json`): 2024-02-05 to 2026-11-14, 3,003 place-days.
  Six models kept. UKMO and CMA were dropped: their forecasts were present on 60.1% and 91.2% of
  days, below the 95% rule.

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), used under its free
non-commercial terms.

## Result

The first honest run was **rejected by the gate**. Mean absolute error in °C on the 90 held-back
days, 2026-08-17 to 2026-11-14 (`reports/evaluation_v1.json`, model package version 1):

| | Auckland | Tara Hills | Wellington | All three |
|---|---|---|---|---|
| Plain mean of the six models | 0.649 | 0.785 | 0.563 | 0.665 |
| 45-day inverse-MAE weights (baseline) | 0.637 | 0.703 | **0.545** | 0.628 |
| Ridge blend | **0.559** | **0.599** | 0.567 | **0.575** |

Over the three places the blend's error is lower than the baseline's (0.575 against 0.628), but at
Wellington it is higher (0.567 against 0.545). The gate requires both, so the version is registered as Rejected
and nothing is deployed to wx. The gate and its margin were fixed before the first run and were
not changed afterwards.

The best single model on these days was ECMWF IFS (0.765 overall). The baseline is the weighting
described by Pelmorex in a
[multi-model bias correction poster](https://ams.confex.com/ams/93Annual/webprogram/Handout/Paper217249/AMSPoster_36x60.pdf).

The comparison leans towards the baseline, so the rejection is a cautious verdict: the baseline's
weights use the previous day's actual temperature and keep updating through the 90 test days,
while the blend's weights are fixed at the end of training. The baseline's 45-day window counts
rows, so where days were dropped for missing data it reaches a little further back.

Where each version came from:

- Version 1 was registered by run `zm6u4zpawi9b`, started from a laptop. Its register step created
  the package and then failed on a missing permission (since added), so that run is recorded as
  Failed. The run id is in the evaluation's S3 path, saved as `evaluation_s3_uri` in
  `reports/evaluation_v1.json`.
- Versions 2 and 3 were registered by runs `i78em7siisup` and `npcom23j0h1j`, started by GitHub
  Actions: their records in `reports/pipeline_runs.json` have `started_by` set to the
  `wx-blend-github` role. Each fetched the data again, and their evaluations
  (`reports/evaluation_v2.json`, `reports/evaluation_v3.json`) are identical to version 1's.

Every pipeline run, including the failed ones while the pipeline was being set up, is in
`reports/pipeline_runs.json`.

## What is live

The blend has not passed the gate, so the baseline stays the champion, and that is what is
deployed. Each run's Evaluate step also computes the baseline's current weights per place from
the last 45 days, and `export.py` ships those when the version is Rejected. wx serves them for
tomorrow behind `?blend=1`, within 25 km of the three places:

```
curl 'wx.rejusamjohn.workers.dev/auckland?blend=1'
curl 'wx.rejusamjohn.workers.dev/auckland?format=json&blend=1'
```

`reports/wx_blend_check.json` is a saved live response: model version 4, weights as of
2026-11-14. When a future blend passes the gate, the same path ships it instead.

## How it runs

Train runs as a SageMaker training job by default. The AWS account this was built in had a
training-job quota of 0 for every instance type when this was written (an increase is requested),
so the recorded runs used `--train-in-processing`: the same `train.py` run as a processing job.
For every run that reached the Train step, `reports/pipeline_runs.json` records which one it used.
The account id in the saved reports is replaced with `<account-id>`.

```bash
uv sync
uv run pytest

# AWS: short-lived administrator credentials, region ap-southeast-2
export WX_AWS_ACCOUNT=<account>   # 12-digit account id; not stored in the repository
terraform -chdir=infra init && terraform -chdir=infra apply
uv run python scripts/fetch.py --out build/raw --s3
uv run python scripts/pipeline.py --run --raw s3://wx-blend-<account>/raw/<date>/ [--train-in-processing]
uv run python scripts/export.py --version <N>
```

Or from GitHub: Actions, `run-pipeline`, Run workflow. The job signs in to AWS with OpenID Connect
from the `sagemaker` environment; no AWS keys are stored in GitHub.

## Seeing it in AWS

SageMaker Studio (domain `wx-blend`, user `reju`, defined in `infra/studio.tf`) shows the pipeline
graph with each run's steps, and the Model Registry group `wx-blend` with every version and its
evaluation. Studio's role is read-only and may not create apps or spaces.

## Cost controls

- No endpoints or notebooks: nothing bills by the hour. Each job stops when its script ends, and
  after 30 minutes at most.
- Processing on `ml.t3.medium`, training on `ml.m5.large`. The pipeline role is denied any other
  instance type. No GPU.
- The gate margin cannot be set below 0.02 (`steps/evaluate.py`), so a run cannot approve a model
  by loosening the gate.
- AWS Budget `wx-blend` for SageMaker: email alerts at USD 2 and USD 5 a month, and at USD 10 a
  budget action attaches a deny policy to both roles so no new job or run can start.
- The SDK's usage telemetry is turned off (`scripts/sagemaker.yaml`).
- `scripts/usage.py` writes `reports/usage.json`: jobs run, billed minutes, and whether anything is
  still running. Cost Explorer had no data yet for this new account, so usage is recorded in
  minutes rather than dollars.

## Limits and next steps

- Actuals are reanalysis, not station readings. wx's forecast ladder records station observations
  for these three places; scoring against those is the next step.
- 90 test days is one season. A rolling evaluation over several seasons would say more.
- Next: a weekly schedule, daily scoring of the live blend against station data, and a retrain
  trigger when its 7-day error is worse than the baseline.
- The SageMaker Python SDK is pinned at 2.257.7 (the v2 line). Moving to v3 is a later task.

## Licence

MIT. See `LICENSE`.
