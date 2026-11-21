"""Fixed settings shared by the local scripts (fetch, pipeline, export)."""

import os

# The three places wx's forecast ladder already records, with the same coordinates
# (wx repo, sql/ladder-schema.sql), so later checks against station data line up.
PLACES = {
    "auckland": (-37.0, 174.8),
    "wellington": (-41.333, 174.8),
    "tara-hills": (-44.517, 169.9),
}

# wx's eight models (wx repo, src/models.ts). prepare.py drops any with gaps in their history.
MODELS = [
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "ukmo_seamless",
    "jma_seamless",
    "gem_seamless",
    "meteofrance_seamless",
    "cma_grapes_global",
]

TIMEZONE = "Pacific/Auckland"
FIRST_DAY = "2024-01-01"  # the Previous Runs API has data from January 2024

REGION = "ap-southeast-2"
# The AWS account id is not kept in the repository. Set WX_AWS_ACCOUNT before any script
# that talks to AWS; the run-pipeline workflow reads it from a repository variable.
ACCOUNT = os.environ.get("WX_AWS_ACCOUNT", "")
BUCKET = f"wx-blend-{ACCOUNT}"
PIPELINE_NAME = "wx-blend"
MODEL_GROUP = "wx-blend"


def require_account() -> None:
    if not ACCOUNT.isdigit() or len(ACCOUNT) != 12:
        raise SystemExit("Set WX_AWS_ACCOUNT to the 12-digit AWS account id first.")
