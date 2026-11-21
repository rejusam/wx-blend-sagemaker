data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  arn_prefix = "arn:aws:sagemaker:${var.region}:${local.account_id}"
}

# --- Bucket: raw data, the clean table, models and evaluation reports ---------------------

resource "aws_s3_bucket" "data" {
  bucket = "wx-blend-${local.account_id}"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

data "aws_iam_policy_document" "tls_only" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "data" {
  bucket = aws_s3_bucket.data.id
  policy = data.aws_iam_policy_document.tls_only.json
}

# --- Model Registry group: every trained version lands here, Approved or Rejected --------

resource "aws_sagemaker_model_package_group" "blend" {
  model_package_group_name        = "wx-blend"
  model_package_group_description = "Ridge blend of weather models, next-day maximum temperature"
}

# --- Role the pipeline and its jobs run as ------------------------------------------------

data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "wx-blend-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "sagemaker" {
  statement {
    sid       = "Bucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    sid = "Jobs"
    actions = [
      "sagemaker:CreateProcessingJob", "sagemaker:DescribeProcessingJob", "sagemaker:StopProcessingJob",
      "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob", "sagemaker:StopTrainingJob",
      "sagemaker:AddTags",
    ]
    resources = ["${local.arn_prefix}:processing-job/*", "${local.arn_prefix}:training-job/*"]
  }
  statement {
    sid       = "OnlySmallInstances"
    effect    = "Deny"
    actions   = ["sagemaker:CreateProcessingJob", "sagemaker:CreateTrainingJob"]
    resources = ["*"]
    condition {
      test     = "ForAnyValue:StringNotEquals"
      variable = "sagemaker:InstanceTypes"
      values   = ["ml.t3.medium", "ml.m5.large"]
    }
  }
  statement {
    sid       = "Registry"
    actions   = ["sagemaker:CreateModelPackage", "sagemaker:DescribeModelPackage", "sagemaker:CreateModelPackageGroup", "sagemaker:DescribeModelPackageGroup", "sagemaker:AddTags"]
    resources = [aws_sagemaker_model_package_group.blend.arn, "${local.arn_prefix}:model-package/wx-blend/*"]
  }
  statement {
    sid       = "PassItselfToJobs"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
  statement {
    sid       = "PullAwsContainers"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # this action has no resource to limit it to
  }
  statement {
    sid     = "PullScikitLearnImage"
    actions = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    # AWS account that publishes the scikit-learn images in ap-southeast-2 (SageMaker SDK image_uri_config/sklearn.json).
    resources = ["arn:aws:ecr:${var.region}:783357654285:repository/sagemaker-scikit-learn"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/sagemaker/*"]
  }
  statement {
    sid       = "JobMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # this action has no resource to limit it to
    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  name   = "wx-blend-sagemaker"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sagemaker.json
}

# --- Role GitHub Actions assumes through OIDC (no stored keys) -----------------------------

# The OIDC provider already exists in this account (created by student-success-early-warning).
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${var.github_environment}"]
    }
  }
}

resource "aws_iam_role" "github" {
  name                 = "wx-blend-github"
  assume_role_policy   = data.aws_iam_policy_document.github_trust.json
  max_session_duration = 3600
}

data "aws_iam_policy_document" "github" {
  statement {
    sid       = "Bucket"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    sid = "Pipeline"
    actions = [
      "sagemaker:CreatePipeline", "sagemaker:UpdatePipeline", "sagemaker:DescribePipeline",
      "sagemaker:StartPipelineExecution", "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps", "sagemaker:ListPipelineParametersForExecution",
      "sagemaker:AddTags",
    ]
    resources = ["${local.arn_prefix}:pipeline/wx-blend", "${local.arn_prefix}:pipeline/wx-blend/*"]
  }
  statement {
    sid       = "ReadRegistry"
    actions   = ["sagemaker:DescribeModelPackage"]
    resources = ["${local.arn_prefix}:model-package/wx-blend/*"]
  }
  statement {
    sid       = "ListRegistry"
    actions   = ["sagemaker:ListModelPackages"]
    resources = ["*"] # this action has no resource to limit it to
  }
  statement {
    sid       = "PassPipelineRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "github" {
  name   = "wx-blend-github"
  role   = aws_iam_role.github.id
  policy = data.aws_iam_policy_document.github.json
}

# --- Budget: email alerts, then an automatic stop -----------------------------------------
# AWS has no hard spending cap. These alert at USD 2 and USD 5 of SageMaker spend in a month,
# and at USD 10 a budget action attaches a deny policy to both wx roles, so no new job or
# pipeline run can start until a person detaches it. Billing data lags by some hours.

resource "aws_budgets_budget" "sagemaker" {
  name         = "wx-blend"
  budget_type  = "COST"
  limit_amount = "10"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon SageMaker"]
  }

  dynamic "notification" {
    for_each = [2, 5]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "ABSOLUTE_VALUE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.budget_email]
    }
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}

data "aws_iam_policy_document" "stop_jobs" {
  statement {
    effect = "Deny"
    actions = [
      "sagemaker:CreateProcessingJob", "sagemaker:CreateTrainingJob",
      "sagemaker:StartPipelineExecution", "sagemaker:CreateEndpoint", "sagemaker:CreateApp",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "stop_jobs" {
  name   = "wx-blend-stop-jobs"
  policy = data.aws_iam_policy_document.stop_jobs.json
}

data "aws_iam_policy_document" "budgets_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["budgets.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "budget_action" {
  name               = "wx-blend-budget-action"
  assume_role_policy = data.aws_iam_policy_document.budgets_trust.json
}

data "aws_iam_policy_document" "budget_action" {
  statement {
    actions   = ["iam:AttachRolePolicy", "iam:DetachRolePolicy"]
    resources = [aws_iam_role.sagemaker.arn, aws_iam_role.github.arn]
    condition {
      test     = "ArnEquals"
      variable = "iam:PolicyARN"
      values   = [aws_iam_policy.stop_jobs.arn]
    }
  }
}

resource "aws_iam_role_policy" "budget_action" {
  name   = "wx-blend-budget-action"
  role   = aws_iam_role.budget_action.id
  policy = data.aws_iam_policy_document.budget_action.json
}

resource "aws_budgets_budget_action" "stop_jobs" {
  budget_name        = aws_budgets_budget.sagemaker.name
  action_type        = "APPLY_IAM_POLICY"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"
  execution_role_arn = aws_iam_role.budget_action.arn

  action_threshold {
    action_threshold_type  = "ABSOLUTE_VALUE"
    action_threshold_value = 10
  }

  definition {
    iam_action_definition {
      policy_arn = aws_iam_policy.stop_jobs.arn
      roles      = [aws_iam_role.sagemaker.name, aws_iam_role.github.name]
    }
  }

  subscriber {
    address           = var.budget_email
    subscription_type = "EMAIL"
  }
}
