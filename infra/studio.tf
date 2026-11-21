# --- SageMaker Studio: a place to look at the pipeline graph and the Model Registry ---------
# The domain itself is free; it creates a small storage volume. Studio's role can read
# SageMaker and this bucket but cannot create apps or spaces, so nothing billed by the hour
# (such as a JupyterLab instance) can be started from Studio.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_iam_role" "studio" {
  name               = "wx-blend-studio"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

resource "aws_iam_role_policy_attachment" "studio_read" {
  role       = aws_iam_role.studio.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerReadOnly"
}

data "aws_iam_policy_document" "studio" {
  statement {
    sid       = "ReadBucket"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    sid       = "NoBilledApps"
    effect    = "Deny"
    actions   = ["sagemaker:CreateApp", "sagemaker:CreateSpace"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "studio" {
  name   = "wx-blend-studio"
  role   = aws_iam_role.studio.id
  policy = data.aws_iam_policy_document.studio.json
}

resource "aws_sagemaker_domain" "studio" {
  domain_name             = "wx-blend"
  auth_mode               = "IAM"
  vpc_id                  = data.aws_vpc.default.id
  subnet_ids              = data.aws_subnets.default.ids
  app_network_access_type = "PublicInternetOnly"

  default_user_settings {
    execution_role    = aws_iam_role.studio.arn
    studio_web_portal = "ENABLED"
  }

  retention_policy {
    home_efs_file_system = "Delete"
  }
}

resource "aws_sagemaker_user_profile" "reju" {
  domain_id         = aws_sagemaker_domain.studio.id
  user_profile_name = "reju"
}
