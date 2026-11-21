output "bucket" {
  value = aws_s3_bucket.data.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}

output "github_role_arn" {
  value = aws_iam_role.github.arn
}
