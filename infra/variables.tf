variable "region" {
  type    = string
  default = "ap-southeast-2"
}

variable "budget_email" {
  type        = string
  description = "Where the budget alert is sent."
}

variable "github_repository" {
  type        = string
  description = "Repository part of the GitHub OIDC subject claim. This account uses immutable subjects: owner@owner_id/name@repo_id."
}

variable "github_environment" {
  type    = string
  default = "sagemaker"
}
