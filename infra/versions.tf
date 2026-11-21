# Applied by hand, by a person signed in with short-lived administrator credentials.
# State is a local file that is never committed (see .gitignore).
terraform {
  required_version = ">= 1.10"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.70.0"
    }
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      project    = "wx-blend-sagemaker"
      managed_by = "terraform"
    }
  }
}
