# Additional S3 buckets for crypto ML pipeline (replaces MinIO buckets on EKS)

resource "aws_s3_bucket" "crypto_features" {
  bucket = local.crypto_features_bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket" "crypto_models" {
  bucket = local.crypto_models_bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket" "crypto_data_versions" {
  bucket = local.crypto_data_versions_bucket_name
  tags   = var.tags
}

resource "aws_s3_bucket" "airflow_logs" {
  bucket = local.airflow_logs_bucket_name
  tags   = var.tags
}

locals {
  crypto_bucket_configs = {
    crypto_features      = aws_s3_bucket.crypto_features
    crypto_models        = aws_s3_bucket.crypto_models
    crypto_data_versions = aws_s3_bucket.crypto_data_versions
    airflow_logs         = aws_s3_bucket.airflow_logs
  }
}

resource "aws_s3_bucket_versioning" "crypto_buckets" {
  for_each = local.crypto_bucket_configs

  bucket = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "crypto_buckets" {
  for_each = local.crypto_bucket_configs

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "crypto_buckets" {
  for_each = local.crypto_bucket_configs

  bucket = each.value.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "airflow_logs" {
  bucket = aws_s3_bucket.airflow_logs.id

  rule {
    id     = "expire-old-logs"
    status = "Enabled"

    expiration {
      days = 90
    }
  }
}
