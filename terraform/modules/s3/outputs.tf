output "mlflow_artifacts_bucket_name" {
  description = "Name of the MLflow artifacts S3 bucket"
  value       = aws_s3_bucket.mlflow_artifacts.id
}

output "mlflow_artifacts_bucket_arn" {
  description = "ARN of the MLflow artifacts S3 bucket"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "raw_data_bucket_name" {
  description = "Name of the crypto raw data S3 bucket"
  value       = aws_s3_bucket.raw_data.id
}

output "raw_data_bucket_arn" {
  description = "ARN of the crypto raw data S3 bucket"
  value       = aws_s3_bucket.raw_data.arn
}

output "crypto_features_bucket_name" {
  value = aws_s3_bucket.crypto_features.id
}

output "crypto_features_bucket_arn" {
  value = aws_s3_bucket.crypto_features.arn
}

output "crypto_models_bucket_name" {
  value = aws_s3_bucket.crypto_models.id
}

output "crypto_models_bucket_arn" {
  value = aws_s3_bucket.crypto_models.arn
}

output "crypto_data_versions_bucket_name" {
  value = aws_s3_bucket.crypto_data_versions.id
}

output "crypto_data_versions_bucket_arn" {
  value = aws_s3_bucket.crypto_data_versions.arn
}

output "airflow_logs_bucket_name" {
  value = aws_s3_bucket.airflow_logs.id
}

output "airflow_logs_bucket_arn" {
  value = aws_s3_bucket.airflow_logs.arn
}

output "all_bucket_arns" {
  description = "All pipeline bucket ARNs (for IAM policies)"
  value = [
    aws_s3_bucket.mlflow_artifacts.arn,
    aws_s3_bucket.raw_data.arn,
    aws_s3_bucket.crypto_features.arn,
    aws_s3_bucket.crypto_models.arn,
    aws_s3_bucket.crypto_data_versions.arn,
    aws_s3_bucket.airflow_logs.arn,
  ]
}
