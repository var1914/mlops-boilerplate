locals {
  mlflow_bucket_name            = "${var.project_name}-mlflow-artifacts-${var.environment}"
  raw_data_bucket_name          = "${var.project_name}-crypto-raw-data-${var.environment}"
  crypto_features_bucket_name   = "${var.project_name}-crypto-features-${var.environment}"
  crypto_models_bucket_name     = "${var.project_name}-crypto-models-${var.environment}"
  crypto_data_versions_bucket_name = "${var.project_name}-crypto-data-versions-${var.environment}"
  airflow_logs_bucket_name      = "${var.project_name}-airflow-logs-${var.environment}"

  common_tags = merge(var.tags, {
    Module = "s3"
  })
}
