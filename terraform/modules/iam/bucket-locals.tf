locals {
  data_bucket_arns = compact([
    var.mlflow_artifacts_bucket_arn,
    var.raw_data_bucket_arn,
    var.crypto_features_bucket_arn,
    var.crypto_models_bucket_arn,
    var.crypto_data_versions_bucket_arn,
  ])

  data_object_arns = [for arn in local.data_bucket_arns : "${arn}/*"]

  airflow_bucket_arns = compact(concat(
    local.data_bucket_arns,
    [var.airflow_logs_bucket_arn],
  ))

  airflow_object_arns = [for arn in local.airflow_bucket_arns : "${arn}/*"]
}
