locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = merge(var.tags, {
    Module = "iam"
  })

  irsa_roles = {
    mlflow = {
      name            = "${var.project_name}-${var.environment}-mlflow"
      namespace       = var.mlflow_namespace
      service_account = var.mlflow_service_account
    }
    airflow = {
      name            = "${var.project_name}-${var.environment}-airflow"
      namespace       = var.airflow_namespace
      service_account = var.airflow_service_account
    }
    api = {
      name            = "${var.project_name}-${var.environment}-api"
      namespace       = var.api_namespace
      service_account = var.api_service_account
    }
  }
}
