resource "kubernetes_secret" "airflow_rds" {
  metadata {
    name      = "airflow-rds-credentials"
    namespace = "airflow"
  }

  type = "Opaque"

  string_data = {
    connection = "postgresql://${var.rds_username}:${var.rds_password}@${var.rds_endpoint}:${var.rds_port}/airflow"
  }
}

resource "kubernetes_secret" "mlflow_rds" {
  metadata {
    name      = "mlflow-rds-credentials"
    namespace = "mlflow"
  }

  type = "Opaque"

  string_data = {
    password = var.rds_password
    host     = var.rds_endpoint
    port     = tostring(var.rds_port)
    username = var.rds_username
    database = var.rds_db_name
  }
}

resource "kubernetes_secret" "app_rds" {
  for_each = local.app_namespaces

  metadata {
    name      = "ml-pipeline-secrets"
    namespace = kubernetes_namespace.app[each.key].metadata[0].name
  }

  type = "Opaque"

  string_data = {
    DB_PASSWORD = var.rds_password
  }
}

resource "kubernetes_config_map" "ml_pipeline_platform" {
  for_each = local.app_namespaces

  metadata {
    name      = "ml-pipeline-config"
    namespace = kubernetes_namespace.app[each.key].metadata[0].name
    labels = {
      app       = "crypto-prediction-api"
      component = "config"
    }
  }

  data = {
    ENVIRONMENT       = coalesce(var.api_runtime_environment, var.environment)
    DEBUG             = "false"
    DB_HOST           = var.rds_endpoint
    DB_PORT           = tostring(var.rds_port)
    DB_NAME           = "crypto"
    DB_USER           = var.rds_username
    REDIS_HOST        = var.redis_endpoint
    REDIS_PORT        = tostring(var.redis_port)
    REDIS_DB          = "0"
    MINIO_ENDPOINT    = "s3.${var.region}.amazonaws.com"
    MINIO_SECURE      = "true"
    MINIO_BUCKET_MODELS = var.crypto_models_bucket
    MINIO_BUCKET_FEATURES = var.crypto_features_bucket
    MINIO_BUCKET_DATA_VERSIONS = var.crypto_data_versions_bucket
    MLFLOW_TRACKING_URI = "http://mlflow.mlflow.svc.cluster.local:5000"
    MLFLOW_EXPERIMENT_PREFIX = "crypto_multi_models"
    MLFLOW_ARTIFACT_ROOT = "s3://${var.mlflow_artifacts_bucket}"
    AWS_DEFAULT_REGION = var.region
    API_HOST           = "0.0.0.0"
    API_PORT           = "8000"
    API_WORKERS        = "4"
    API_LOG_LEVEL      = "info"
    MONITORING_ENABLE_METRICS = "true"
  }
}

resource "kubernetes_service_account" "api" {
  for_each = local.app_namespaces

  metadata {
    name      = "crypto-prediction-api"
    namespace = kubernetes_namespace.app[each.key].metadata[0].name
    annotations = var.api_irsa_role_arn != "" ? {
      "eks.amazonaws.com/role-arn" = var.api_irsa_role_arn
    } : {}
  }
}
