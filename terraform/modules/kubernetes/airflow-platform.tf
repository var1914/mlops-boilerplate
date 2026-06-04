resource "random_password" "airflow_fernet" {
  length  = 32
  special = false
}

resource "random_password" "airflow_jwt" {
  length  = 32
  special = false
}

resource "random_password" "airflow_internal_api" {
  length  = 32
  special = false
}

resource "random_password" "airflow_webserver_secret" {
  length  = 32
  special = false
}

resource "kubernetes_secret" "airflow_runtime" {
  metadata {
    name      = "airflow-runtime-secrets"
    namespace = "airflow"
  }

  type = "Opaque"

  string_data = {
    AIRFLOW__CORE__FERNET_KEY              = random_password.airflow_fernet.result
    AIRFLOW__API_AUTH__JWT_SECRET          = random_password.airflow_jwt.result
    AIRFLOW__CORE__INTERNAL_API_SECRET_KEY = random_password.airflow_internal_api.result
    AIRFLOW__WEBSERVER__SECRET_KEY         = random_password.airflow_webserver_secret.result
  }
}

resource "kubernetes_config_map" "airflow_platform" {
  metadata {
    name      = "airflow-platform-config"
    namespace = "airflow"
  }

  data = {
    AWS_DEFAULT_REGION                          = var.region
    REDIS_HOST                                  = var.redis_endpoint
    REDIS_PORT                                  = tostring(var.redis_port)
    MLFLOW_TRACKING_URI                         = "http://mlflow.mlflow.svc.cluster.local:5000"
    MLFLOW_S3_ARTIFACT_BUCKET                   = var.mlflow_artifacts_bucket
    S3_DATA_BUCKET                              = var.raw_data_bucket
    MINIO_BUCKET_FEATURES                       = var.crypto_features_bucket
    MINIO_BUCKET_MODELS                         = var.crypto_models_bucket
    MINIO_BUCKET_DATA_VERSIONS                  = var.crypto_data_versions_bucket
    PROMETHEUS_PUSHGATEWAY_URL                  = "http://prometheus-pushgateway.monitoring.svc.cluster.local:9091"
    AIRFLOW__LOGGING__REMOTE_LOGGING            = "True"
    AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER    = "s3://${var.airflow_logs_bucket}/logs"
    AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID        = "aws_default"
    AIRFLOW__LOGGING__DELETE_LOCAL_LOGS         = "True"
    AIRFLOW__LOGGING__ENCRYPT_S3_LOGS           = "False"
  }
}
