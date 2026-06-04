# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

resource "helm_release" "mlflow" {
  name             = "mlflow"
  repository       = "https://community-charts.github.io/helm-charts"
  chart            = "mlflow"
  version          = "0.14.0"
  namespace        = "mlflow"
  create_namespace = false

  values = [templatefile("${path.module}/values/mlflow.yaml", {
    rds_endpoint            = var.rds_endpoint
    rds_port                = var.rds_port
    rds_db_name             = var.rds_db_name
    rds_username            = var.rds_username
    rds_secret_arn          = var.rds_secret_arn
    mlflow_role_arn         = var.mlflow_role_arn
    mlflow_artifacts_bucket = var.mlflow_artifacts_bucket
    region                  = var.region
  })]

  set_sensitive {
    name  = "backendStore.postgres.password"
    value = var.rds_password
  }

  depends_on = [helm_release.istiod, kubernetes_namespace.mlflow]
}
