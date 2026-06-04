# ---------------------------------------------------------------------------
# Prometheus Pushgateway (batch metrics from Airflow ETL / training tasks)
# ---------------------------------------------------------------------------

resource "helm_release" "pushgateway" {
  name             = "prometheus-pushgateway"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "prometheus-pushgateway"
  version          = "3.1.0"
  namespace        = "monitoring"
  create_namespace = false

  values = [file("${path.module}/values/pushgateway.yaml")]

  depends_on = [kubernetes_namespace.monitoring]
}
