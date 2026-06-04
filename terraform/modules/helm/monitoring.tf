# ---------------------------------------------------------------------------
# kube-prometheus-stack
# ---------------------------------------------------------------------------

resource "helm_release" "kube_prometheus" {
  name             = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "72.6.2"
  namespace        = "monitoring"
  create_namespace = false

  values = [file("${path.module}/values/kube-prometheus.yaml")]

  depends_on = [helm_release.istiod, kubernetes_namespace.monitoring]
}

# ---------------------------------------------------------------------------
# Loki
# ---------------------------------------------------------------------------

resource "helm_release" "loki" {
  name       = "loki"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "loki-stack"
  version    = "2.10.2"
  namespace  = "monitoring"

  values = [file("${path.module}/values/loki.yaml")]

  depends_on = [helm_release.kube_prometheus]
}

# ---------------------------------------------------------------------------
# Kiali
# ---------------------------------------------------------------------------

resource "helm_release" "kiali" {
  name       = "kiali"
  repository = "https://kiali.org/helm-charts"
  chart      = "kiali-server"
  version    = "2.8.0"
  namespace  = "istio-system"

  values = [templatefile("${path.module}/values/kiali.yaml", {
    prometheus_url = "http://kube-prometheus-stack-prometheus.monitoring:9090"
    grafana_url    = "http://kube-prometheus-stack-grafana.monitoring:80"
  })]

  depends_on = [helm_release.istiod, helm_release.kube_prometheus]
}
