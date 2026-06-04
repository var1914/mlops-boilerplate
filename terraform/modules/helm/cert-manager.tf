# ---------------------------------------------------------------------------
# cert-manager
# ---------------------------------------------------------------------------

resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  version          = "v1.17.2"
  namespace        = "cert-manager"
  create_namespace = false
  wait             = true
  timeout          = 600

  values = [file("${path.module}/values/cert-manager.yaml")]

  set {
    name  = "crds.enabled"
    value = "true"
  }

  depends_on = [kubernetes_namespace.cert_manager]
}

# Allow cert-manager webhooks and CRDs to become ready before ClusterIssuer manifests apply.
resource "time_sleep" "wait_for_cert_manager" {
  depends_on      = [helm_release.cert_manager]
  create_duration = "30s"
}
