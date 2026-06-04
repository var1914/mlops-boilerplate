# ---------------------------------------------------------------------------
# istio-base
# ---------------------------------------------------------------------------

resource "helm_release" "istio_base" {
  name             = "istio-base"
  repository       = "https://istio-release.storage.googleapis.com/charts"
  chart            = "base"
  version          = "1.25.2"
  namespace        = "istio-system"
  create_namespace = false

  values = [file("${path.module}/values/istio-base.yaml")]

  depends_on = [kubernetes_namespace.istio_system]
}

# ---------------------------------------------------------------------------
# istiod
# ---------------------------------------------------------------------------

resource "helm_release" "istiod" {
  name       = "istiod"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "istiod"
  version    = "1.25.2"
  namespace  = "istio-system"

  values = [file("${path.module}/values/istiod.yaml")]

  depends_on = [helm_release.istio_base]
}

# ---------------------------------------------------------------------------
# istio-gateway
# ---------------------------------------------------------------------------

resource "helm_release" "istio_gateway" {
  name       = "istio-ingressgateway"
  repository = "https://istio-release.storage.googleapis.com/charts"
  chart      = "gateway"
  version    = "1.25.2"
  namespace  = "istio-system"

  values = [file("${path.module}/values/istio-gateway.yaml")]

  depends_on = [helm_release.istiod]
}
