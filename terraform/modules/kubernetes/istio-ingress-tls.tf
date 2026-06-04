# TLS ingress for the inference API (optional UI hostname).
# DNS: point ingress_api_host at the istio-ingressgateway LoadBalancer.

locals {
  ingress_hosts = compact([
    var.ingress_ui_host,
    var.ingress_api_host,
  ])
  cluster_issuer_name    = var.acme_use_staging ? "letsencrypt-staging" : "letsencrypt-prod"
  acme_server            = var.acme_use_staging ? "https://acme-staging-v02.api.letsencrypt.org/directory" : "https://acme-v02.api.letsencrypt.org/directory"
  istio_system_namespace = "istio-system"
  gateway_name           = "istio-ingressgateway"
  tls_secret_name        = "mlops-boilerplate-tls"
  api_service_host       = "crypto-prediction-api.${var.api_service_namespace}.svc.cluster.local"

  manifest_template_vars = {
    cluster_issuer_name    = local.cluster_issuer_name
    acme_server            = local.acme_server
    acme_email             = var.acme_email
    ingress_hosts          = local.ingress_hosts
    istio_system_namespace = local.istio_system_namespace
    gateway_name           = local.gateway_name
    tls_secret_name        = local.tls_secret_name
  }

  api_ingress_template_vars = {
    ingress_api_host       = var.ingress_api_host
    istio_system_namespace = local.istio_system_namespace
    gateway_name           = local.gateway_name
    api_namespace          = var.api_service_namespace
    api_service_host       = local.api_service_host
    api_service_port       = var.api_service_port
  }
}

# Brief pause after Helm (cert-manager CRDs/webhooks) before applying cert-manager API objects.
resource "time_sleep" "wait_for_cert_manager" {
  create_duration = "45s"
}

resource "kubernetes_manifest" "cluster_issuer" {
  manifest = yamldecode(templatefile(
    "${path.module}/manifests/cluster-issuer.yaml.tpl",
    local.manifest_template_vars,
  ))

  depends_on = [time_sleep.wait_for_cert_manager]
}

resource "kubernetes_manifest" "mlops_certificate" {
  manifest = yamldecode(templatefile(
    "${path.module}/manifests/certificate.yaml.tpl",
    local.manifest_template_vars,
  ))

  depends_on = [kubernetes_manifest.cluster_issuer]
}

resource "kubernetes_manifest" "mlops_gateway" {
  manifest = yamldecode(templatefile(
    "${path.module}/manifests/gateway.yaml.tpl",
    local.manifest_template_vars,
  ))
}

# Public API routing: Gateway TLS termination → crypto-prediction-api Service.
resource "kubernetes_manifest" "api_virtual_service" {
  count = var.ingress_api_host != "" ? 1 : 0

  manifest = yamldecode(templatefile(
    "${path.module}/manifests/api-virtualservice.yaml.tpl",
    local.api_ingress_template_vars,
  ))

  depends_on = [kubernetes_manifest.mlops_gateway]
}
