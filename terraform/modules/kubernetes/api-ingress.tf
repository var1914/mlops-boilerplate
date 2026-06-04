resource "kubernetes_manifest" "api_virtualservice" {
  count = var.ingress_api_host != "" ? 1 : 0

  manifest = yamldecode(templatefile(
    "${path.module}/manifests/api-virtualservice.yaml.tpl",
    {
      ingress_api_host = var.ingress_api_host
    },
  ))

  depends_on = [kubernetes_manifest.mlops_gateway]
}
