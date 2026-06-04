resource "kubernetes_namespace" "app" {
  for_each = local.app_namespaces

  metadata {
    name = each.key
    labels = merge(
      local.common_labels,
      lookup(each.value, "labels", {}),
      each.value.istio_injection ? { "istio-injection" = "enabled" } : {},
    )
  }
}
