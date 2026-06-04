locals {
  common_labels = merge(var.tags, {
    "app.kubernetes.io/managed-by" = "terraform"
  })

  app_namespaces = {
    for ns in var.app_namespaces : ns.name => ns
  }
}
