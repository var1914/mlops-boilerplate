locals {
  common_labels = merge(var.tags, {
    "app.kubernetes.io/managed-by" = "terraform"
  })
}
