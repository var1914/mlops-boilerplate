locals {
  cluster_name = "${var.project_name}-${var.environment}-eks"

  common_tags = merge(var.tags, {
    Module = "eks"
  })
}
