locals {
  name_prefix = "${var.project_name}-${var.environment}"

  nat_gateway_count = var.single_nat_gateway ? 1 : length(var.availability_zones)

  common_tags = merge(var.tags, {
    Module = "vpc"
  })
}
