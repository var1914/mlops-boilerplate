resource "aws_elasticache_subnet_group" "this" {
  name       = "${local.name_prefix}-redis"
  subnet_ids = var.database_subnet_ids
  tags       = local.common_tags
}

resource "aws_elasticache_cluster" "this" {
  cluster_id      = "${local.name_prefix}-redis"
  engine          = "redis"
  engine_version  = var.engine_version
  node_type       = var.node_type
  num_cache_nodes = var.num_cache_nodes

  subnet_group_name  = aws_elasticache_subnet_group.this.name
  security_group_ids = [aws_security_group.redis.id]

  parameter_group_name = "default.redis7"

  tags = local.common_tags
}
