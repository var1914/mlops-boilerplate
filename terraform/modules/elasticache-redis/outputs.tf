output "redis_endpoint" {
  description = "Redis primary endpoint address"
  value       = aws_elasticache_cluster.this.cache_nodes[0].address
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_cluster.this.cache_nodes[0].port
}

output "redis_security_group_id" {
  description = "Security group ID attached to the Redis cluster"
  value       = aws_security_group.redis.id
}
