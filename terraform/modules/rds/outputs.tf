output "db_endpoint" {
  description = "RDS instance endpoint address (without port)"
  value       = aws_db_instance.this.address
}

output "db_port" {
  description = "RDS instance port"
  value       = aws_db_instance.this.port
}

output "db_name" {
  description = "Name of the default database"
  value       = aws_db_instance.this.db_name
}

output "db_username" {
  description = "Master username"
  value       = aws_db_instance.this.username
}

output "db_instance_id" {
  description = "RDS instance identifier"
  value       = aws_db_instance.this.id
}

output "db_security_group_id" {
  description = "Security group ID attached to the RDS instance"
  value       = aws_security_group.rds.id
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret containing DB credentials"
  value       = aws_secretsmanager_secret.rds_credentials.arn
}

output "db_password" {
  description = "RDS master password"
  sensitive   = true
  value       = random_password.master.result
}

output "db_connection_string" {
  description = "PostgreSQL connection string"
  sensitive   = true
  value       = "postgresql://${aws_db_instance.this.username}:${random_password.master.result}@${aws_db_instance.this.address}:${aws_db_instance.this.port}/${aws_db_instance.this.db_name}"
}
