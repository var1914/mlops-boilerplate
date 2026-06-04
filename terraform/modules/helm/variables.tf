variable "project_name" {
  description = "Project identifier"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
}

variable "cluster_endpoint" {
  description = "EKS cluster endpoint"
  type        = string
}

variable "cluster_ca_data" {
  description = "EKS cluster CA certificate data"
  type        = string
}

variable "oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  type        = string
}

variable "oidc_provider_url" {
  description = "EKS OIDC provider URL without https://"
  type        = string
}

variable "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  type        = string
}

variable "rds_port" {
  description = "RDS PostgreSQL port"
  type        = number
  default     = 5432
}

variable "rds_db_name" {
  description = "RDS database name"
  type        = string
}

variable "rds_username" {
  description = "RDS master username"
  type        = string
}

variable "rds_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "rds_secret_arn" {
  description = "Secrets Manager ARN for RDS credentials"
  type        = string
}

variable "mlflow_artifacts_bucket" {
  description = "S3 bucket name for MLflow artifacts"
  type        = string
}

variable "mlflow_role_arn" {
  description = "IAM role ARN for MLflow pods"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "tags" {
  description = "Resource tags"
  type        = map(string)
  default     = {}
}

