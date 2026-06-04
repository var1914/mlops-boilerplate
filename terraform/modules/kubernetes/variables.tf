variable "project_name" {
  description = "Project identifier used in resource naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
}

variable "api_runtime_environment" {
  description = "ENVIRONMENT value in ml-pipeline-config; defaults to var.environment when null"
  type        = string
  default     = null
  nullable    = true
}

variable "api_service_namespace" {
  description = "Namespace of the crypto-prediction-api Service for Istio routing"
  type        = string
  default     = "ml-pipeline"
}

variable "api_service_port" {
  description = "Service port for the inference API"
  type        = number
  default     = 8000
}

variable "app_namespaces" {
  description = "Additional namespaces for application workloads (API, UI, etc.)"
  type = list(object({
    name            = string
    istio_injection = optional(bool, false)
    labels          = optional(map(string), {})
  }))
  default = []
}

variable "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  type        = string
  default     = ""
}

variable "rds_port" {
  description = "RDS PostgreSQL port"
  type        = number
  default     = 5432
}

variable "rds_username" {
  description = "RDS master username"
  type        = string
  default     = ""
}

variable "rds_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "rds_db_name" {
  description = "Default RDS database name (used for app workloads)"
  type        = string
  default     = "mlflow"
}

variable "region" {
  description = "AWS region (injected into Airflow platform ConfigMap)"
  type        = string
  default     = "us-east-1"
}

variable "redis_endpoint" {
  description = "ElastiCache Redis endpoint for Airflow workloads"
  type        = string
  default     = ""
}

variable "redis_port" {
  description = "ElastiCache Redis port"
  type        = number
  default     = 6379
}

variable "mlflow_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  type        = string
  default     = ""
}

variable "raw_data_bucket" {
  description = "S3 bucket for crypto raw data"
  type        = string
  default     = ""
}

variable "crypto_features_bucket" {
  type    = string
  default = ""
}

variable "crypto_models_bucket" {
  type    = string
  default = ""
}

variable "crypto_data_versions_bucket" {
  type    = string
  default = ""
}

variable "airflow_logs_bucket" {
  description = "S3 bucket for Airflow remote task logs"
  type        = string
  default     = ""
}

variable "api_irsa_role_arn" {
  description = "IAM role ARN for inference API IRSA"
  type        = string
  default     = ""
}

variable "airflow_irsa_role_arn" {
  description = "IAM role ARN for Airflow IRSA"
  type        = string
  default     = ""
}

variable "ingress_ui_host" {
  description = "Optional UI ingress hostname"
  type        = string
  default     = ""
}

variable "ingress_api_host" {
  description = "API ingress hostname"
  type        = string
  default     = ""
}

variable "acme_email" {
  type    = string
  default = ""
}

variable "acme_use_staging" {
  type    = bool
  default = false
}

variable "tags" {
  description = "Additional tags applied via labels where supported"
  type        = map(string)
  default     = {}
}
