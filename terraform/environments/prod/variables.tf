variable "project_name" {
  description = "Project identifier used in resource naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

# --- VPC ---
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.1.0.0/16"
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.1.1.0/24", "10.1.2.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.1.101.0/24", "10.1.102.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for database subnets"
  type        = list(string)
  default     = ["10.1.201.0/24", "10.1.202.0/24"]
}

# --- EKS ---
variable "kubernetes_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.31"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS node group"
  type        = list(string)
  default     = ["m5.large"]
}

variable "node_desired_size" {
  description = "Desired number of EKS nodes"
  type        = number
  default     = 3
}

variable "node_min_size" {
  description = "Minimum number of EKS nodes"
  type        = number
  default     = 2
}

variable "node_max_size" {
  description = "Maximum number of EKS nodes"
  type        = number
  default     = 6
}

# --- RDS ---
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "rds_multi_az" {
  description = "Enable Multi-AZ for RDS"
  type        = bool
  default     = true
}

# --- ElastiCache ---
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.small"
}

variable "ecr_repositories" {
  description = "ECR repository suffixes (full name: {project}-{env}-{suffix})"
  type        = list(string)
  default     = ["inference-api", "airflow"]
}

variable "acme_email" {
  description = "Email for Let's Encrypt (cert-manager ClusterIssuer)"
  type        = string
}

variable "acme_use_staging" {
  description = "Use Let's Encrypt staging CA"
  type        = bool
  default     = false
}

variable "ingress_ui_host" {
  description = "Optional public hostname for a UI (leave empty if API-only)"
  type        = string
  default     = ""
}

variable "ingress_api_host" {
  description = "Public hostname for the inference API"
  type        = string
  default     = "api.crypto-ml.example.com"
}

variable "api_runtime_environment" {
  description = "ENVIRONMENT value in ml-pipeline-config (defaults to environment, e.g. dev or prod)"
  type        = string
  default     = null
  nullable    = true
}

variable "app_namespaces" {
  description = "Namespaces for application workloads"
  type = list(object({
    name            = string
    istio_injection = optional(bool, false)
    labels          = optional(map(string), {})
  }))
  default = [
    { name = "ml-pipeline", istio_injection = true },
  ]
}
