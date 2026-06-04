variable "project_name" {
  description = "Name of the project, used as prefix for all resources"
  type        = string
}

variable "environment" {
  description = "Deployment environment (e.g. dev, staging, prod)"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC where the EKS cluster will be deployed"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for the EKS cluster and node groups"
  type        = list(string)
}

variable "kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.31"
}

variable "node_instance_types" {
  description = "List of EC2 instance types for the managed node group"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "node_desired_size" {
  description = "Desired number of nodes in the managed node group"
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of nodes in the managed node group"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in the managed node group"
  type        = number
  default     = 4
}

variable "node_disk_size" {
  description = "Disk size in GB for each node in the managed node group"
  type        = number
  default     = 50
}

variable "cluster_role_arn" {
  description = "ARN of the IAM role for the EKS cluster (from IAM module)"
  type        = string
}

variable "node_role_arn" {
  description = "ARN of the IAM role for the EKS node group (from IAM module)"
  type        = string
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
