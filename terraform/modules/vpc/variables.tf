variable "project_name" {
  description = "Project identifier used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev/prod)"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones to deploy into"
  type        = list(string)
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets (one per AZ)"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets (one per AZ)"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24"]
}

variable "database_subnet_cidrs" {
  description = "CIDR blocks for isolated database subnets (one per AZ)"
  type        = list(string)
  default     = ["10.0.201.0/24", "10.0.202.0/24"]
}

variable "single_nat_gateway" {
  description = "Use a single NAT gateway (cheaper for dev) instead of one per AZ (HA for prod)"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
