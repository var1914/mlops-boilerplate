variable "project_name" {
  description = "Project identifier used for resource naming"
  type        = string
}

variable "environment" {
  description = "Deployment environment (dev/prod)"
  type        = string
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
