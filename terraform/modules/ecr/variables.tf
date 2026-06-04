variable "project_name" {
  description = "Project identifier used in repository naming"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, prod)"
  type        = string
}

variable "repositories" {
  description = "List of ECR repository suffixes (e.g. api, ui)"
  type        = list(string)
  default     = ["api", "ui", "airflow"]
}

variable "image_tag_mutability" {
  description = "MUTABLE or IMMUTABLE tag mutability"
  type        = string
  default     = "MUTABLE"
}

variable "scan_on_push" {
  description = "Enable image scanning on push"
  type        = bool
  default     = true
}

variable "max_image_count" {
  description = "Maximum number of images to retain per repository"
  type        = number
  default     = 10
}

variable "tags" {
  description = "Additional tags for ECR repositories"
  type        = map(string)
  default     = {}
}
