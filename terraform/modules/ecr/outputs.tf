output "inference_api_repository_url" {
  description = "ECR repository URL for the inference API image"
  value       = try(aws_ecr_repository.this["inference-api"].repository_url, null)
}

output "airflow_repository_url" {
  description = "ECR repository URL for the Airflow image"
  value       = try(aws_ecr_repository.this["airflow"].repository_url, null)
}
