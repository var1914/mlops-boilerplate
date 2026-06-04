output "vpc_id" {
  description = "VPC ID"
  value       = module.vpc.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "eks_update_kubeconfig" {
  description = "Command to update kubeconfig"
  value       = "aws eks --region ${var.region} update-kubeconfig --name ${module.eks.cluster_name}"
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = module.rds.db_endpoint
}

output "rds_secret_arn" {
  description = "Secrets Manager ARN for RDS credentials"
  value       = module.rds.secrets_manager_secret_arn
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.elasticache.redis_endpoint
}

output "s3_buckets" {
  description = "S3 bucket names used by the pipeline"
  value = {
    mlflow_artifacts     = module.s3.mlflow_artifacts_bucket_name
    crypto_raw_data      = module.s3.raw_data_bucket_name
    crypto_features      = module.s3.crypto_features_bucket_name
    crypto_models        = module.s3.crypto_models_bucket_name
    crypto_data_versions = module.s3.crypto_data_versions_bucket_name
    airflow_logs         = module.s3.airflow_logs_bucket_name
  }
}

output "app_namespace_names" {
  description = "App deployment namespace names"
  value       = module.kubernetes.app_namespace_names
}

output "ecr_inference_api_repository_url" {
  description = "ECR repository URL for inference API images"
  value       = module.ecr.inference_api_repository_url
}

output "ecr_airflow_repository_url" {
  description = "ECR repository URL for Airflow images"
  value       = module.ecr.airflow_repository_url
}

output "ecr_repository_urls" {
  description = "All ECR repository URLs"
  value       = module.ecr.repository_urls
}

output "api_irsa_role_arn" {
  description = "IAM role ARN for API pods (IRSA)"
  value       = module.iam.api_role_arn
}

output "airflow_irsa_role_arn" {
  description = "IAM role ARN for Airflow worker pods (IRSA)"
  value       = module.iam.airflow_role_arn
}

output "mlflow_irsa_role_arn" {
  description = "IAM role ARN for MLflow pods (IRSA)"
  value       = module.iam.mlflow_role_arn
}

output "ingress_api_url" {
  description = "Public HTTPS API URL"
  value       = "https://${module.kubernetes.ingress_api_host}"
}
