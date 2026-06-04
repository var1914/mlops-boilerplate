output "vpc_id" {
  value = module.vpc.vpc_id
}

output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_update_kubeconfig" {
  value = "aws eks --region ${var.region} update-kubeconfig --name ${module.eks.cluster_name}"
}

output "rds_endpoint" {
  value = module.rds.db_endpoint
}

output "rds_secret_arn" {
  value = module.rds.secrets_manager_secret_arn
}

output "redis_endpoint" {
  value = module.elasticache.redis_endpoint
}

output "s3_buckets" {
  value = {
    mlflow_artifacts     = module.s3.mlflow_artifacts_bucket_name
    crypto_raw_data      = module.s3.raw_data_bucket_name
    crypto_features      = module.s3.crypto_features_bucket_name
    crypto_models        = module.s3.crypto_models_bucket_name
    crypto_data_versions = module.s3.crypto_data_versions_bucket_name
    airflow_logs         = module.s3.airflow_logs_bucket_name
  }
}

output "ecr_inference_api_repository_url" {
  value = module.ecr.inference_api_repository_url
}

output "ecr_airflow_repository_url" {
  value = module.ecr.airflow_repository_url
}

output "api_irsa_role_arn" {
  value = module.iam.api_role_arn
}

output "airflow_irsa_role_arn" {
  value = module.iam.airflow_role_arn
}

output "ingress_api_url" {
  value = "https://${module.kubernetes.ingress_api_host}"
}
