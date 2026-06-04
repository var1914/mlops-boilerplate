locals {
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ──────────────────────────────────────────────
#  Networking
# ──────────────────────────────────────────────
module "vpc" {
  source = "../../modules/vpc"

  project_name          = var.project_name
  environment           = var.environment
  vpc_cidr              = var.vpc_cidr
  availability_zones    = var.availability_zones
  private_subnet_cidrs  = var.private_subnet_cidrs
  public_subnet_cidrs   = var.public_subnet_cidrs
  database_subnet_cidrs = var.database_subnet_cidrs
  single_nat_gateway    = true
  tags                  = local.tags
}

# ──────────────────────────────────────────────
#  Object Storage
# ──────────────────────────────────────────────
module "s3" {
  source = "../../modules/s3"

  project_name = var.project_name
  environment  = var.environment
  tags         = local.tags
}

# ──────────────────────────────────────────────
#  ECR (container images)
# ──────────────────────────────────────────────
module "ecr" {
  source = "../../modules/ecr"

  project_name = var.project_name
  environment  = var.environment
  repositories = var.ecr_repositories
  tags         = local.tags
}

# ──────────────────────────────────────────────
#  IAM
# ──────────────────────────────────────────────
module "iam" {
  source = "../../modules/iam"

  project_name                = var.project_name
  environment                 = var.environment
  oidc_provider_arn           = module.eks.oidc_provider_arn
  oidc_provider_url           = module.eks.oidc_provider_url
  mlflow_artifacts_bucket_arn = module.s3.mlflow_artifacts_bucket_arn
  raw_data_bucket_arn         = module.s3.raw_data_bucket_arn
  crypto_features_bucket_arn  = module.s3.crypto_features_bucket_arn
  crypto_models_bucket_arn    = module.s3.crypto_models_bucket_arn
  crypto_data_versions_bucket_arn = module.s3.crypto_data_versions_bucket_arn
  airflow_logs_bucket_arn     = module.s3.airflow_logs_bucket_arn
  tags                        = local.tags
}

# ──────────────────────────────────────────────
#  EKS Cluster
# ──────────────────────────────────────────────
module "eks" {
  source = "../../modules/eks"

  project_name        = var.project_name
  environment         = var.environment
  vpc_id              = module.vpc.vpc_id
  private_subnet_ids  = module.vpc.private_subnet_ids
  cluster_role_arn    = module.iam.eks_cluster_role_arn
  node_role_arn       = module.iam.eks_node_role_arn
  kubernetes_version  = var.kubernetes_version
  node_instance_types = var.node_instance_types
  node_desired_size   = var.node_desired_size
  node_min_size       = var.node_min_size
  node_max_size       = var.node_max_size
  tags                = local.tags
}

# ──────────────────────────────────────────────
#  RDS PostgreSQL
# ──────────────────────────────────────────────
module "rds" {
  source = "../../modules/rds"

  project_name               = var.project_name
  environment                = var.environment
  vpc_id                     = module.vpc.vpc_id
  database_subnet_ids        = module.vpc.database_subnet_ids
  allowed_security_group_ids = [module.eks.node_security_group_id]
  instance_class             = var.rds_instance_class
  multi_az                   = var.rds_multi_az
  deletion_protection        = false
  skip_final_snapshot        = true
  tags                       = local.tags
}

# ──────────────────────────────────────────────
#  ElastiCache Redis
# ──────────────────────────────────────────────
module "elasticache" {
  source = "../../modules/elasticache-redis"

  project_name               = var.project_name
  environment                = var.environment
  vpc_id                     = module.vpc.vpc_id
  database_subnet_ids        = module.vpc.database_subnet_ids
  allowed_security_group_ids = [module.eks.node_security_group_id]
  node_type                  = var.redis_node_type
  tags                       = local.tags
}

# ──────────────────────────────────────────────
#  Helm (namespaces + platform charts)
# ──────────────────────────────────────────────
module "helm" {
  source = "../../modules/helm"

  project_name            = var.project_name
  environment             = var.environment
  cluster_name            = module.eks.cluster_name
  cluster_endpoint        = module.eks.cluster_endpoint
  cluster_ca_data         = module.eks.cluster_certificate_authority_data
  oidc_provider_arn       = module.eks.oidc_provider_arn
  oidc_provider_url       = module.eks.oidc_provider_url
  rds_endpoint            = module.rds.db_endpoint
  rds_port                = module.rds.db_port
  rds_db_name             = module.rds.db_name
  rds_username            = module.rds.db_username
  rds_password            = module.rds.db_password
  rds_secret_arn          = module.rds.secrets_manager_secret_arn
  mlflow_artifacts_bucket = module.s3.mlflow_artifacts_bucket_name
  mlflow_role_arn         = module.iam.mlflow_role_arn
  region                  = var.region
  tags                    = local.tags
}

# ──────────────────────────────────────────────
#  Kubernetes (secrets, app namespaces, ingress TLS)
# ──────────────────────────────────────────────
module "kubernetes" {
  source = "../../modules/kubernetes"

  project_name            = var.project_name
  environment             = var.environment
  api_runtime_environment = var.api_runtime_environment
  app_namespaces          = var.app_namespaces
  rds_endpoint            = module.rds.db_endpoint
  rds_port                = module.rds.db_port
  rds_username            = module.rds.db_username
  rds_password            = module.rds.db_password
  rds_db_name             = module.rds.db_name
  region                  = var.region
  redis_endpoint          = module.elasticache.redis_endpoint
  redis_port              = module.elasticache.redis_port
  mlflow_artifacts_bucket     = module.s3.mlflow_artifacts_bucket_name
  raw_data_bucket             = module.s3.raw_data_bucket_name
  crypto_features_bucket      = module.s3.crypto_features_bucket_name
  crypto_models_bucket        = module.s3.crypto_models_bucket_name
  crypto_data_versions_bucket = module.s3.crypto_data_versions_bucket_name
  airflow_logs_bucket         = module.s3.airflow_logs_bucket_name
  api_irsa_role_arn           = module.iam.api_role_arn
  airflow_irsa_role_arn       = module.iam.airflow_role_arn
  acme_email                  = var.acme_email
  acme_use_staging        = var.acme_use_staging
  ingress_ui_host         = var.ingress_ui_host
  ingress_api_host        = var.ingress_api_host
  tags                    = local.tags

  depends_on = [module.helm]
}
