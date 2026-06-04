output "eks_cluster_role_arn" {
  description = "ARN of the IAM role for the EKS cluster"
  value       = aws_iam_role.cluster.arn
}

output "eks_node_role_arn" {
  description = "ARN of the IAM role for the EKS node group"
  value       = aws_iam_role.node.arn
}

output "mlflow_role_arn" {
  description = "ARN of the IAM role for the MLflow service account"
  value       = aws_iam_role.mlflow.arn
}

output "airflow_role_arn" {
  description = "ARN of the IAM role for the Airflow worker service account"
  value       = aws_iam_role.airflow.arn
}

output "api_role_arn" {
  description = "ARN of the IAM role for the API service account"
  value       = aws_iam_role.api.arn
}
