output "grafana_namespace" {
  description = "Namespace where Grafana is deployed"
  value       = "monitoring"
}

output "prometheus_namespace" {
  description = "Namespace where Prometheus is deployed"
  value       = "monitoring"
}

output "mlflow_namespace" {
  description = "Namespace where MLflow is deployed"
  value       = "mlflow"
}

output "istio_namespace" {
  description = "Namespace where Istio is deployed"
  value       = "istio-system"
}

output "platform_helm_ready" {
  description = "Signals cert-manager, Istio, MLflow, and monitoring Helm releases are installed"
  value       = true
  depends_on = [
    helm_release.cert_manager,
    time_sleep.wait_for_cert_manager,
    helm_release.istio_gateway,
    helm_release.mlflow,
    helm_release.kube_prometheus,
  ]
}

