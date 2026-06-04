# Platform namespaces

resource "kubernetes_namespace" "istio_system" {
  metadata {
    name   = "istio-system"
    labels = local.common_labels
  }
}

resource "kubernetes_namespace" "cert_manager" {
  metadata {
    name   = "cert-manager"
    labels = local.common_labels
  }
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    labels = merge(local.common_labels, {
      "istio-injection" = "enabled"
    })
  }
}

resource "kubernetes_namespace" "airflow" {
  metadata {
    name = "airflow"
    labels = merge(local.common_labels, {
      "istio-injection" = "enabled"
    })
  }
}

resource "kubernetes_namespace" "mlflow" {
  metadata {
    name = "mlflow"
    labels = merge(local.common_labels, {
      "istio-injection" = "enabled"
    })
  }
}
