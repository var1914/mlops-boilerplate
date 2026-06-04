# Grafana dashboards for kube-prometheus-stack (sidecar label grafana_dashboard=1).

locals {
  grafana_dashboard_dir = "${path.module}/../../../monitoring/grafana/dashboards"
  grafana_dashboard_files = {
    for filename in fileset(local.grafana_dashboard_dir, "*.json") :
    trimsuffix(filename, ".json") => "${local.grafana_dashboard_dir}/${filename}"
  }
}

resource "kubernetes_config_map_v1" "grafana_dashboards" {
  for_each = local.grafana_dashboard_files

  metadata {
    name      = "grafana-dashboard-${each.key}"
    namespace = "monitoring"
    labels = {
      grafana_dashboard = "1"
    }
  }

  data = {
    "${each.key}.json" = file(each.value)
  }
}
