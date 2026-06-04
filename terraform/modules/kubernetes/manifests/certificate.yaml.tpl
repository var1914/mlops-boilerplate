apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: mlops-boilerplate-tls
  namespace: ${istio_system_namespace}
spec:
  secretName: ${tls_secret_name}
  issuerRef:
    name: ${cluster_issuer_name}
    kind: ClusterIssuer
  dnsNames:
%{ for host in ingress_hosts ~}
    - ${host}
%{ endfor ~}
