apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: ${cluster_issuer_name}
spec:
  acme:
    server: ${acme_server}
    email: ${acme_email}
    privateKeySecretRef:
      name: ${cluster_issuer_name}-account-key
    solvers:
      - http01:
          ingress:
            class: istio
