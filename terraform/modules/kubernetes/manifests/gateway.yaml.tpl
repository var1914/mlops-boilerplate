apiVersion: networking.istio.io/v1
kind: Gateway
metadata:
  name: ${gateway_name}
  namespace: ${istio_system_namespace}
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
%{ for host in ingress_hosts ~}
        - ${host}
%{ endfor ~}
    - port:
        number: 443
        name: https
        protocol: HTTPS
      hosts:
%{ for host in ingress_hosts ~}
        - ${host}
%{ endfor ~}
      tls:
        mode: SIMPLE
        credentialName: ${tls_secret_name}
