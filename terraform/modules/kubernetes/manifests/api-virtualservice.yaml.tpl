# Routes public API hostname (Istio Gateway) to the inference Service in-cluster.
apiVersion: networking.istio.io/v1
kind: VirtualService
metadata:
  name: crypto-prediction-api
  namespace: ${api_namespace}
spec:
  hosts:
    - ${ingress_api_host}
  gateways:
    - ${istio_system_namespace}/${gateway_name}
  http:
    - match:
        - uri:
            prefix: /
      route:
        - destination:
            host: ${api_service_host}
            port:
              number: ${api_service_port}
