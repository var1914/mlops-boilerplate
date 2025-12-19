# Kubernetes Deployment with Helm Charts

> **⚠️ STATUS: WORK IN PROGRESS - NOT YET TESTED**
>
> This guide documents the planned Helm-based deployment approach. The manifests exist but have not been validated in a live cluster yet.

---

## 📋 Overview

This boilerplate provides Helm values files for deploying supporting infrastructure services in Kubernetes:

- **MLflow** - Model registry and experiment tracking
- **MinIO** - S3-compatible object storage for model artifacts
- **Grafana/Prometheus** - Monitoring and visualization
- **Airflow** - (Optional) Workflow orchestration for training pipelines

---

## 🗂️ Available Helm Configurations

```
├── mlflow/
│   └── values.yaml        # MLflow with PostgreSQL backend
├── minio/
│   └── values.yaml        # MinIO object storage
├── grafana/
│   └── values.yaml        # Grafana + Prometheus stack
└── airflow/
    └── values.yaml        # Airflow (for training workflows)
```

---

## 🚀 Deployment Steps (UNTESTED)

### Prerequisites

```bash
# Install Helm 3
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version

# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add apache-airflow https://airflow.apache.org
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
```

### Step 1: Create Namespace

```bash
kubectl create namespace ml-pipeline
```

### Step 2: Deploy MinIO (Object Storage)

```bash
# Deploy MinIO using custom values
helm install minio bitnami/minio \
  --namespace ml-pipeline \
  --values minio/values.yaml

# Wait for MinIO to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=minio -n ml-pipeline --timeout=300s

# Get MinIO service endpoint
kubectl get svc -n ml-pipeline minio
```

**Configuration** (`minio/values.yaml`):
- Root User: `admin`
- Root Password: `admin123`
- Persistence: 10Gi
- Console enabled

### Step 3: Deploy MLflow (Model Registry)

```bash
# Deploy MLflow with PostgreSQL backend
helm install mlflow bitnami/mlflow \
  --namespace ml-pipeline \
  --values mlflow/values.yaml

# Wait for MLflow to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mlflow -n ml-pipeline --timeout=300s

# Get MLflow service endpoint
kubectl get svc -n ml-pipeline mlflow
```

**Configuration** (`mlflow/values.yaml`):
- Service Type: ClusterIP
- Port: 5000
- PostgreSQL: Enabled (10Gi persistence)
- Artifact Root: `s3://mlflow-artifacts`
- MinIO Integration: Configured

### Step 4: Deploy Grafana + Prometheus (Monitoring)

```bash
# Deploy Prometheus stack with Grafana
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace ml-pipeline \
  --values grafana/values.yaml

# Wait for Grafana to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana -n ml-pipeline --timeout=300s

# Get Grafana endpoint
kubectl get svc -n ml-pipeline kube-prometheus-stack-grafana
```

**Configuration** (`grafana/values.yaml`):
- Admin Password: `admin123`
- Service Type: NodePort (30001)
- Prometheus Scraping: Configured for pushgateway
- Dashboard Sidecar: Enabled

### Step 5: (Optional) Deploy Airflow

```bash
# Deploy Airflow for training workflows
helm install airflow apache-airflow/airflow \
  --namespace ml-pipeline \
  --values airflow/values.yaml

# Wait for Airflow to be ready
kubectl wait --for=condition=ready pod -l component=webserver -n ml-pipeline --timeout=600s

# Get Airflow webserver endpoint
kubectl get svc -n ml-pipeline airflow-webserver
```

**Configuration** (`airflow/values.yaml`):
- Executor: KubernetesExecutor
- Custom Image: `localhost:5050/custom-airflow:0.0.4`
- DAGs Persistence: 1Gi
- Logs Persistence: 5Gi
- Default User: admin/admin123

### Step 6: Deploy Inference API

```bash
# Build and push API image (REQUIRED FIRST)
docker build -t <your-registry>/ml-inference-api:1.0.0 -f docker/Dockerfile.inference .
docker push <your-registry>/ml-inference-api:1.0.0

# Update kustomization.yaml with your registry
# Then deploy
kubectl apply -k k8s/
```

---

## 🔌 Service Endpoints

After deployment, services will be available at:

| Service | Endpoint | Port | Access |
|---------|----------|------|--------|
| **MLflow** | `mlflow.ml-pipeline.svc.cluster.local` | 5000 | ClusterIP |
| **MinIO** | `minio.ml-pipeline.svc.cluster.local` | 9000 | ClusterIP |
| **MinIO Console** | `minio.ml-pipeline.svc.cluster.local` | 9001 | ClusterIP |
| **Grafana** | NodePort | 30001 | NodePort |
| **Prometheus** | `kube-prometheus-stack-prometheus.ml-pipeline.svc.cluster.local` | 9090 | ClusterIP |
| **Airflow** | `airflow-webserver.ml-pipeline.svc.cluster.local` | 8080 | ClusterIP |
| **Inference API** | `crypto-prediction-api.ml-pipeline.svc.cluster.local` | 8000 | ClusterIP |

---

## 🔧 ConfigMap Updates Needed

The `k8s/api-configmap.yaml` needs to be updated to match Helm service names:

```yaml
data:
  # MLflow
  MLFLOW_TRACKING_URI: "http://mlflow.ml-pipeline.svc.cluster.local:5000"

  # MinIO
  MINIO_ENDPOINT: "minio.ml-pipeline.svc.cluster.local:9000"
  MINIO_ACCESS_KEY: "admin"  # Move to secrets
  MINIO_SECRET_KEY: "admin123"  # Move to secrets

  # PostgreSQL (from MLflow chart)
  DB_HOST: "mlflow-postgresql.ml-pipeline.svc.cluster.local"
  DB_PORT: "5432"
  DB_NAME: "mlflowdb"
  DB_USER: "mlflow"

  # Redis (needs separate deployment or use external)
  REDIS_HOST: "redis.ml-pipeline.svc.cluster.local"
  REDIS_PORT: "6379"
```

---

## 📊 Verification Steps

### Check All Deployments

```bash
# View all resources
kubectl get all -n ml-pipeline

# Check pod status
kubectl get pods -n ml-pipeline

# Check services
kubectl get svc -n ml-pipeline

# Check persistent volumes
kubectl get pvc -n ml-pipeline
```

### Test MLflow

```bash
# Port forward to access MLflow UI
kubectl port-forward -n ml-pipeline svc/mlflow 5000:5000

# Access at http://localhost:5000
```

### Test MinIO

```bash
# Port forward MinIO console
kubectl port-forward -n ml-pipeline svc/minio 9001:9001

# Access at http://localhost:9001
# Login: admin / admin123
```

### Test Grafana

```bash
# Access via NodePort (if using minikube)
minikube service -n ml-pipeline kube-prometheus-stack-grafana

# Or port forward
kubectl port-forward -n ml-pipeline svc/kube-prometheus-stack-grafana 3000:80

# Login: admin / admin123
```

### Test Airflow (if deployed)

```bash
# Port forward Airflow webserver
kubectl port-forward -n ml-pipeline svc/airflow-webserver 8080:8080

# Access at http://localhost:8080
# Login: admin / admin123
```

---

## ⚠️ Known Issues & Limitations

### 1. **Not Tested in Real Cluster**
- These configurations have not been validated in a live Kubernetes environment
- Service names and connectivity need verification
- Resource limits may need tuning

### 2. **Missing Redis Deployment**
- ConfigMap references Redis but no Helm chart provided
- Need to either:
  - Add Redis Helm chart deployment
  - Use external Redis service
  - Remove Redis dependency (use in-memory caching)

### 3. **Secrets Management**
- Credentials are in values.yaml files (not secure)
- Should use Kubernetes Secrets or external secret management
- Move passwords from ConfigMap to Secrets

### 4. **Airflow Custom Image**
- References `localhost:5050/custom-airflow:0.0.4`
- This image needs to be built and pushed to an accessible registry
- May not be needed for inference-only setup

### 5. **Network Policies**
- No network policies defined
- Services are open within the cluster
- Should add network segmentation for production

### 6. **Resource Limits**
- Helm charts use default resource limits
- May need tuning based on:
  - Model size
  - Traffic volume
  - Available cluster resources

### 7. **High Availability**
- Single replica for most services
- No HA configuration
- For production, should configure:
  - Multiple replicas
  - Pod disruption budgets
  - Affinity/anti-affinity rules

---

## 🧪 Testing Checklist

Before using in production, validate:

- [ ] All Helm charts deploy successfully
- [ ] All pods are in Running state
- [ ] Services are accessible within cluster
- [ ] MLflow can connect to PostgreSQL
- [ ] MLflow can store artifacts in MinIO
- [ ] Grafana can scrape Prometheus metrics
- [ ] Inference API can connect to all services
- [ ] Model registration workflow works end-to-end
- [ ] Predictions work correctly
- [ ] Monitoring dashboards show data
- [ ] Persistent volumes retain data after pod restart

---

## 🔄 Cleanup

```bash
# Delete all Helm releases
helm uninstall minio -n ml-pipeline
helm uninstall mlflow -n ml-pipeline
helm uninstall kube-prometheus-stack -n ml-pipeline
helm uninstall airflow -n ml-pipeline

# Delete inference API
kubectl delete -k k8s/

# Delete namespace (this will delete PVCs too!)
kubectl delete namespace ml-pipeline
```

---

## 📝 Next Steps

1. **Test Helm Deployments**
   - Deploy to minikube/kind cluster
   - Validate all services start correctly
   - Test connectivity between services

2. **Update ConfigMap**
   - Fix service endpoint references
   - Move secrets to Kubernetes Secrets
   - Add Redis deployment or remove dependency

3. **Security Hardening**
   - Use Kubernetes Secrets for passwords
   - Add network policies
   - Configure RBAC
   - Enable TLS for inter-service communication

4. **Production Readiness**
   - Add resource limits to Helm values
   - Configure HA for critical services
   - Setup backup strategy for persistent volumes
   - Add monitoring alerts

5. **Documentation**
   - Document tested deployment procedure
   - Add troubleshooting guide
   - Create architecture diagram with service dependencies

---

**Last Updated**: 2025-12-19
**Status**: ⚠️ UNTESTED - Work in Progress
