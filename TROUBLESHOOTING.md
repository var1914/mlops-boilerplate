# Troubleshooting Guide

Common issues and fixes when running the ML Engineering Platform.

---

## Kubernetes / Airflow Issues

### 1. API Server Error: "invalid choice: 'api-server'"

**Symptom** — API server pod crashes with:
```
airflow command error: argument GROUP_OR_COMMAND: invalid choice: 'api-server'
/home/airflow/.local/lib/python3.9/site-packages/airflow/...
```

**Root cause** — Wrong Airflow version deployed (2.x base image instead of 3.0).

**Fix:**
```bash
# Ensure you're on Helm chart version 1.18.0
helm upgrade --install airflow apache-airflow/airflow \
  --namespace ml-pipeline \
  --values airflow/values.yaml \
  --version 1.18.0 \
  --wait --timeout 10m

# Rebuild the custom image without cache
docker build --no-cache \
  -t localhost:5050/custom-airflow:0.0.8 \
  -f docker/Dockerfile.airflow .
docker push localhost:5050/custom-airflow:0.0.8
```

Check pod logs show Python 3.12 (not 3.9) to confirm the right image is running.

---

### 2. DAGs Not Visible in Airflow UI

**Symptom** — UI shows "No DAGs found" but the CLI shows them:
```bash
kubectl exec -n ml-pipeline deployment/airflow-scheduler -- airflow dags list
```

**Root cause** — A DAG persistence volume is mounting and overriding the DAGs baked into the Docker image.

**Fix** — Disable DAG persistence in [airflow/values.yaml](airflow/values.yaml):
```yaml
dags:
  persistence:
    enabled: false   # DAGs are baked into the custom Docker image
```

Then redeploy:
```bash
./scripts/k8s-bootstrap.sh --infra-only
```

---

### 3. Worker Pods Failing with "DAG not found"

**Symptom** — Worker pods crash with:
```json
{"level":"error","event":"DAG not found during start up","dag_id":"etl_crypto_data_pipeline"}
```

**Root cause** — Same as issue #2 above.

**Fix** — Same as issue #2: disable DAG persistence in `airflow/values.yaml`.

---

### 4. Database Schema Mismatch

**Symptom:**
```
Database error during bulk insert: column "trades_count" of relation "crypto_data" does not exist
```

**Root cause** — Old table schema used `num_trades` instead of `trades_count`.

**Fix** — Drop and recreate the table:
```bash
# Drop the old table
kubectl exec -n ml-pipeline postgresql-0 -- \
  psql -U postgres -d crypto -c "DROP TABLE IF EXISTS crypto_data CASCADE;"

# Redeploy (script will recreate with the correct schema)
./scripts/k8s-bootstrap.sh --infra-only
```

Correct schema (from [dags/etl/loader.py](dags/etl/loader.py)):
```sql
CREATE TABLE crypto_data (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    open_time BIGINT NOT NULL,
    close_time BIGINT NOT NULL,
    open_price REAL NOT NULL,
    high_price REAL NOT NULL,
    low_price REAL NOT NULL,
    close_price REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL NOT NULL,
    trades_count INTEGER NOT NULL,  -- NOT num_trades
    buy_ratio REAL,
    batch_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, open_time)
);
```

---

### 5. Airflow UI Login Fails / Session Errors

**Symptom** — UI shows database session errors or login fails repeatedly.

**Root cause** — Missing `session` table for Flask-Session.

**Fix:**
```bash
kubectl exec -n ml-pipeline postgresql-0 -- \
  psql -U postgres -d airflow -c "
CREATE TABLE IF NOT EXISTS session (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    data BYTEA,
    expiry TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_session_session_id ON session(session_id);
CREATE INDEX IF NOT EXISTS ix_session_expiry ON session(expiry);
GRANT ALL PRIVILEGES ON TABLE session TO airflow;
GRANT USAGE, SELECT ON SEQUENCE session_id_seq TO airflow;
"

# Restart the API server
kubectl rollout restart deployment/airflow-api-server -n ml-pipeline
```

---

### 6. Logs Not Accessible After Task Completes

**Symptom:**
```
Could not read served logs: HTTPConnectionPool(host='etl-crypto-data-pipeline-...', port=8793):
Max retries exceeded
```

**Root cause** — KubernetesExecutor deletes worker pods after tasks complete, and Docker Desktop's `local-path` provisioner doesn't support the `ReadWriteMany` volume mode needed for shared log storage.

**Workarounds:**

Development (accept the limitation):
```bash
# View logs while the task is still running
kubectl logs -n ml-pipeline <worker-pod-name> --follow
```

Production (remote logging):
```yaml
# airflow/values.yaml
config:
  logging:
    remote_logging: "True"
    remote_base_log_folder: "s3://your-bucket/airflow-logs/"
    remote_log_conn_id: "aws_default"
```

---

### 7. Airflow Pods Stuck Waiting for Migrations

**Symptom** — Pods stuck in init containers; timeout waiting for migrations.

**Root cause** — `waitForMigrations.enabled: true` in Helm values, but the migration job runs separately before Helm deployment.

**Fix** — Disable for all components in [airflow/values.yaml](airflow/values.yaml):
```yaml
webserver:
  waitForMigrations:
    enabled: false
scheduler:
  waitForMigrations:
    enabled: false
triggerer:
  waitForMigrations:
    enabled: false
dagProcessor:
  waitForMigrations:
    enabled: false
apiServer:
  waitForMigrations:
    enabled: false
```

Migrations run in a dedicated Kubernetes Job before Helm deployment — handled automatically by [scripts/k8s-bootstrap.sh](scripts/k8s-bootstrap.sh).

---

### 8. MinIO Connection Errors in ETL Tasks

**Symptom** — ETL tasks fail with MinIO connection errors.

**Root cause** — Wrong MinIO service name in configuration.

**Fix** — Ensure [dags/etl/config.py](dags/etl/config.py) uses `minio:9000` (not `ml-minio:9000`):
```python
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'minio:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'admin'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'admin123'),
    'secure': False
}
```

Verify the service name in your cluster:
```bash
kubectl get svc -n ml-pipeline | grep minio
```

---

## Inference API Issues

### API Returns 503 "No models loaded"

**Cause** — The API is waiting for models to be registered and promoted to Production stage in MLflow. The ML training pipeline must complete first.

**Steps:**
1. Run the ML training pipeline (see main README)
2. Open MLflow UI and promote models to "Production" stage
3. Reload the API: `curl -X POST http://localhost:8000/models/reload`
4. Verify: `curl http://localhost:8000/models`

### Model Not Loading After Reload

**Check the model naming convention:**
```
crypto_{model_type}_{task}_{symbol}
```
- Correct: `crypto_lightgbm_return_1step_BTCUSDT`
- Wrong: `my_model_v1`

**Check model stage** — must be "Production" or "Staging" in MLflow.

**Check API logs:**
```bash
# Docker Compose
docker-compose logs api

# Kubernetes
kubectl logs -n ml-pipeline -l app=crypto-prediction-api
```

---

## Docker Compose Issues

### Port Already in Use

```bash
# Find what's using the port
lsof -i :8000   # or :5001, :9090, etc.

# Or change the host port in docker-compose.yml
ports:
  - "8001:8000"   # use 8001 on your machine instead
```

### Services Crash on Startup (OOM)

Increase Docker memory allocation:
- Docker Desktop → Settings → Resources → Memory → set to 6–8 GB

### Service Not Starting

```bash
# Check logs for a specific service
docker-compose logs mlflow
docker-compose logs api

# Restart a single service
docker-compose restart api

# Rebuild and restart
docker-compose up -d --build api
```

---

## Kubernetes General Debugging

```bash
# See all pods and their status
kubectl get pods -n ml-pipeline

# Get details about a failing pod (look at the Events section)
kubectl describe pod -n ml-pipeline <pod-name>

# Stream logs from a pod
kubectl logs -n ml-pipeline <pod-name> --follow

# Stream Airflow scheduler logs
kubectl logs -n ml-pipeline deployment/airflow-scheduler --follow

# Run Airflow CLI commands
kubectl exec -n ml-pipeline deployment/airflow-scheduler -- airflow dags list
kubectl exec -n ml-pipeline deployment/airflow-scheduler -- \
  airflow tasks list etl_crypto_data_pipeline

# Check data in PostgreSQL
kubectl exec -n ml-pipeline postgresql-0 -- \
  psql -U postgres -d crypto -c "SELECT COUNT(*) FROM crypto_data;"

kubectl exec -n ml-pipeline postgresql-0 -- \
  psql -U postgres -d crypto \
  -c "SELECT symbol, COUNT(*) FROM crypto_data GROUP BY symbol;"

# Check MinIO buckets
kubectl run minio-check --rm -i --restart=Never -n ml-pipeline \
  --image=minio/mc:latest \
  --command -- /bin/sh -c \
  "mc alias set myminio http://minio:9000 admin admin123 && mc ls myminio/"

# Check Helm release status
helm status airflow -n ml-pipeline
helm status postgresql -n ml-pipeline

# Check HPA (auto-scaling) status
kubectl get hpa -n ml-pipeline
kubectl top pods -n ml-pipeline

# Check if metrics server is installed (required for HPA)
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml
```

---

## ML Pipeline Issues

### 9. `libgomp.so.1: cannot open shared object file`

**Symptom** — Training task pods crash at import with:
```
OSError: libgomp.so.1: cannot open shared object file: No such file or directory
```

**Root cause** — LightGBM requires the GNU OpenMP system library, which is not included in the base Airflow image.

**Fix** — Already included in the custom Dockerfile:
```dockerfile
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
USER airflow
```

---

### 10. `Invalid auth token: Signature verification failed`

**Symptom** — Task pods fail with `Signature verification failed` when calling the Airflow API server.

**Root cause** — After `helm upgrade --force`, the `airflow-jwt-secret` K8s secret is regenerated with a new random value, while the `AIRFLOW__API_AUTH__JWT_SECRET` env var in `values.yaml` stays at the old value. Task pods use the env var; the api-server uses the K8s secret → mismatch.

**Fix** — The bootstrap script patches the K8s secret to match the fixed env var value after every Helm install:
```bash
kubectl patch secret airflow-jwt-secret -n ml-pipeline \
  --type=json \
  -p='[{"op":"replace","path":"/data/jwt-secret","value":"'$(echo -n 'cF1A7mWl4zbZNwUyySQHxg' | base64)'"}]'
```

**Prevention** — Use `helm upgrade` (not `helm upgrade --force`) for normal updates. Only use `--force` for stuck deployments, and always follow it with the secret patch + pod restart.

---

### 11. MLflow Returns 403 — "Invalid Host header"

**Symptom** — ML training tasks log:
```
API request to endpoint /api/2.0/mlflow/experiments/get-by-name failed with error code 403
Response body: 'Invalid Host header - possible DNS rebinding attack detected'
```

**Root cause** — MLflow 3.x added DNS rebinding protection. Requests from Airflow task pods arrive with `Host: ml-mlflow:5000` (the K8s service name), which MLflow rejects.

**Fix** — The bootstrap script patches the MLflow deployment after Helm install to replace `--gunicorn-opts` with `--allowed-hosts=*` (uvicorn mode):
```bash
kubectl patch deployment ml-mlflow -n ml-pipeline --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": [
    "server", "--host=0.0.0.0", "--port=5000",
    "--backend-store-uri=...", "--default-artifact-root=s3://mlflow-artifacts/",
    "--allowed-hosts=*"
  ]}
]'
```

Note: `--allowed-hosts` only works with MLflow's uvicorn server. The community-charts/mlflow Helm chart defaults to gunicorn — these two are mutually exclusive in MLflow 3.x.

---

### 12. Airflow Log PVC Stuck in Pending

**Symptom** — After `helm install airflow`, the `airflow-logs` PVC stays in `Pending` state and pods can't start.

**Root cause** — Docker Desktop Kubernetes uses `rancher.io/local-path` with `WaitForFirstConsumer` binding mode. Dynamic PVCs created by Helm stay pending because no pod is scheduled yet.

**Fix** — Use a static PV with `storageClassName: ""` (no dynamic provisioner) that binds immediately:
```bash
# The bootstrap script creates this automatically via setup_airflow_log_storage()
# Logs land at /private/tmp/airflow-logs on your Mac (virtiofs-mounted into Docker VM)
kubectl get pv airflow-logs-pv
kubectl get pvc airflow-logs-pvc -n ml-pipeline
```

---

### 13. `PermissionError: Permission denied: '/opt/airflow/logs/dag_id=...'`

**Symptom** — dag-processor or task pods crash with permission denied on the log directory.

**Root cause** — The hostPath directory `/private/tmp/airflow-logs` is created by your Mac user. Inside Docker Desktop's Linux VM, Airflow runs as uid 50000 which can't write to it.

**Fix** — The bootstrap script runs a K8s Job as root to fix permissions before Airflow starts:
```bash
# Runs automatically in setup_airflow_log_storage()
# chown -R 50000:0 /opt/airflow/logs
```

The `extraInitContainers` in `airflow/values.yaml` also runs this fix on every pod startup (self-healing).

---

## Known Limitations (not bugs)

| Limitation | Reason | Workaround |
|---|---|---|
| AVAXUSDT / DOTUSDT / MATICUSDT / SOLUSDT have fewer records | Less historical data available on Binance | Expected behavior |
| Local Docker registry at `localhost:5050` | Simplifies local development | For multi-node K8s, use a cloud registry (ECR, GCR, ACR) |
| `helm upgrade --force` breaks JWT auth | Force-recreates the jwt-secret with a new random value | Use `helm upgrade` without `--force`; bootstrap script patches the secret after install |
| MLflow success messages appear as ERROR in Airflow logs | MLflow 3.x writes INFO messages to stderr; Airflow labels all stderr as ERROR | Cosmetic only — operations succeed |
