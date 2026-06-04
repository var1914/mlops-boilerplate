# ML Engineering Platform with MLOps

**Automate the full lifecycle of a machine learning system — from raw data to live predictions.**

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5)](https://kubernetes.io/)
[![Airflow](https://img.shields.io/badge/Airflow-3.0-017CEE)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2)](https://mlflow.org/)
[![Terraform](https://img.shields.io/badge/Terraform-1.5+-326CE5)](https://www.terraform.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What Is This?

This is a **ready-to-run platform** that shows how a real ML system works end-to-end in production. It automatically:

1. **Fetches data** — pulls cryptocurrency market data from the Binance API every hour
2. **Trains models** — builds prediction models on that data every night
3. **Serves predictions** — exposes an API so any app can request predictions in real time
4. **Monitors everything** — dashboards and alerts so you know when something breaks

Think of it as a reference implementation: a working system you can deploy, learn from, and adapt to your own data and use case.

```
Raw Data  →  Store  →  Feature Engineering  →  Train Models  →  Register  →  Serve  →  Monitor
```

---

## Who Is This For?

| You are... | How this helps |
|---|---|
| **ML engineer** | Production-ready MLOps patterns (Airflow, MLflow, K8s) already wired together |
| **Backend engineer** | Learn how ML systems differ from regular web services |
| **Engineering student** | See how a complete, real-world ML system is structured |
| **Tech lead / architect** | Reference architecture for evaluating MLOps tooling |

You don't need an ML background to run this. You do need basic comfort with the command line and Docker.

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| ETL Pipeline (data collection) | ✅ Working | ~2.28M records, fully automated end-to-end |
| Infrastructure (Docker Compose) | ✅ Working | One command, zero manual steps |
| Infrastructure (Kubernetes) | ✅ Working | Helm-based, one command, fully reproducible |
| Infrastructure (AWS) | ✅ Working | Terraform-managed, one command, fully reproducible |
| ML Training Pipeline | ✅ Working | 60 models trained and registered in MLflow |
| Model Promotion | ✅ Working | All models tagged `@champion` alias automatically |
| Inference API (predictions) | ✅ Working | FastAPI service, loads `@champion` models, Prometheus metrics |
| Auto model reload | ✅ Working | `reload_api` Airflow task reloads inference API after each training run |
| Grafana MLOps dashboards | ✅ Working | 4 custom dashboards: pipeline health, model performance, predictions, data quality |


---

## How It Works

The platform has three automated pipelines that run in sequence:

### 1. ETL Pipeline — runs hourly
Fetches raw market data (price, volume, trades) for 10 cryptocurrencies from Binance, saves the raw files, and loads the cleaned records into a database.

```
Binance API  →  MinIO (raw file storage)  →  PostgreSQL (database)
```

### 2. ML Training Pipeline — runs nightly at 2 AM
Reads the database, computes 80+ features (moving averages, momentum indicators, volatility, etc.), trains LightGBM and XGBoost models for each symbol, evaluates them, and registers the best versions in MLflow.

```
PostgreSQL  →  Feature Engineering  →  Model Training  →  MLflow (model registry)
                                            ↓
                              10 symbols × 3 tasks × 2 algorithms = 60 models
```

**What it trains per symbol:**
- `return_4step` — predicted price return over the next hour (regression)
- `direction_4step` — predicted price direction up/down (classification)
- `volatility_4step` — predicted price volatility over the next hour (regression)

After training, all models are promoted to the **`@champion`** alias in MLflow — making them queryable by name at inference time.

### 3. Inference API — always running
Loads the `@champion` models from MLflow and serves predictions over HTTP. Scales automatically under load.

```
HTTP Request  →  Feature Generation  →  Model Prediction  →  HTTP Response
```

After training completes, the `reload_api` task automatically calls `POST /models/reload` on the inference API — no manual step required.

All three pipelines are orchestrated by **Apache Airflow**, which schedules, monitors, and retries tasks automatically.

---

## Quick Start

### Option A: Docker Compose (Easiest — no Kubernetes needed)

Good for exploring the system locally.

**Prerequisites:** Docker Desktop with at least 6GB RAM allocated

```bash
git clone <repo-url>
cd ml-eng-with-ops

# Start all services (databases, Airflow, MLflow, MinIO, monitoring)
docker compose up -d
```

**What happens automatically on first start:**
- PostgreSQL creates all databases (`airflow`, `mlflow`, `crypto`)
- MinIO creates all required buckets (`crypto-raw-data`, `crypto-features`, `mlflow-artifacts`, `crypto-models`)
- Airflow runs DB migrations and creates all required task pools (`binance_api_pool`, `postgres_pool`, `ml_training_pool`)

> **First run takes 8–12 minutes.** Airflow and MLflow install ~500MB of Python ML libraries (scikit-learn, LightGBM, XGBoost, etc.) at startup. You will see `(health: starting)` or `(unhealthy)` in Docker Desktop during this time — this is normal.

**How to know when it's actually ready:**

```bash
# Returns JSON when Airflow is fully up (ignore Docker's health status display)
curl http://localhost:8080/api/v2/monitor/health

# MLflow
curl http://localhost:5001/health
```

When Airflow is ready, the response looks like:
```json
{"metadatabase":{"status":"healthy"},"scheduler":{"status":"healthy"},"dag_processor":{"status":"healthy"}}
```

**Open the dashboards:**

| Service | URL | Login |
|---|---|---|
| Airflow (pipeline scheduler) | http://localhost:8080 | admin / admin123 |
| MLflow (model registry) | http://localhost:5001 | no login |
| Grafana (monitoring) | http://localhost:3000 | admin / admin |
| MinIO (file storage) | http://localhost:9001 | admin / admin123 |
| Prometheus (raw metrics) | http://localhost:9090 | no login |

---

### Option B: Kubernetes (Recommended for production-like setup)

**Prerequisites:**
- Docker Desktop with Kubernetes enabled *(Settings → Kubernetes → Enable Kubernetes)*
- `kubectl`: `brew install kubectl`
- `helm`: `brew install helm`
- At least 8GB RAM, 4 CPUs allocated to Docker Desktop

```bash
git clone <repo-url>
cd ml-eng-with-ops

# Deploy everything — first run takes 20-30 minutes (builds custom Docker images)
./scripts/k8s-bootstrap.sh

# Check everything is running
./scripts/k8s-bootstrap.sh --status
```

> **What happens during first run:** The script starts a local Docker registry, builds custom Airflow and inference API images, pushes them to the local registry, creates a persistent volume for logs, deploys all services via Helm (PostgreSQL, MinIO, Redis, MLflow, Airflow, Prometheus, Grafana, inference API), configures Airflow, and applies Grafana dashboards. Subsequent runs skip steps that are already complete.

**Access the services** (port-forward to your laptop):

```bash
# Airflow — pipeline scheduler
kubectl port-forward -n ml-pipeline svc/airflow-api-server 8080:8080
# Open: http://localhost:8080  (admin / admin123)

# MLflow — model registry
kubectl port-forward -n ml-pipeline svc/ml-mlflow 5000:5000
# Open: http://localhost:5000

# Grafana — monitoring dashboards
kubectl port-forward -n ml-pipeline svc/ml-monitoring-grafana 3000:80
# Open: http://localhost:3000  (admin / admin123)

# MinIO — file storage console
kubectl port-forward -n ml-pipeline svc/minio 9001:9001
# Open: http://localhost:9001  (admin / admin123)
```

---

## Running the ETL Pipeline

The ETL pipeline fetches ~2.28 million crypto records across 10 symbols. It runs automatically on a schedule, but you can trigger it manually too.

> **Note:** DAGs start paused by default. Once Airflow is healthy, unpause and trigger the pipeline.

**Via Airflow UI:**
1. Open http://localhost:8080 and log in with `admin / admin123`
2. Find the `etl_crypto_data_pipeline` DAG
3. Toggle it to "On" (the toggle on the left), then click the play button (▶)
4. Watch tasks complete in the Grid view — takes ~15–20 minutes to fetch all historical data

**Via command line:**
```bash
# Get an auth token
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Unpause and trigger
curl -X PATCH http://localhost:8080/api/v2/dags/etl_crypto_data_pipeline \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"is_paused": false}'

curl -X POST http://localhost:8080/api/v2/dags/etl_crypto_data_pipeline/dagRuns \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"logical_date\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
```

**Verify data loaded:**
```bash
# Docker Compose
docker exec ml-postgres psql -U crypto -d crypto \
  -c "SELECT symbol, COUNT(*) FROM crypto_data GROUP BY symbol ORDER BY symbol;"

# Kubernetes
kubectl exec -n ml-pipeline postgresql-0 -- \
  bash -c "PGPASSWORD=postgres123 psql -U postgres -d crypto \
  -c 'SELECT symbol, COUNT(*) FROM crypto_data GROUP BY symbol ORDER BY symbol;'"
```

Expected output (~2.28M total records across 10 symbols):
```
   symbol    |  count
-------------+--------
 ADAUSDT     | 250001
 AVAXUSDT    | 195575   ← less history available on Binance
 BNBUSDT     | 250001
 BTCUSDT     | 250001
 DOTUSDT     | 198869
 ETHUSDT     | 250001
 LINKUSDT    | 250001
 MATICUSDT   | 188250
 SOLUSDT     | 199609
 XRPUSDT     | 250001
```

---

## Running the ML Training Pipeline

The ML training pipeline runs nightly at 2 AM automatically after ETL. You can also trigger it manually once ETL has loaded data.

**Via Airflow UI:**
1. Open http://localhost:8080
2. Find the `ml_training_pipeline` DAG
3. Toggle it to "On", then click ▶ to trigger

**Via command line:**
```bash
TOKEN=$(curl -s -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X PATCH http://localhost:8080/api/v2/dags/ml_training_pipeline \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"is_paused": false}'

curl -X POST http://localhost:8080/api/v2/dags/ml_training_pipeline/dagRuns \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"logical_date\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
```

**Pipeline stages:**

```
validate_data  →  feature_group (×10 symbols, parallel)
                       ↓
               training_group (×10 symbols, max 3 concurrent)
                       ↓
               promote_models  →  reload_api  →  end
```

The final `reload_api` task automatically calls `POST /models/reload` on the running inference API — no manual intervention needed after training.

Each training task per symbol:
1. Loads 250K+ feature rows from MinIO
2. Trains LightGBM with early stopping
3. Trains XGBoost with early stopping
4. Logs both models + metrics to MLflow
5. Registers models in MLflow model registry

After all 10 symbols complete, `promote_models` sets the `@champion` alias on every model's latest version.

**What gets trained:**
- 10 symbols × 3 prediction tasks × 2 algorithms = **60 models**
- Training takes ~15–25 minutes total (3 symbols run in parallel, constrained by `ml_training_pool`)

**Verify models registered:**
```bash
# Port-forward MLflow first (K8s) or use http://localhost:5001 (Docker Compose)
kubectl port-forward -n ml-pipeline svc/ml-mlflow 5000:5000

curl -s "http://localhost:5000/api/2.0/mlflow/registered-models/search?max_results=100" | \
  python3 -c "
import sys, json
models = json.load(sys.stdin).get('registered_models', [])
champion = [m for m in models if any(a['alias']=='champion' for a in m.get('aliases', []))]
print(f'Total models: {len(models)}')
print(f'With @champion alias: {len(champion)}')
"
```

Expected: 60 models registered, all with `@champion` alias.

---

## End-to-End Verification

After the full pipeline has run (ETL → ML training → model promotion → API reload), verify everything is working:

```bash
# Port-forward the inference API
kubectl port-forward -n ml-pipeline svc/crypto-prediction-api 8000:8000

# 1. Health check — should show 30 loaded models, all 10 symbols
curl http://localhost:8000/health

# 2. Readiness — 200 means models are loaded
curl http://localhost:8000/ready

# 3. Predictions for each symbol
for sym in BTCUSDT ETHUSDT SOLUSDT BNBUSDT LINKUSDT; do
  echo "=== $sym ===" && curl -s http://localhost:8000/predict/$sym/summary | python3 -m json.tool
done

# 4. MLflow — should show 60 registered models, all with @champion
kubectl port-forward -n ml-pipeline svc/ml-mlflow 5001:5000
curl -s "http://localhost:5001/api/2.0/mlflow/registered-models/search?max_results=100" | \
  python3 -c "
import sys,json
models=json.load(sys.stdin).get('registered_models',[])
champ=[m for m in models if any(a['alias']=='champion' for a in m.get('aliases',[]))]
print(f'Total models: {len(models)}, @champion: {len(champ)}')
"

# 5. Grafana dashboards — open http://localhost:3000 after port-forwarding
kubectl port-forward -n ml-pipeline svc/ml-monitoring-grafana 3000:80
# Login: admin / admin123
# Dashboards: ML Pipeline Health, Model Performance, Prediction Metrics, Data Quality
```

Expected: 60 MLflow models, 30 loaded in API (XGBoost + LightGBM × 3 tasks × 10 symbols), predictions returning for all symbols.

---

## Adapt to Your Own Use Case

The crypto example is just the default dataset. The platform works for **any ML problem**.

```bash
cd examples/generic-ml-usecase

# Try a pre-built example
./run_demo.sh demand_forecasting     # predict product demand
./run_demo.sh churn_prediction       # predict customer churn
./run_demo.sh fraud_detection        # detect fraudulent transactions

# Or train on your own CSV
python examples/generic-ml-usecase/train_model.py \
    --data your_data.csv \
    --target your_target_column \
    --task regression \
    --model-name my_model \
    --promote
```

**Supported problem types:**

| Problem type | Examples |
|---|---|
| Regression | Demand forecasting, price prediction, sales forecasting |
| Classification | Fraud detection, churn prediction, spam detection |
| Time series | Stock prediction, energy consumption, traffic forecasting |

See [examples/generic-ml-usecase/ADAPTATION_GUIDE.md](examples/generic-ml-usecase/ADAPTATION_GUIDE.md) for step-by-step instructions.

---

## API Reference

Once models are trained and promoted, the inference API is available at `http://localhost:8000`.

| Endpoint | Method | What it does |
|---|---|---|
| `/predict/{symbol}` | POST | Get predictions for one symbol |
| `/predict/batch` | POST | Predictions for multiple symbols at once |
| `/models` | GET | List all loaded models |
| `/models/reload` | POST | Reload models from registry (no downtime) |
| `/health` | GET | Full system health check |
| `/ready` | GET | Is the service ready to accept requests? |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

**Start the inference API (Docker Compose):**
```bash
docker compose --profile inference up -d api
```

**Start the inference API (Kubernetes) — after ML training completes:**
```bash
# The API is already deployed by the bootstrap script (returns 503 until models are loaded).
# After ML training finishes and models are registered with @champion alias, reload:
kubectl port-forward -n ml-pipeline svc/crypto-prediction-api 8000:8000

# Force reload models from MLflow
curl -X POST http://localhost:8000/models/reload

# Check status
curl http://localhost:8000/health
curl http://localhost:8000/ready  # 200 = models loaded, 503 = still loading

# Get predictions
curl http://localhost:8000/predict/BTCUSDT/summary
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTCUSDT", "ETHUSDT"]}'
```

> **Note:** The API starts immediately but returns `503 /ready` until ML training has run and models are registered in MLflow. Once the `ml_training_pipeline` DAG completes, its final `reload_api` task calls `POST /models/reload` automatically — the API becomes ready without any manual step.

**Grafana MLOps dashboards** (open after port-forwarding Grafana):

| Dashboard | What it shows |
|---|---|
| **ML Pipeline Health** | Airflow task success/failure rates, pool utilization, pod restarts |
| **Model Performance** | Loaded model count per symbol, prediction success rate, latency percentiles |
| **Prediction Metrics** | Request rate, HTTP latency p50/p95/p99, error rate, requests by endpoint |
| **Data Quality** | Records per symbol, data freshness, last ETL timestamp, run history |

---

## Project Structure

```
ml-eng-with-ops/
├── dags/
│   ├── airflow_dags/            # DAG definitions (ETL + ML training schedules)
│   │   ├── etl_pipeline_dag.py
│   │   └── ml_training_dag.py
│   ├── etl/                     # ETL logic (extraction, loading, config)
│   └── ml/                      # ML pipeline modules
│       ├── feature_eng.py       # 80+ technical indicators
│       ├── model_training.py    # LightGBM + XGBoost training + MLflow logging
│       ├── model_promotion.py   # Model promotion workflow
│       ├── automated_data_validation.py
│       └── inference_feature_pipeline.py
│
├── app/
│   └── production_api.py        # FastAPI inference service
│
├── src/
│   └── config/                  # Pydantic settings (DB, MinIO, Redis, MLflow)
│
├── scripts/
│   └── k8s-bootstrap.sh         # One-command Kubernetes deployment + teardown
│
├── docker/
│   ├── Dockerfile.airflow       # Custom Airflow image (libgomp1 + all ML deps)
│   └── Dockerfile.inference     # Inference API image
│
├── terraform/                   # AWS infrastructure (Terraform)
│   ├── bootstrap/               # Remote state (S3 + DynamoDB)
│   ├── environments/dev|prod/   # Per-environment apply
│   └── modules/                 # VPC, EKS, RDS, S3, IAM, Helm, …
├── airflow/                     # Airflow Helm values
├── mlflow/                      # MLflow Helm values
├── postgresql/                  # PostgreSQL Helm values (correct schema)
├── monitoring/                  # Grafana dashboards + Prometheus config
├── examples/                    # Generic ML use case templates
├── docker-compose.yml           # Local development stack
└── .env.example                 # Environment variable template
```

---

## Tech Stack

| Tool | What it does in plain English |
|---|---|
| **Apache Airflow 3.0** | Scheduler — runs pipelines on a schedule, retries failures, shows status |
| **MLflow 3.x** | Model registry — tracks experiments, versions models, manages aliases (`@champion`) |
| **PostgreSQL** | Main database — stores all the structured data |
| **MinIO** | File storage — stores raw data files, features, and model artifacts (like AWS S3, but local) |
| **Redis** | Cache — speeds up repeated lookups during inference |
| **FastAPI** | Web framework — exposes predictions via HTTP API |
| **LightGBM / XGBoost** | ML algorithms — the actual models that make predictions |
| **Prometheus** | Metrics collection — records numbers over time (request counts, latencies, etc.) |
| **Grafana** | Dashboards — visualizes the Prometheus metrics |
| **Kubernetes** | Container orchestration — runs and scales all the services |
| **Helm** | Package manager for Kubernetes — simplifies deploying complex service stacks |
| **Docker** | Containerization — packages each service with its dependencies |
| **Terraform** | Infrastructure as code — provisions and manages AWS resources |

---

## Glossary

**DAG** (Directed Acyclic Graph) — In Airflow, a DAG is a pipeline definition. It describes which tasks to run, in what order, and how often.

**MLflow** — A tool that tracks ML experiments (which parameters were used, what metrics were achieved) and stores model versions in a registry so you can deploy specific versions.

**`@champion` alias** — MLflow 3.x way of tagging the current best model version. The inference API loads `@champion` so you can promote a new version without touching code.

**MinIO** — An open-source file storage system compatible with Amazon S3. Used here to store raw data files, engineered features, and model artifacts.

**KubernetesExecutor** — An Airflow setting that runs each pipeline task inside its own isolated container (pod), then deletes it when done. Saves resources and avoids conflicts.

**HPA (Horizontal Pod Autoscaler)** — A Kubernetes feature that automatically adds more copies of a service when traffic increases, and removes them when traffic drops.

**MLOps** — The practice of applying software engineering principles (automation, monitoring, versioning) to machine learning systems.

**OHLCV** — Open, High, Low, Close, Volume. Standard fields in financial market data representing price and trading activity for a time period.

**Feature engineering** — The process of transforming raw data into inputs that a model can learn from. For example, turning raw price data into "14-day moving average" or "momentum over 7 days".

**Inference** — Using a trained model to make predictions on new data. As opposed to *training*, which is the process of fitting the model.

**Upsert** — A database operation that inserts a new row if it doesn't exist, or updates it if it does. Prevents duplicate records on repeated runs.

---

## Going to Production

The [Quick Start](#quick-start) paths cover **local** deployment (Compose or Kubernetes on Docker Desktop). That is enough for learning and for running the pipelines in this README end to end.

**AWS production** is a third path: the same DAGs, inference API, and workflows above, with managed backing services on EKS. Configure and deploy via [terraform/README.md](terraform/README.md) and [scripts/deploy-aws-prod.sh](scripts/deploy-aws-prod.sh)

**Bootstrap:**
```bash
./scripts/deploy-aws-prod.sh bootstrap
```

**Development:**
```bash
cd terraform/environments/dev
cp terraform.tfvars.example terraform.tfvars
./scripts/deploy-aws-prod.sh all dev
```

**Production deploy:**
```bash
cd terraform/environments/prod
cp terraform.tfvars.example terraform.tfvars
./scripts/deploy-aws-prod.sh all prod
```

Then point your API DNS at the Istio load balancer (see [terraform/README.md](terraform/README.md)). During `all` / `infra`, the deploy script shows a Terraform plan and asks **`[y/N]`** before applying; use `TF_AUTO_APPROVE=1` only for CI. Staged commands (`infra`, `databases`, `build-push`, `apps`) are documented in [terraform/README.md](terraform/README.md).

---

## Contributing

Contributions are welcome. Areas where help is most needed:

- Adding pytest test coverage for ETL and ML pipeline modules
- Model drift detection and alerting dashboards
- Support for PyTorch and TensorFlow models
- HPA (Horizontal Pod Autoscaler) configuration for the inference API

**To contribute:**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes and test (both Docker Compose and K8s if possible)
4. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Built With

[FastAPI](https://fastapi.tiangolo.com/) · [MLflow](https://mlflow.org/) · [Apache Airflow](https://airflow.apache.org/) · [Prometheus](https://prometheus.io/) · [Grafana](https://grafana.com/) · [Kubernetes](https://kubernetes.io/) · [MinIO](https://min.io/) · [Redis](https://redis.io/) · [LightGBM](https://lightgbm.readthedocs.io/) · [XGBoost](https://xgboost.readthedocs.io/) · [Terraform](https://www.terraform.io/)
