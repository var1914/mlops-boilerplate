# ML Engineering Platform with MLOps

**Automate the full lifecycle of a machine learning system — from raw data to live predictions.**

[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5)](https://kubernetes.io/)
[![Airflow](https://img.shields.io/badge/Airflow-3.0-017CEE)](https://airflow.apache.org/)
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
Raw Data  →  Store  →  Train Models  →  Serve Predictions  →  Monitor
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
| ETL Pipeline (data collection) | ✅ Working | 2.25M records loaded and verified |
| Infrastructure (databases, storage, monitoring) | ✅ Working | 18 services running on Kubernetes |
| ML Training Pipeline | ⏳ Testing in progress | Code deployed, end-to-end test pending |
| Inference API (predictions) | 📋 Next up | Deploys after training pipeline is validated |

---

## How It Works

The platform has three automated pipelines that run in sequence:

### 1. ETL Pipeline — runs hourly
Fetches raw market data (price, volume, trades) for 10 cryptocurrencies from Binance, saves the raw files, and loads the cleaned records into a database.

```
Binance API  →  MinIO (raw file storage)  →  PostgreSQL (database)
```

### 2. ML Training Pipeline — runs nightly at 2 AM
Reads the database, computes 80+ features (moving averages, momentum indicators, etc.), trains two model types (LightGBM and XGBoost) for each symbol, and registers the best models.

```
PostgreSQL  →  Feature Engineering  →  Model Training  →  MLflow (model registry)
```

### 3. Inference API — always running
Loads the latest approved models from MLflow and serves predictions over HTTP. Scales automatically under load.

```
HTTP Request  →  Feature Generation  →  Model Prediction  →  HTTP Response
```

All three pipelines are orchestrated by **Apache Airflow**, which schedules, monitors, and retries tasks automatically.

---

## Quick Start

### Option A: Docker Compose (Easiest — no Kubernetes needed)

Good for exploring the system locally.

**Prerequisites:** Docker Desktop with at least 6GB RAM allocated

```bash
git clone <repo-url>
cd ml-eng-with-ops

# Start all services
docker-compose up -d

# Run the end-to-end demo (trains a model and makes predictions)
pip install mlflow scikit-learn requests minio
python3 scripts/demo-e2e-workflow.py
```

**Open the dashboards:**

| Service | URL | Login |
|---|---|---|
| MLflow (model registry) | http://localhost:5001 | no login |
| Grafana (monitoring) | http://localhost:3000 | admin / admin |
| MinIO (file storage) | http://localhost:9001 | admin / admin123 |
| API docs | http://localhost:8000/docs | no login |
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

# Deploy all infrastructure (takes 5-10 minutes)
./scripts/k8s-bootstrap.sh --infra-only

# Check everything is running
./scripts/k8s-bootstrap.sh --status
```

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
# Open: http://localhost:3000  (admin / prom-operator)

# MinIO — file storage console
kubectl port-forward -n ml-pipeline svc/minio 9001:9001
# Open: http://localhost:9001  (admin / admin123)
```

---

## Running the ETL Pipeline

The ETL pipeline fetches 2.25 million crypto records across 10 symbols. It has been fully tested.

**Via Airflow UI:**
1. Open http://localhost:8080
2. Find the `etl_crypto_data_pipeline` DAG
3. Toggle it to "On", then click the play button (▶)
4. Watch tasks complete in the Grid view (~15-20 minutes)

**Via command line:**
```bash
kubectl exec -n ml-pipeline deployment/airflow-scheduler -- \
  airflow dags trigger etl_crypto_data_pipeline
```

**Verify data loaded:**
```bash
kubectl exec -n ml-pipeline postgresql-0 -- \
  psql -U postgres -d crypto -c \
  "SELECT symbol, COUNT(*) FROM crypto_data GROUP BY symbol ORDER BY symbol;"
```

Expected output (~2.25M total records across 10 symbols):
```
   symbol    |  count
-------------+--------
 ADAUSDT     | 250000
 AVAXUSDT    | 186728   ← less history available on Binance
 BNBUSDT     | 250000
 BTCUSDT     | 250000
 DOTUSDT     | 250000
 ETHUSDT     | 250000
 LINKUSDT    | 250000
 MATICUSDT   | 250000
 SOLUSDT     | 250000
 XRPUSDT     | 250000
```

---

## Running the ML Training Pipeline

> Status: Testing in progress

```bash
# Create the training resource pool first (one-time setup)
kubectl exec -n ml-pipeline deployment/airflow-scheduler -- \
  airflow pools set ml_training_pool 3 "ML training pool"

# Trigger the training pipeline
kubectl exec -n ml-pipeline deployment/airflow-scheduler -- \
  airflow dags trigger ml_training_pipeline
```

**What it trains:**
- 10 symbols × 3 prediction tasks × 2 algorithms = **60 models**
- Tasks: short-term return, medium-term return, direction (up/down)
- Algorithms: LightGBM and XGBoost
- All models versioned and tracked in MLflow

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

---

## Project Structure

```
ml-eng-with-ops/
├── dags/                        # Pipeline code (runs inside Airflow)
│   ├── airflow_dags/            # Pipeline definitions (ETL + ML training)
│   ├── etl/                     # Data extraction and loading logic
│   ├── feature_eng.py           # Feature engineering (80+ indicators)
│   ├── model_training.py        # Model training logic
│   └── model_promotion.py       # Promote models to production
│
├── app/
│   └── production_api.py        # FastAPI inference service
│
├── scripts/
│   ├── k8s-bootstrap.sh         # One-command Kubernetes deployment
│   ├── demo-e2e-workflow.py     # End-to-end demo
│   └── demo-etl-pipeline.sh    # ETL-only demo
│
├── docker/
│   ├── Dockerfile.airflow       # Custom Airflow image (includes DAGs)
│   └── Dockerfile.inference     # Inference API image
│
├── k8s/                         # Kubernetes config for the inference API
├── airflow/                     # Airflow configuration (Helm values)
├── monitoring/                  # Grafana dashboards and Prometheus config
├── examples/                    # Generic ML use case templates
├── docker-compose.yml           # Local development stack
└── .env.example                 # Environment variable template
```

---

## Tech Stack

| Tool | What it does in plain English |
|---|---|
| **Apache Airflow** | Scheduler — runs pipelines on a schedule, retries failures, shows status |
| **MLflow** | Model registry — tracks experiments, versions models, manages deployment stages |
| **PostgreSQL** | Main database — stores all the structured data |
| **MinIO** | File storage — stores raw data files and model artifacts (like AWS S3, but local) |
| **Redis** | Cache — speeds up repeated lookups during inference |
| **FastAPI** | Web framework — exposes predictions via HTTP API |
| **LightGBM / XGBoost** | ML algorithms — the actual models that make predictions |
| **Prometheus** | Metrics collection — records numbers over time (request counts, latencies, etc.) |
| **Grafana** | Dashboards — visualizes the Prometheus metrics |
| **Kubernetes** | Container orchestration — runs and scales all the services |
| **Helm** | Package manager for Kubernetes — simplifies deploying complex service stacks |
| **Docker** | Containerization — packages each service with its dependencies |

---

## Glossary

**DAG** (Directed Acyclic Graph) — In Airflow, a DAG is a pipeline definition. It describes which tasks to run, in what order, and how often.

**MLflow** — A tool that tracks ML experiments (which parameters were used, what metrics were achieved) and stores model versions in a registry so you can deploy specific versions.

**MinIO** — An open-source file storage system compatible with Amazon S3. Used here to store raw data files and model artifacts.

**KubernetesExecutor** — An Airflow setting that runs each pipeline task inside its own isolated container (pod), then deletes it when done. Saves resources and avoids conflicts.

**HPA (Horizontal Pod Autoscaler)** — A Kubernetes feature that automatically adds more copies of a service when traffic increases, and removes them when traffic drops.

**MLOps** — The practice of applying software engineering principles (automation, monitoring, versioning) to machine learning systems.

**OHLCV** — Open, High, Low, Close, Volume. Standard fields in financial market data representing price and trading activity for a time period.

**Feature engineering** — The process of transforming raw data into inputs that a model can learn from. For example, turning raw price data into "14-day moving average" or "momentum over 7 days".

**Inference** — Using a trained model to make predictions on new data. As opposed to *training*, which is the process of fitting the model.

**Upsert** — A database operation that inserts a new row if it doesn't exist, or updates it if it does. Prevents duplicate records on repeated runs.

---

## Going to Production

This setup runs on Docker Desktop's single-node Kubernetes, which is fine for learning and development. For production:

- Replace MinIO with cloud object storage (AWS S3, GCP Cloud Storage, Azure Blob)
- Replace local PostgreSQL with a managed database (AWS RDS, GCP Cloud SQL)
- Use a cloud container registry (AWS ECR, GCP Artifact Registry) instead of `localhost:5050`
- Configure remote log storage for Airflow (logs disappear when pods are deleted by default)
- Enable TLS and secrets management

See [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md) for detailed production recommendations.

For common errors and fixes, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) (Airflow 3.0 specific issues, MinIO connection errors, database schema mismatches, and more).

---

## Contributing

Contributions are welcome. Areas where help is most needed:

- Testing the ML training pipeline end-to-end
- Adding pytest test coverage
- Cloud provider deployment examples (AWS EKS, GCP GKE, Azure AKS)
- Additional Grafana dashboards
- Support for PyTorch and TensorFlow models

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

[FastAPI](https://fastapi.tiangolo.com/) · [MLflow](https://mlflow.org/) · [Apache Airflow](https://airflow.apache.org/) · [Prometheus](https://prometheus.io/) · [Grafana](https://grafana.com/) · [Kubernetes](https://kubernetes.io/) · [MinIO](https://min.io/) · [Redis](https://redis.io/) · [LightGBM](https://lightgbm.readthedocs.io/) · [XGBoost](https://xgboost.readthedocs.io/)
