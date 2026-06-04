# MLOps Boilerplate — AWS Production Infrastructure

Production AWS deployment for the crypto ML pipeline.

There is **no** root-level `terraform.tfvars` — each environment has its own `terraform.tfvars` under `environments/{dev,prod}/`.

Application images are deployed **after** Terraform via `./scripts/deploy-aws-prod.sh`.

## Prerequisites

- AWS CLI, Terraform >= 1.5, kubectl, Helm 3, Docker
- DNS control for your API hostname (for TLS)

## Quick start

### 1. Bootstrap remote state (once)

```bash
./scripts/deploy-aws-prod.sh bootstrap
```

### 2. Configure environment variables

Terraform variables are set per environment. Example files list every variable: **required** lines are uncommented; **optional** lines are commented with defaults from `environments/{dev,prod}/variables.tf` (uncomment only to override).

**Bootstrap** (optional — all variables have defaults in `bootstrap/variables.tf`):

```bash
cd terraform/bootstrap
# cp terraform.tfvars.example terraform.tfvars   # only if overriding bucket/region names
terraform init && terraform apply
```

**Development or production:**

```bash
cd terraform/environments/dev    # or prod
cp terraform.tfvars.example terraform.tfvars
```

Edit at minimum: `acme_email`. For production, uncomment and set `ingress_api_host` if you are not using the default hostname. You do not need to fill in every optional block.

**Alternative — environment variables instead of `terraform.tfvars`:**

`./scripts/deploy-aws-prod.sh infra dev` and `infra prod` run Terraform from the matching environment directory and pick up `terraform.tfvars` if present.

### 3. Deploy everything

**One-shot pipeline** (you approve Terraform before later steps run):

```bash
./scripts/deploy-aws-prod.sh all prod
```

During `infra`, the script runs `terraform plan`, shows the diff, then asks **`Apply Terraform plan for prod? [y/N]`**. Answer `y` to apply and continue with databases → ECR → apps; anything else stops the pipeline with no apply.

**Unattended / CI** (skip the prompt):

```bash
TF_AUTO_APPROVE=1 ./scripts/deploy-aws-prod.sh all prod
```

**Step-by-step** (same apply prompt on `infra` / `bootstrap`):

```bash
./scripts/deploy-aws-prod.sh infra prod
./scripts/deploy-aws-prod.sh databases prod
./scripts/deploy-aws-prod.sh build-push prod
./scripts/deploy-aws-prod.sh apps prod
```

### 4. Point DNS

```bash
kubectl get svc -n istio-system istio-ingressgateway \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}{"\n"}'
```

Create a CNAME: `api.yourdomain.com` → NLB hostname.

## Ingress (Istio)

Terraform creates:

- **Gateway** + TLS certificate (cert-manager) on `istio-ingressgateway`
- **VirtualService** routing `ingress_api_host` → `crypto-prediction-api.ml-pipeline:8000`

After apply, point DNS at the NLB (see step 4 below). The API Deployment must exist (`deploy-aws-prod.sh apps`) and the namespace must have Istio injection enabled (default for `ml-pipeline`).

## Step-by-step commands

```bash
# Infrastructure only (plan + [y/N] to apply)
./scripts/deploy-aws-prod.sh infra prod

# Create airflow + crypto databases on RDS (after infra)
./scripts/deploy-aws-prod.sh databases prod

# Build & push to ECR
./scripts/deploy-aws-prod.sh build-push prod

# Deploy Airflow (S3 logs, RDS, IRSA) + inference API
./scripts/deploy-aws-prod.sh apps prod

# Show outputs (ECR URLs, IRSA ARNs, bucket names)
./scripts/deploy-aws-prod.sh outputs prod
```


## What Terraform provisions

| Component       | AWS service                     |
|-----------------|--------------------------------|
| Compute         | EKS 1.31                       |
| Database        | RDS PostgreSQL 15              |
| Cache           | ElastiCache Redis 7            |
| Object storage  | S3 (6 buckets)                 |
| Images          | ECR                            |
| Secrets         | Secrets Manager + K8s secrets  |
| Auth            | IRSA                           |
| Ingress TLS     | Istio NLB + cert-manager       |
| Airflow         | Helm on EKS (KubernetesExecutor)|
| Airflow logs    | S3 remote logging              |
| MLflow          | Helm on EKS                    |
| Monitoring      | Prometheus + Grafana + Loki    |

### S3 buckets

| Bucket suffix | Purpose |
|---------------|---------|
| `mlflow-artifacts-{env}` | MLflow model artifacts |
| `crypto-raw-data-{env}` | Raw Binance extracts |
| `crypto-features-{env}` | Engineered features |
| `crypto-models-{env}` | Model binaries |
| `crypto-data-versions-{env}` | DVC/data versioning |
| `airflow-logs-{env}` | Airflow task logs (remote logging) |

## Directory layout

```
terraform/
├── bootstrap/              # Remote state (S3 + DynamoDB) — run once
├── environments/
│   ├── dev/
│   │   ├── terraform.tfvars.example   # copy → terraform.tfvars
│   │   └── variables.tf
│   └── prod/
│       ├── terraform.tfvars.example
│       └── variables.tf
└── modules/
    ├── vpc, eks, rds, elasticache-redis, s3, ecr, iam
    ├── helm/               # Istio, cert-manager, MLflow, monitoring
    └── kubernetes/         # Secrets, IRSA service accounts, TLS ingress
```

## Secrets management

| Secret | Storage | Used by |
|--------|---------|---------|
| RDS credentials | AWS Secrets Manager | CI/CD, `deploy-aws-prod.sh databases` |
| Airflow Fernet/JWT keys | K8s secret `airflow-runtime-secrets` | Airflow pods |
| DB password for API | K8s secret `ml-pipeline-secrets` | Inference API |
| TLS certificates | cert-manager → K8s secret | Istio gateway |

RDS password is generated by Terraform and stored in Secrets Manager (`{project}-{env}-rds-credentials`).

## Production vs local

| Setting | Local (`k8s-bootstrap.sh`) | Production (Terraform) |
|---------|---------------------------|------------------------|
| Object storage | MinIO `admin/admin123` | S3 via IRSA |
| PostgreSQL | in-cluster Helm | RDS Multi-AZ |
| Redis | in-cluster Helm | ElastiCache |
| API image | `localhost:5050/crypto-api` | ECR `inference-api` |
| Airflow logs | PVC on node | S3 `airflow-logs-{env}` |
| TLS | none | Let's Encrypt via cert-manager |
| Ingress | port-forward | Istio NLB |

## Related files

| File | Purpose |
|------|---------|
| `scripts/deploy-aws-prod.sh` | Orchestrates Terraform + Helm + kubectl |
| `airflow/values-aws-prod.yaml` | Airflow Helm template (S3 logs, RDS, IRSA) |
| `k8s/prod/api-deployment.yaml` | Production API manifest (ConfigMap + IRSA) |
| `k8s/` (unchanged) | Local Docker Desktop deployment |

## Cleanup

```bash
cd terraform/environments/prod
terraform destroy
```

> Prod RDS has deletion protection enabled. Disable in `terraform.tfvars` before destroy if needed.
