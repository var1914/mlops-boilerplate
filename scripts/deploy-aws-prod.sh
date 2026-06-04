#!/bin/bash
# =============================================================================
# Production AWS deployment for mlops-boilerplate
# =============================================================================
#
# This script orchestrates AWS production deploy in ordered steps. Terraform
# provisions the platform (VPC, EKS, RDS, ECR, IAM, Helm addons, K8s secrets);
# later steps build app images and deploy Airflow + the inference API.
#
# Usage:
#   ./scripts/deploy-aws-prod.sh bootstrap
#   ./scripts/deploy-aws-prod.sh plan [dev|prod]
#   ./scripts/deploy-aws-prod.sh infra [dev|prod]
#   ./scripts/deploy-aws-prod.sh databases [dev|prod]
#   ./scripts/deploy-aws-prod.sh build-push [dev|prod]
#   ./scripts/deploy-aws-prod.sh apps [dev|prod]
#   ./scripts/deploy-aws-prod.sh all [dev|prod]
#   ./scripts/deploy-aws-prod.sh outputs [dev|prod]
#
# Environment:
#   AWS_REGION / TF_VAR_region     (default: us-east-1) — must match Terraform region
#   TF_AUTO_APPROVE=1               skip apply prompt (CI / unattended only)
#   TF_VAR_acme_email               required for TLS (prod)
#   IMAGE_TAG                       (default: latest)
#   AIRFLOW_CHART_VERSION           (default: 1.18.0)
#
# Order: bootstrap (once) → infra → databases → build-push → apps
# =============================================================================

set -euo pipefail

# --- Terminal colors for log helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# --- Runtime configuration (overridable via env) ---
AWS_REGION="${AWS_REGION:-${TF_VAR_region:-us-east-1}}"
ENV="${2:-prod}"                                    # second CLI arg: dev or prod
IMAGE_TAG="${IMAGE_TAG:-latest}"                    # ECR tag for both images
AIRFLOW_CHART_VERSION="${AIRFLOW_CHART_VERSION:-1.18.0}"

# K8s namespaces: ml-pipeline = API + Terraform secrets; airflow = Helm release
NAMESPACE="ml-pipeline"
AIRFLOW_NAMESPACE="airflow"

# Repo paths (script lives in scripts/, root is parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TF_ENV_DIR="$ROOT_DIR/terraform/environments/$ENV"
TF_BOOTSTRAP_DIR="$ROOT_DIR/terraform/bootstrap"

# RDS password JSON from Secrets Manager — fetched once per command, then reused
_RDS_CREDS_JSON=""

# --- Logging helpers (all write to stderr so command substitution stays clean) ---
log_info()    { echo -e "${BLUE}[INFO]${NC} $1" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" >&2; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_step() {
  echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
  echo -e "${CYAN}  $1${NC}" >&2
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" >&2
}

# Ensure ENV is dev|prod and the matching terraform/environments/* directory exists
validate_env() {
  case "$ENV" in
    dev|prod) ;;
    *)
      log_error "Invalid environment '$ENV' (use dev or prod)"
      exit 1
      ;;
  esac
  if [ ! -d "$TF_ENV_DIR" ]; then
    log_error "Missing $TF_ENV_DIR"
    exit 1
  fi
}

# Fail fast if a required CLI tool is not installed
require_cmd() {
  for c in "$@"; do
    command -v "$c" >/dev/null || { log_error "Missing required command: $c"; exit 1; }
  done
}

# True after a successful `infra` apply — we use eks_cluster_name as a proxy for state
require_tf_state() {
  if ! terraform -chdir="$TF_ENV_DIR" output -raw eks_cluster_name &>/dev/null; then
    log_error "No Terraform state in $TF_ENV_DIR (or outputs missing)"
    log_info "Run: $0 infra $ENV"
    exit 1
  fi
}

# Read a single Terraform output; exit with a clear message if infra was not applied
tf_output() {
  local name=$1 value
  if ! value=$(terraform -chdir="$TF_ENV_DIR" output -raw "$name" 2>&1); then
    log_error "Terraform output '$name' unavailable: $value"
    log_info "Run: $0 infra $ENV"
    exit 1
  fi
  if [ -z "$value" ]; then
    log_error "Terraform output '$name' is empty"
    exit 1
  fi
  printf '%s' "$value"
}

# Pull master RDS credentials from AWS Secrets Manager (created by Terraform RDS module)
rds_credentials_json() {
  if [ -z "$_RDS_CREDS_JSON" ]; then
    local secret_arn
    secret_arn=$(tf_output rds_secret_arn)
    _RDS_CREDS_JSON=$(aws secretsmanager get-secret-value \
      --secret-id "$secret_arn" \
      --region "$AWS_REGION" \
      --query SecretString \
      --output text)
  fi
  printf '%s' "$_RDS_CREDS_JSON"
}

# Extract username, password, or dbname from the secret JSON (Python avoids shell quoting bugs)
rds_field() {
  local field=$1
  rds_credentials_json | python3 -c "import sys,json; print(json.load(sys.stdin)[sys.argv[1]], end='')" "$field"
}

# When set, Terraform applies without a confirmation prompt (CI / automation only)
tf_auto_approve_enabled() {
  [ "${TF_AUTO_APPROVE:-}" = "1" ] || [ "${TF_AUTO_APPROVE:-}" = "true" ]
}

# Ask the user to accept or deny a saved plan (stdin must be a TTY).
confirm_terraform_apply() {
  local label=${1:-apply}
  if ! [ -t 0 ]; then
    log_error "Cannot prompt for Terraform approval (stdin is not a terminal)."
    log_info "Run this command in an interactive shell, or set TF_AUTO_APPROVE=1 to skip the prompt."
    return 1
  fi
  echo "" >&2
  read -r -p "Apply Terraform plan for ${label}? [y/N]: " reply </dev/tty || return 1
  case "$reply" in
    y|Y|yes|YES)
      return 0
      ;;
    *)
      log_warn "Terraform apply declined — no changes applied."
      return 1
      ;;
  esac
}

# Plan first, then apply. Default: user must confirm (y/N). TF_AUTO_APPROVE=1 skips the prompt.
terraform_apply() {
  local dir=$1 label=${2:-apply}
  local plan_file="$dir/tfplan"

  log_info "terraform plan ($label)"
  terraform -chdir="$dir" plan -input=false -out=tfplan

  if tf_auto_approve_enabled; then
    log_info "TF_AUTO_APPROVE=1 — applying without confirmation"
    terraform -chdir="$dir" apply -input=false -auto-approve tfplan
    rm -f "$plan_file"
    return 0
  fi

  log_info "Review the plan above before confirming."
  if ! confirm_terraform_apply "$label"; then
    rm -f "$plan_file"
    exit 1
  fi
  terraform -chdir="$dir" apply -input=false tfplan
  rm -f "$plan_file"
}

# Point kubectl at the EKS cluster from Terraform outputs
ensure_kubeconfig() {
  require_tf_state
  local cluster
  cluster=$(tf_output eks_cluster_name)
  aws eks update-kubeconfig --name "$cluster" --region "$AWS_REGION" >/dev/null
  kubectl cluster-info >/dev/null
  log_success "kubeconfig → cluster $cluster ($AWS_REGION)"
}

# Create namespace if missing (idempotent)
ensure_namespace() {
  kubectl create namespace "$1" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
}

# ECR repo URL is registry/account/repo — login only needs the registry host
ecr_registry_from_repo() {
  echo "${1%%/*}"
}

ecr_login() {
  aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$1"
}

# -----------------------------------------------------------------------------
# bootstrap — one-time remote state (S3 bucket + DynamoDB lock table)
# Does not use ENV; configure environments/*/provider.tf backend to match output.
# -----------------------------------------------------------------------------
cmd_bootstrap() {
  log_step "Bootstrap Terraform remote state"
  require_cmd terraform aws
  terraform -chdir="$TF_BOOTSTRAP_DIR" init -input=false
  terraform_apply "$TF_BOOTSTRAP_DIR" "bootstrap"
  log_success "Bootstrap complete. Ensure environments/*/provider.tf backend matches the bucket."
}

# -----------------------------------------------------------------------------
# plan — show infrastructure changes without applying (writes tfplan for review)
# -----------------------------------------------------------------------------
cmd_plan() {
  log_step "Terraform plan ($ENV)"
  validate_env
  require_cmd terraform aws
  terraform -chdir="$TF_ENV_DIR" init -input=false
  terraform -chdir="$TF_ENV_DIR" plan -input=false -out=tfplan
  log_success "Plan saved: $TF_ENV_DIR/tfplan"
  log_info "Apply after review: terraform -chdir=$TF_ENV_DIR apply tfplan"
}

# -----------------------------------------------------------------------------
# infra — full platform via Terraform (VPC, EKS, RDS, Redis, S3, ECR, IRSA,
# Istio, cert-manager, MLflow, monitoring, K8s ConfigMaps/secrets in cluster)
# -----------------------------------------------------------------------------
cmd_infra() {
  log_step "Apply Terraform ($ENV)"
  validate_env
  require_cmd terraform aws kubectl helm python3 docker
  if [ ! -f "$TF_ENV_DIR/terraform.tfvars" ] && [ -z "${TF_VAR_acme_email:-}" ]; then
    log_warn "No terraform.tfvars and TF_VAR_acme_email unset — cert-manager may fail"
  fi
  terraform -chdir="$TF_ENV_DIR" init -input=false
  terraform_apply "$TF_ENV_DIR" "$ENV"
  # Infra may rotate RDS secret; drop cache so later steps read fresh credentials
  _RDS_CREDS_JSON=""
  ensure_kubeconfig
  terraform -chdir="$TF_ENV_DIR" output
}

# -----------------------------------------------------------------------------
# databases — RDS ships with one DB (mlflow). Airflow and the API need airflow
# and crypto. We run a short-lived Job inside the cluster so psql reaches RDS
# on the private network (same as node → RDS security group rules).
# -----------------------------------------------------------------------------
cmd_databases() {
  log_step "Initialize RDS databases (airflow, crypto)"
  validate_env
  require_cmd kubectl aws python3
  require_tf_state
  ensure_kubeconfig
  ensure_namespace "$NAMESPACE"

  local rds_host rds_user rds_connect_db
  rds_host=$(tf_output rds_endpoint)
  rds_user=$(rds_field username)
  # Connect to the instance default DB (mlflow), not the template "postgres" DB
  rds_connect_db=$(rds_field dbname)

  log_info "RDS $rds_host — connect via database '$rds_connect_db', create airflow + crypto"

  # Temp secret for the Job only; deleted after success or failure
  kubectl create secret generic rds-init-creds \
    --namespace "$NAMESPACE" \
    --from-literal=password="$(rds_field password)" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null

  local job_name
  job_name="rds-init-$(date +%s)"
  # Job runs psql from inside EKS; CREATE DATABASE is skipped if DB already exists
  kubectl apply -n "$NAMESPACE" -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${job_name}
spec:
  ttlSecondsAfterFinished: 300
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: psql
        image: postgres:15-alpine
        env:
        - name: PGPASSWORD
          valueFrom:
            secretKeyRef:
              name: rds-init-creds
              key: password
        command:
        - /bin/sh
        - -ec
        - |
          set -e
          for db in airflow crypto; do
            exists=\$(psql -h "${rds_host}" -U "${rds_user}" -d "${rds_connect_db}" -tAc \\
              "SELECT 1 FROM pg_database WHERE datname='\${db}'")
            if [ "\$exists" = "1" ]; then
              echo "Database \${db} already exists"
              continue
            fi
            psql -h "${rds_host}" -U "${rds_user}" -d "${rds_connect_db}" -v ON_ERROR_STOP=1 \\
              -c "CREATE DATABASE \${db};"
            echo "Created database \${db}"
          done
EOF

  if ! kubectl wait --for=condition=complete "job/${job_name}" -n "$NAMESPACE" --timeout=300s; then
    log_error "RDS init job failed:"
    kubectl logs "job/${job_name}" -n "$NAMESPACE" --tail=50 || true
    kubectl delete secret rds-init-creds -n "$NAMESPACE" --ignore-not-found >/dev/null
    exit 1
  fi
  kubectl delete job "$job_name" -n "$NAMESPACE" --ignore-not-found >/dev/null
  kubectl delete secret rds-init-creds -n "$NAMESPACE" --ignore-not-found >/dev/null
  log_success "RDS databases: ${rds_connect_db} (default), airflow, crypto"
}

# -----------------------------------------------------------------------------
# build-push — build Docker images locally and push to ECR repos from Terraform
# -----------------------------------------------------------------------------
cmd_build_push() {
  log_step "Build and push images ($ENV)"
  validate_env
  require_cmd docker aws python3
  require_tf_state

  local api_repo airflow_repo
  api_repo=$(tf_output ecr_inference_api_repository_url)
  airflow_repo=$(tf_output ecr_airflow_repository_url)

  ecr_login "$(ecr_registry_from_repo "$api_repo")"

  for f in "$ROOT_DIR/docker/Dockerfile.inference" "$ROOT_DIR/docker/Dockerfile.airflow"; do
    [ -f "$f" ] || { log_error "Missing $f"; exit 1; }
  done

  log_info "Pushing ${api_repo}:${IMAGE_TAG}"
  docker build -t "${api_repo}:${IMAGE_TAG}" -f "$ROOT_DIR/docker/Dockerfile.inference" "$ROOT_DIR"
  docker push "${api_repo}:${IMAGE_TAG}"

  log_info "Pushing ${airflow_repo}:${IMAGE_TAG}"
  docker build -t "${airflow_repo}:${IMAGE_TAG}" -f "$ROOT_DIR/docker/Dockerfile.airflow" "$ROOT_DIR"
  docker push "${airflow_repo}:${IMAGE_TAG}"

  log_success "Images pushed"
}

# Fill airflow/values-aws-prod.yaml placeholders (RDS, ECR, IRSA, S3 logs bucket).
# Returns path to a temp file; caller must delete after Helm consumes it.
render_airflow_values() {
  local out values_src="$ROOT_DIR/airflow/values-aws-prod.yaml"
  out=$(mktemp)

  _RDS_CREDS_JSON=""
  VALUES_SRC="$values_src" VALUES_OUT="$out" \
  REPLACE_RDS_HOST="$(tf_output rds_endpoint)" \
  REPLACE_RDS_USERNAME="$(rds_field username)" \
  REPLACE_RDS_PASSWORD="$(rds_field password)" \
  REPLACE_AIRFLOW_ECR_REPOSITORY="$(tf_output ecr_airflow_repository_url)" \
  REPLACE_AIRFLOW_IRSA_ROLE_ARN="$(tf_output airflow_irsa_role_arn)" \
  REPLACE_AIRFLOW_LOGS_BUCKET="$(terraform -chdir="$TF_ENV_DIR" output -json s3_buckets | python3 -c "import sys,json; print(json.load(sys.stdin)['airflow_logs'])")" \
  python3 <<'PY'
import os, pathlib

text = pathlib.Path(os.environ["VALUES_SRC"]).read_text()
password = os.environ["REPLACE_RDS_PASSWORD"]

def replace(key):
    global text
    text = text.replace(key, os.environ[key])

for key in (
    "REPLACE_RDS_HOST",
    "REPLACE_RDS_USERNAME",
    "REPLACE_AIRFLOW_ECR_REPOSITORY",
    "REPLACE_AIRFLOW_IRSA_ROLE_ARN",
    "REPLACE_AIRFLOW_LOGS_BUCKET",
):
    replace(key)

# RDS passwords often contain YAML-special chars; wrap in double quotes
pw_yaml = '"' + password.replace("\\", "\\\\").replace('"', '\\"') + '"'
text = text.replace("REPLACE_RDS_PASSWORD", pw_yaml)

pathlib.Path(os.environ["VALUES_OUT"]).write_text(text)
PY

  echo "$out"
}

# Helm generates airflow-jwt-secret with a random value; task pods use JWT from
# airflow-runtime-secrets (Terraform). Patch Helm secret to match, then restart
# scheduler + api-server so tokens verify (see TROUBLESHOOTING.md).
sync_airflow_jwt_secret() {
  local jwt_b64
  jwt_b64=$(kubectl get secret airflow-runtime-secrets -n "$AIRFLOW_NAMESPACE" \
    -o jsonpath='{.data.AIRFLOW__API_AUTH__JWT_SECRET}' 2>/dev/null || true)
  [ -n "$jwt_b64" ] || { log_warn "airflow-runtime-secrets missing — skip JWT sync"; return 0; }

  if ! kubectl get secret airflow-jwt-secret -n "$AIRFLOW_NAMESPACE" &>/dev/null; then
    log_warn "airflow-jwt-secret not created yet — skip JWT sync"
    return 0
  fi

  kubectl patch secret airflow-jwt-secret -n "$AIRFLOW_NAMESPACE" --type=json \
    -p="[{\"op\":\"replace\",\"path\":\"/data/jwt-secret\",\"value\":\"${jwt_b64}\"}]"
  log_info "Synced airflow-jwt-secret from Terraform runtime secret"
  kubectl rollout restart deployment/airflow-scheduler deployment/airflow-api-server \
    -n "$AIRFLOW_NAMESPACE" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# apps — Helm installs Airflow (RDS + S3 logs + IRSA); kubectl applies API
# manifests that reference Terraform-created ConfigMap/Secret in ml-pipeline.
# Requires: infra, databases, build-push (images must exist in ECR).
# -----------------------------------------------------------------------------
cmd_apps() {
  log_step "Deploy Airflow + inference API ($ENV)"
  validate_env
  require_cmd helm kubectl aws python3
  require_tf_state
  ensure_kubeconfig
  ensure_namespace "$NAMESPACE"
  ensure_namespace "$AIRFLOW_NAMESPACE"

  # API deployment envFrom these objects (created by Terraform kubernetes module)
  if ! kubectl get configmap ml-pipeline-config -n "$NAMESPACE" &>/dev/null; then
    log_error "ml-pipeline-config missing — run: $0 infra $ENV"
    exit 1
  fi
  if ! kubectl get secret ml-pipeline-secrets -n "$NAMESPACE" &>/dev/null; then
    log_error "ml-pipeline-secrets missing — run: $0 infra $ENV"
    exit 1
  fi

  local api_repo rendered_values
  api_repo=$(tf_output ecr_inference_api_repository_url)
  rendered_values=$(render_airflow_values)
  trap 'rm -f "$rendered_values"' RETURN

  helm repo add apache-airflow https://airflow.apache.org 2>/dev/null || true
  helm repo update

  # Chart runs migrateDatabaseJob against metadataConnection.db=airflow
  log_info "Helm: airflow chart ${AIRFLOW_CHART_VERSION}"
  helm upgrade --install airflow apache-airflow/airflow \
    --namespace "$AIRFLOW_NAMESPACE" \
    --create-namespace \
    --version "$AIRFLOW_CHART_VERSION" \
    --values "$rendered_values" \
    --set "images.airflow.tag=${IMAGE_TAG}" \
    --wait --timeout 25m

  sync_airflow_jwt_secret

  # API: substitute ECR image in k8s/prod template; Service + optional HPA
  log_info "Deploying inference API"
  sed "s|REPLACE_ECR_IMAGE|${api_repo}:${IMAGE_TAG}|g" \
    "$ROOT_DIR/k8s/prod/api-deployment.yaml" | kubectl apply -f -
  kubectl apply -f "$ROOT_DIR/k8s/prod/api-service.yaml"
  if [ -f "$ROOT_DIR/k8s/api-hpa.yaml" ]; then
    kubectl apply -f "$ROOT_DIR/k8s/api-hpa.yaml"
  else
    log_warn "HPA skipped (k8s/api-hpa.yaml not found)"
  fi

  kubectl rollout status deployment/crypto-prediction-api -n "$NAMESPACE" --timeout=10m

  local api_url="(set ingress_api_host in terraform.tfvars)"
  if api_url=$(terraform -chdir="$TF_ENV_DIR" output -raw ingress_api_url 2>/dev/null) && [ -n "$api_url" ]; then
    :
  fi
  log_success "Applications deployed"
  log_info "API: $api_url"
}

# Print all Terraform outputs (URLs, ARNs, bucket map, etc.)
cmd_outputs() {
  validate_env
  require_tf_state
  terraform -chdir="$TF_ENV_DIR" output
}

usage() {
  cat <<EOF
Usage: $0 <command> [dev|prod]

Commands (run in order after bootstrap):
  bootstrap     S3 + DynamoDB for Terraform state (once)
  plan          Save tfplan only (no apply)
  infra         Terraform plan + prompt to apply (y/N)
  databases     CREATE DATABASE airflow, crypto on RDS
  build-push    docker build → ECR
  apps          Helm (Airflow) + kubectl (API)
  all           infra → databases → build-push → apps (infra prompts for apply)
  outputs       Print terraform outputs

Environment:
  ENV=${ENV}  AWS_REGION=${AWS_REGION}  IMAGE_TAG=${IMAGE_TAG}
  TF_AUTO_APPROVE=${TF_AUTO_APPROVE:-0}   set to 1 to skip the apply prompt

Examples:
  $0 plan prod
  $0 infra prod                              # plan, then [y/N] to apply
  $0 all prod                                # same prompt during infra step
  TF_AUTO_APPROVE=1 $0 all prod              # CI: apply without prompt
EOF
}

# Dispatch first CLI argument to the matching command handler
main() {
  case "${1:-}" in
    bootstrap) cmd_bootstrap ;;
    plan)      cmd_plan ;;
    infra)     cmd_infra ;;
    databases) cmd_databases ;;
    build-push) cmd_build_push ;;
    apps)      cmd_apps ;;
    all)
      # infra runs terraform plan + user confirm; later steps run only if apply succeeds
      log_info "Pipeline: infra (you will confirm apply) → databases → build-push → apps"
      cmd_infra
      cmd_databases
      cmd_build_push
      cmd_apps
      ;;
    outputs) cmd_outputs ;;
    -h|--help) usage ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
