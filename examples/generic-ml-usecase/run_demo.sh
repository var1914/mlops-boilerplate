#!/bin/bash
#
# End-to-End Demo: Generic ML Use Case
#
# This script demonstrates the complete ML workflow:
# 1. Generate sample data
# 2. Train model
# 3. Register to MLflow
# 4. Promote to Production
# 5. Test inference via API
#
# Usage:
#   ./run_demo.sh                      # Run with defaults (demand forecasting)
#   ./run_demo.sh churn_prediction     # Run with churn data
#   ./run_demo.sh fraud_detection      # Run with fraud data
#   ./run_demo.sh --help               # Show help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
USE_CASE="${1:-demand_forecasting}"
SAMPLES=5000
MLFLOW_URL="${MLFLOW_URL:-http://localhost:5001}"
API_URL="${API_URL:-http://localhost:8000}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

show_help() {
    cat << EOF
End-to-End Demo: Generic ML Use Case

Usage:
    ./run_demo.sh [USE_CASE] [OPTIONS]

Use Cases:
    demand_forecasting    Retail demand prediction (regression)
    churn_prediction      Customer churn (classification)
    fraud_detection       Transaction fraud (classification)
    price_optimization    Dynamic pricing (regression)
    generic               Generic tabular data

Environment Variables:
    MLFLOW_URL    MLflow tracking server (default: http://localhost:5001)
    API_URL       Inference API URL (default: http://localhost:8000)
    SAMPLES       Number of samples to generate (default: 5000)

Examples:
    # Run demand forecasting demo
    ./run_demo.sh demand_forecasting

    # Run churn prediction with custom MLflow
    MLFLOW_URL=http://mlflow:5000 ./run_demo.sh churn_prediction

    # Run fraud detection demo
    ./run_demo.sh fraud_detection
EOF
}

# Handle --help
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
    exit 0
fi

# Determine task type based on use case
case "$USE_CASE" in
    demand_forecasting|price_optimization|generic)
        TASK_TYPE="regression"
        TARGET_COL=$([ "$USE_CASE" == "demand_forecasting" ] && echo "demand" || \
                    ([ "$USE_CASE" == "price_optimization" ] && echo "optimal_price" || echo "target"))
        ;;
    churn_prediction)
        TASK_TYPE="classification"
        TARGET_COL="churned"
        ;;
    fraud_detection)
        TASK_TYPE="classification"
        TARGET_COL="is_fraud"
        ;;
    *)
        print_error "Unknown use case: $USE_CASE"
        show_help
        exit 1
        ;;
esac

MODEL_NAME="${USE_CASE}_model"

print_header "ML Pipeline Demo: ${USE_CASE}"

echo "Configuration:"
echo "  Use Case:    $USE_CASE"
echo "  Task Type:   $TASK_TYPE"
echo "  Target:      $TARGET_COL"
echo "  Model Name:  $MODEL_NAME"
echo "  MLflow URL:  $MLFLOW_URL"
echo "  API URL:     $API_URL"
echo ""

# Check if services are running
print_header "Step 1: Checking Services"

check_service() {
    local url=$1
    local name=$2
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|302"; then
        print_success "$name is running at $url"
        return 0
    else
        print_error "$name is not accessible at $url"
        return 1
    fi
}

SERVICES_OK=true
check_service "$MLFLOW_URL/health" "MLflow" || SERVICES_OK=false
check_service "$API_URL/health" "API" || SERVICES_OK=false

if [ "$SERVICES_OK" = false ]; then
    print_error "Some services are not running. Please start the infrastructure first:"
    echo "  docker-compose up -d"
    exit 1
fi

# Generate sample data
print_header "Step 2: Generating Sample Data"

mkdir -p "$DATA_DIR"
cd "$SCRIPT_DIR"

python3 data_generator.py \
    --use-case "$USE_CASE" \
    --samples "$SAMPLES" \
    --output "$DATA_DIR/"

DATA_FILE="$DATA_DIR/${USE_CASE}_data.csv"
if [ ! -f "$DATA_FILE" ]; then
    # Handle generic case
    DATA_FILE="$DATA_DIR/generic_${TASK_TYPE}_data.csv"
fi

print_success "Generated data at $DATA_FILE"

# Train model and register to MLflow
print_header "Step 3: Training Model"

# Determine ID columns based on use case
case "$USE_CASE" in
    demand_forecasting)
        ID_COLS="product_id,date"
        ;;
    churn_prediction)
        ID_COLS="customer_id"
        ;;
    fraud_detection)
        ID_COLS="transaction_id,timestamp"
        ;;
    price_optimization)
        ID_COLS="product_id"
        ;;
    *)
        ID_COLS="entity_id"
        ;;
esac

python3 train_model.py \
    --data "$DATA_FILE" \
    --target "$TARGET_COL" \
    --task "$TASK_TYPE" \
    --id-cols "$ID_COLS" \
    --model-name "$MODEL_NAME" \
    --experiment "${USE_CASE}_experiment" \
    --mlflow-url "$MLFLOW_URL" \
    --promote

print_success "Model trained and registered to MLflow"

# Trigger API to reload models
print_header "Step 4: Reloading API Models"

RELOAD_RESPONSE=$(curl -s -X POST "$API_URL/models/reload")
echo "API Response: $RELOAD_RESPONSE"
print_success "API model reload triggered"

sleep 3  # Wait for models to load

# Check loaded models
print_header "Step 5: Checking Results"

echo "MLflow UI: $MLFLOW_URL"
echo "  - Experiments: $MLFLOW_URL/#/experiments"
echo "  - Models: $MLFLOW_URL/#/models"
echo ""

MODELS_RESPONSE=$(curl -s "$API_URL/models")
echo "Loaded Models:"
echo "$MODELS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE"
echo ""

# Show metrics endpoint
print_header "Step 6: Prometheus Metrics"

echo "Metrics available at: $API_URL/metrics"
echo ""
curl -s "$API_URL/metrics" | head -20
echo "..."
print_info "Full metrics available at $API_URL/metrics"

# Summary
print_header "Demo Complete!"

cat << EOF
What was demonstrated:
  ✓ Generated ${SAMPLES} samples of ${USE_CASE} data
  ✓ Trained multiple ML models (LightGBM, XGBoost, RandomForest)
  ✓ Evaluated and compared model performance
  ✓ Registered best model to MLflow
  ✓ Promoted model to Production stage
  ✓ Triggered API to reload models

Next Steps:
  1. View experiments in MLflow: $MLFLOW_URL
  2. Check API docs: $API_URL/docs
  3. View Grafana dashboards: http://localhost:3000

To adapt this for your own data:
  1. Prepare your CSV/Parquet file
  2. Run: python train_model.py --data your_data.csv --target your_target_column
  3. See examples/generic-ml-usecase/ADAPTATION_GUIDE.md for detailed instructions
EOF
