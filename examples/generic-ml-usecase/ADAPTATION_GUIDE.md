# Adaptation Guide: Using This Boilerplate for Your Business

This guide explains how to adapt the ML Inference Infrastructure Boilerplate for your specific business use case.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Step-by-Step Adaptation](#step-by-step-adaptation)
4. [Use Case Examples](#use-case-examples)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Start the Infrastructure

```bash
# Start all services
docker-compose up -d

# Verify everything is running
./scripts/validate-deployment.sh --env docker
```

### 2. Run the Demo with Sample Data

```bash
cd examples/generic-ml-usecase

# Run demand forecasting demo
./run_demo.sh demand_forecasting

# Or try other use cases
./run_demo.sh churn_prediction
./run_demo.sh fraud_detection
```

### 3. Adapt for Your Data

```bash
# Train on your own data
python train_model.py \
    --data /path/to/your_data.csv \
    --target your_target_column \
    --task regression \
    --model-name your_model_name \
    --promote
```

---

## Understanding the Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Data Source │───▶│ ML Pipeline │───▶│  Inference  │     │
│  └─────────────┘    └─────────────┘    │     API     │     │
│                            │           └─────────────┘     │
│                            ▼                  │            │
│                     ┌─────────────┐           │            │
│                     │   MLflow    │◀──────────┘            │
│                     │  Registry   │                        │
│                     └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### What Each Component Does

| Component | Purpose | Your Customization |
|-----------|---------|-------------------|
| **Data Source** | Raw data storage | Connect your database/API/files |
| **ML Pipeline** | Feature engineering + training | Add your features and models |
| **MLflow** | Model versioning and registry | Configure experiments |
| **Inference API** | Serve predictions | Add custom endpoints |
| **Monitoring** | Track metrics | Add business KPIs |

---

## Step-by-Step Adaptation

### Step 1: Prepare Your Data

Your data should be in a tabular format (CSV, Parquet, or database table).

**Required Structure:**
```
| id_column | feature_1 | feature_2 | ... | target |
|-----------|-----------|-----------|-----|--------|
| ID_001    | 10.5      | category_a| ... | 150    |
| ID_002    | 8.3       | category_b| ... | 120    |
```

**Data Checklist:**
- [ ] Has a unique identifier column
- [ ] Target column is clearly defined
- [ ] No data leakage (future information in features)
- [ ] Reasonable number of missing values (<20%)
- [ ] At least 1,000 samples (more is better)

### Step 2: Choose Your Task Type

| Task Type | Target Column | Example Use Cases |
|-----------|---------------|-------------------|
| **Regression** | Continuous number | Price, demand, revenue, time |
| **Binary Classification** | 0 or 1 | Churn, fraud, click, conversion |
| **Multi-class Classification** | Categories | Segment, risk level, product type |

### Step 3: Train Your Model

```bash
python examples/generic-ml-usecase/train_model.py \
    --data your_data.csv \
    --target target_column \
    --task regression \
    --id-cols id_column,date_column \
    --model-name your_business_model \
    --experiment your_experiment \
    --mlflow-url http://localhost:5001 \
    --promote
```

**Parameters Explained:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--data` | Path to your data file | `data/sales.csv` |
| `--target` | Column to predict | `revenue` |
| `--task` | regression or classification | `regression` |
| `--id-cols` | Columns to exclude from features | `customer_id,date` |
| `--model-name` | Name in MLflow registry | `sales_predictor` |
| `--experiment` | MLflow experiment name | `sales_forecasting` |
| `--promote` | Auto-promote to Production | (flag) |

### Step 4: Verify in MLflow

1. Open MLflow UI: http://localhost:5001
2. Check your experiment for logged metrics
3. Go to "Models" tab to see registered model
4. Verify model is in "Production" stage

### Step 5: Use the Inference API

```bash
# Reload models in API
curl -X POST http://localhost:8000/models/reload

# Check loaded models
curl http://localhost:8000/models

# Make predictions (requires API customization for your endpoints)
curl -X POST http://localhost:8000/predict/your_entity \
    -H "Content-Type: application/json" \
    -d '{"features": {...}}'
```

---

## Use Case Examples

### Example 1: E-commerce Demand Forecasting

**Business Problem:** Predict daily product demand to optimize inventory.

**Data Schema:**
```python
{
    'product_id': str,      # SKU identifier
    'date': datetime,       # Prediction date
    'price': float,         # Current price
    'promotion': int,       # Is on sale (0/1)
    'day_of_week': int,     # 0-6
    'season': str,          # spring/summer/fall/winter
    'competitor_price': float,
    'demand': int           # TARGET: Units sold
}
```

**Training Command:**
```bash
python train_model.py \
    --data demand_data.csv \
    --target demand \
    --task regression \
    --id-cols product_id,date \
    --model-name demand_forecaster \
    --experiment demand_forecasting
```

**Model Naming Convention:**
```
{business}_{algorithm}_{task}_{entity}
# Example: ecommerce_lightgbm_demand_PROD001
```

---

### Example 2: Customer Churn Prediction

**Business Problem:** Identify customers likely to churn for retention campaigns.

**Data Schema:**
```python
{
    'customer_id': str,         # Customer identifier
    'tenure_months': int,       # Months as customer
    'monthly_charges': float,   # Monthly bill
    'total_charges': float,     # Lifetime value
    'contract_type': str,       # month-to-month/annual
    'num_support_tickets': int, # Support interactions
    'satisfaction_score': int,  # 1-5 rating
    'churned': int              # TARGET: 0 or 1
}
```

**Training Command:**
```bash
python train_model.py \
    --data churn_data.csv \
    --target churned \
    --task classification \
    --id-cols customer_id \
    --model-name churn_predictor \
    --experiment churn_prediction
```

---

### Example 3: Fraud Detection

**Business Problem:** Flag potentially fraudulent transactions in real-time.

**Data Schema:**
```python
{
    'transaction_id': str,      # Transaction ID
    'timestamp': datetime,      # Transaction time
    'amount': float,            # Transaction amount
    'merchant_category': str,   # Merchant type
    'is_international': int,    # Cross-border (0/1)
    'distance_from_home': float,# Miles from home
    'num_txn_24h': int,         # Recent transaction count
    'is_fraud': int             # TARGET: 0 or 1
}
```

**Training Command:**
```bash
python train_model.py \
    --data fraud_data.csv \
    --target is_fraud \
    --task classification \
    --id-cols transaction_id,timestamp \
    --model-name fraud_detector \
    --experiment fraud_detection
```

---

### Example 4: Dynamic Pricing

**Business Problem:** Set optimal prices to maximize revenue.

**Data Schema:**
```python
{
    'product_id': str,          # Product SKU
    'base_cost': float,         # Product cost
    'competitor_price': float,  # Market price
    'demand_elasticity': float, # Price sensitivity
    'inventory_days': int,      # Days of stock
    'season': str,              # Current season
    'optimal_price': float      # TARGET: Best price
}
```

**Training Command:**
```bash
python train_model.py \
    --data pricing_data.csv \
    --target optimal_price \
    --task regression \
    --id-cols product_id \
    --model-name price_optimizer \
    --experiment price_optimization
```

---

## Best Practices

### Data Quality

```python
# Check data quality before training
df = pd.read_csv('your_data.csv')

# Missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Target distribution
print(f"Target distribution:\n{df['target'].describe()}")

# Check for data leakage
# Ensure no features are derived from the target
```

### Feature Engineering Tips

1. **Time Features**: Extract day, month, hour, day-of-week from timestamps
2. **Lag Features**: Include historical values (t-1, t-7, t-30)
3. **Rolling Statistics**: Moving averages, standard deviations
4. **Categorical Encoding**: One-hot or label encoding for categories
5. **Interaction Features**: Combine related features (price × quantity)

### Model Selection Guidelines

| Data Size | Recommended Models |
|-----------|-------------------|
| < 1,000 | Random Forest, Ridge/Logistic |
| 1,000 - 100,000 | LightGBM, XGBoost |
| > 100,000 | LightGBM (fast), Neural Networks |

### Monitoring Recommendations

Track these metrics in production:
- **Model Performance**: Accuracy/RMSE drift over time
- **Prediction Distribution**: Detect concept drift
- **Latency**: API response times
- **Business KPIs**: Revenue impact, conversion rates

---

## Troubleshooting

### Common Issues

**1. Model Not Loading in API**

```bash
# Check model name in MLflow
curl http://localhost:5001/api/2.0/mlflow/registered-models/list

# Verify model stage is "Production"
# Model name must follow convention: {prefix}_{algorithm}_{task}_{entity}
```

**2. Low Model Performance**

- Check for data leakage
- Increase training data
- Try different algorithms
- Add more relevant features
- Handle class imbalance (for classification)

**3. API Returns Errors**

```bash
# Check API logs
docker-compose logs api

# Test MLflow connection
curl http://localhost:5001/health

# Reload models
curl -X POST http://localhost:8000/models/reload
```

**4. MLflow Connection Issues**

```bash
# Verify MLflow is running
curl http://localhost:5001/health

# Check MinIO for artifacts
curl http://localhost:9001  # MinIO console

# Set environment variables if needed
export MLFLOW_TRACKING_URI=http://localhost:5001
```

---

## Next Steps

1. **Customize the API**: Add endpoints specific to your business logic
2. **Add Feature Pipeline**: Automate feature engineering with Airflow
3. **Set Up Monitoring**: Create Grafana dashboards for your KPIs
4. **Implement A/B Testing**: Compare model versions in production
5. **Scale with Kubernetes**: Deploy to production K8s cluster

For more details, see:
- [Main README](../../README.md)
- [K8s Deployment Guide](../../k8s/README.md)
- [Airflow Integration Plan](../../.claude/plans/)

---

## Support

- **Issues**: https://github.com/your-org/ml-eng-with-ops/issues
- **Documentation**: See `/docs` folder
- **Examples**: See `/examples` folder
