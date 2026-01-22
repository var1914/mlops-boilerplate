#!/usr/bin/env python3
"""
Sample Data Generator for ML Pipeline Demo

This script generates synthetic datasets for different business use cases.
Use this to test the ML pipeline before integrating your real data.

Supported Use Cases:
    - demand_forecasting: Retail/E-commerce demand prediction
    - churn_prediction: Customer churn classification
    - fraud_detection: Transaction fraud classification
    - price_optimization: Dynamic pricing regression
    - generic: Generic tabular data for custom use cases

Usage:
    python data_generator.py --use-case demand_forecasting --output data/
    python data_generator.py --use-case churn_prediction --samples 10000
    python data_generator.py --use-case generic --features 20 --samples 5000
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


def generate_demand_forecasting_data(
    n_samples: int = 10000,
    n_products: int = 50,
    start_date: str = "2023-01-01"
) -> pd.DataFrame:
    """
    Generate synthetic demand forecasting data.

    Features:
        - product_id: Product identifier
        - date: Date of observation
        - price: Product price
        - promotion: Whether product is on promotion (0/1)
        - day_of_week: Day of week (0-6)
        - month: Month (1-12)
        - holiday: Whether it's a holiday (0/1)
        - competitor_price: Competitor's price
        - inventory_level: Current inventory
        - weather_score: Weather impact score (1-10)

    Target:
        - demand: Units sold (regression target)
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=n_samples // n_products, freq='D')

    data = []
    for product_id in range(1, n_products + 1):
        base_demand = np.random.uniform(50, 500)
        base_price = np.random.uniform(10, 100)

        for date in dates:
            # Features
            price = base_price * np.random.uniform(0.8, 1.2)
            promotion = np.random.choice([0, 1], p=[0.8, 0.2])
            day_of_week = date.dayofweek
            month = date.month
            holiday = 1 if date.dayofweek >= 5 or month in [11, 12] else 0
            competitor_price = price * np.random.uniform(0.9, 1.1)
            inventory_level = np.random.randint(0, 1000)
            weather_score = np.random.randint(1, 11)

            # Target: demand influenced by features
            demand = base_demand
            demand *= (1 - 0.3 * (price / base_price - 1))  # Price elasticity
            demand *= (1 + 0.5 * promotion)  # Promotion boost
            demand *= (1 + 0.2 * (day_of_week >= 5))  # Weekend boost
            demand *= (1 + 0.3 * holiday)  # Holiday boost
            demand *= (1 + 0.1 * (competitor_price / price - 1))  # Competitor effect
            demand *= np.random.uniform(0.8, 1.2)  # Random noise
            demand = max(0, int(demand))

            data.append({
                'product_id': f'PROD_{product_id:03d}',
                'date': date,
                'price': round(price, 2),
                'promotion': promotion,
                'day_of_week': day_of_week,
                'month': month,
                'holiday': holiday,
                'competitor_price': round(competitor_price, 2),
                'inventory_level': inventory_level,
                'weather_score': weather_score,
                'demand': demand  # Target
            })

    return pd.DataFrame(data)


def generate_churn_prediction_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic customer churn data.

    Features:
        - customer_id: Customer identifier
        - tenure_months: Months as customer
        - monthly_charges: Monthly bill amount
        - total_charges: Lifetime charges
        - contract_type: Month-to-month, One year, Two year
        - payment_method: Payment method used
        - num_support_tickets: Support tickets filed
        - avg_monthly_usage: Average usage per month
        - num_products: Number of products subscribed
        - satisfaction_score: Last satisfaction rating (1-5)

    Target:
        - churned: Whether customer churned (0/1)
    """
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        tenure = np.random.exponential(24)  # Average 24 months
        tenure = min(int(tenure), 72)  # Cap at 72 months

        monthly_charges = np.random.uniform(20, 150)
        total_charges = monthly_charges * tenure * np.random.uniform(0.9, 1.1)

        contract_type = np.random.choice(
            ['month-to-month', 'one_year', 'two_year'],
            p=[0.5, 0.3, 0.2]
        )
        payment_method = np.random.choice(
            ['credit_card', 'bank_transfer', 'electronic_check', 'mailed_check'],
            p=[0.35, 0.25, 0.25, 0.15]
        )

        num_support_tickets = np.random.poisson(2)
        avg_monthly_usage = np.random.uniform(10, 100)
        num_products = np.random.randint(1, 6)
        satisfaction_score = np.random.randint(1, 6)

        # Churn probability based on features
        churn_prob = 0.2  # Base churn rate
        churn_prob += 0.3 if contract_type == 'month-to-month' else -0.1
        churn_prob += 0.05 * num_support_tickets
        churn_prob -= 0.02 * tenure
        churn_prob -= 0.05 * satisfaction_score
        churn_prob += 0.1 if payment_method == 'electronic_check' else 0
        churn_prob = np.clip(churn_prob, 0.05, 0.95)

        churned = np.random.choice([0, 1], p=[1 - churn_prob, churn_prob])

        data.append({
            'customer_id': f'CUST_{i+1:06d}',
            'tenure_months': tenure,
            'monthly_charges': round(monthly_charges, 2),
            'total_charges': round(total_charges, 2),
            'contract_type': contract_type,
            'payment_method': payment_method,
            'num_support_tickets': num_support_tickets,
            'avg_monthly_usage': round(avg_monthly_usage, 2),
            'num_products': num_products,
            'satisfaction_score': satisfaction_score,
            'churned': churned  # Target
        })

    return pd.DataFrame(data)


def generate_fraud_detection_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic fraud detection data.

    Features:
        - transaction_id: Transaction identifier
        - timestamp: Transaction time
        - amount: Transaction amount
        - merchant_category: Type of merchant
        - is_international: Cross-border transaction
        - distance_from_home: Miles from customer's home
        - time_since_last_txn: Minutes since last transaction
        - avg_txn_amount_7d: Average transaction amount (7 days)
        - num_txn_24h: Number of transactions in 24 hours
        - device_type: Device used for transaction

    Target:
        - is_fraud: Whether transaction is fraudulent (0/1)
    """
    np.random.seed(42)

    data = []
    base_time = datetime(2024, 1, 1)

    for i in range(n_samples):
        is_fraud = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud rate

        if is_fraud:
            # Fraudulent transactions tend to be unusual
            amount = np.random.exponential(500) + 100
            distance_from_home = np.random.exponential(500)
            time_since_last_txn = np.random.exponential(5)  # Quick succession
            num_txn_24h = np.random.poisson(10) + 5
            is_international = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            # Normal transactions
            amount = np.random.exponential(50) + 10
            distance_from_home = np.random.exponential(20)
            time_since_last_txn = np.random.exponential(120)
            num_txn_24h = np.random.poisson(3)
            is_international = np.random.choice([0, 1], p=[0.9, 0.1])

        timestamp = base_time + timedelta(minutes=i * np.random.uniform(1, 10))
        merchant_category = np.random.choice([
            'retail', 'food', 'travel', 'entertainment', 'online', 'atm'
        ])
        avg_txn_amount_7d = np.random.uniform(30, 200)
        device_type = np.random.choice(['mobile', 'desktop', 'pos_terminal', 'atm'])

        data.append({
            'transaction_id': f'TXN_{i+1:08d}',
            'timestamp': timestamp,
            'amount': round(amount, 2),
            'merchant_category': merchant_category,
            'is_international': is_international,
            'distance_from_home': round(distance_from_home, 2),
            'time_since_last_txn': round(time_since_last_txn, 2),
            'avg_txn_amount_7d': round(avg_txn_amount_7d, 2),
            'num_txn_24h': num_txn_24h,
            'device_type': device_type,
            'is_fraud': is_fraud  # Target
        })

    return pd.DataFrame(data)


def generate_price_optimization_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic price optimization data.

    Features:
        - product_id: Product identifier
        - base_cost: Product cost
        - competitor_min_price: Minimum competitor price
        - competitor_max_price: Maximum competitor price
        - demand_elasticity: Historical price sensitivity
        - inventory_days: Days of inventory remaining
        - season: Current season
        - customer_segment: Target customer segment
        - brand_strength: Brand recognition score (1-10)
        - time_to_expiry: Days until product expires (if applicable)

    Target:
        - optimal_price: Optimal selling price (regression)
    """
    np.random.seed(42)

    data = []
    for i in range(n_samples):
        base_cost = np.random.uniform(5, 200)
        competitor_min = base_cost * np.random.uniform(1.1, 1.5)
        competitor_max = competitor_min * np.random.uniform(1.1, 1.5)
        demand_elasticity = np.random.uniform(-3, -0.5)  # Typically negative
        inventory_days = np.random.randint(1, 90)
        season = np.random.choice(['spring', 'summer', 'fall', 'winter'])
        customer_segment = np.random.choice(['budget', 'mid_range', 'premium'])
        brand_strength = np.random.randint(1, 11)
        time_to_expiry = np.random.choice([0, np.random.randint(1, 365)], p=[0.7, 0.3])

        # Calculate optimal price based on features
        margin_multiplier = 1.3  # Base 30% margin
        margin_multiplier += 0.1 * (brand_strength / 10)  # Brand premium
        margin_multiplier += 0.1 if customer_segment == 'premium' else -0.1 if customer_segment == 'budget' else 0
        margin_multiplier -= 0.1 * max(0, (30 - inventory_days) / 30)  # Discount excess inventory
        margin_multiplier -= 0.2 * max(0, (30 - time_to_expiry) / 30) if time_to_expiry > 0 else 0

        optimal_price = base_cost * margin_multiplier
        optimal_price = np.clip(optimal_price, competitor_min * 0.9, competitor_max * 1.1)
        optimal_price *= np.random.uniform(0.95, 1.05)  # Add noise

        data.append({
            'product_id': f'SKU_{i+1:06d}',
            'base_cost': round(base_cost, 2),
            'competitor_min_price': round(competitor_min, 2),
            'competitor_max_price': round(competitor_max, 2),
            'demand_elasticity': round(demand_elasticity, 2),
            'inventory_days': inventory_days,
            'season': season,
            'customer_segment': customer_segment,
            'brand_strength': brand_strength,
            'time_to_expiry': time_to_expiry,
            'optimal_price': round(optimal_price, 2)  # Target
        })

    return pd.DataFrame(data)


def generate_generic_data(
    n_samples: int = 10000,
    n_features: int = 20,
    task_type: str = 'regression'
) -> pd.DataFrame:
    """
    Generate generic synthetic data for custom use cases.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: 'regression' or 'classification'

    Returns:
        DataFrame with generic features and target
    """
    np.random.seed(42)

    # Generate feature matrix
    X = np.random.randn(n_samples, n_features)

    # Add some non-linear relationships
    X[:, 0] = X[:, 0] ** 2  # Squared feature
    X[:, 1] = np.sin(X[:, 1])  # Sinusoidal feature
    X[:, 2] = np.abs(X[:, 2])  # Absolute feature

    # Generate target based on features
    weights = np.random.randn(n_features)
    y = X @ weights + np.random.randn(n_samples) * 0.5

    if task_type == 'classification':
        # Convert to binary classification
        threshold = np.median(y)
        y = (y > threshold).astype(int)
    else:
        # Scale regression target
        y = (y - y.mean()) / y.std() * 10 + 50

    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['entity_id'] = [f'ENTITY_{i+1:06d}' for i in range(n_samples)]
    df['target'] = y if task_type == 'regression' else y.astype(int)

    # Reorder columns
    cols = ['entity_id'] + feature_names + ['target']
    df = df[cols]

    return df


def save_data(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """Save DataFrame to CSV and Parquet formats."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f'{filename}.csv')
    parquet_path = os.path.join(output_dir, f'{filename}.parquet')

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"Saved {len(df)} samples to:")
    print(f"  - {csv_path}")
    print(f"  - {parquet_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample data for ML pipeline demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate demand forecasting data
  python data_generator.py --use-case demand_forecasting

  # Generate churn prediction data with 20000 samples
  python data_generator.py --use-case churn_prediction --samples 20000

  # Generate generic data with 30 features
  python data_generator.py --use-case generic --features 30 --task classification

  # Save to custom directory
  python data_generator.py --use-case fraud_detection --output ./my_data/
        """
    )

    parser.add_argument(
        '--use-case',
        choices=['demand_forecasting', 'churn_prediction', 'fraud_detection',
                 'price_optimization', 'generic'],
        default='generic',
        help='Type of data to generate'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10000,
        help='Number of samples to generate (default: 10000)'
    )
    parser.add_argument(
        '--features',
        type=int,
        default=20,
        help='Number of features for generic data (default: 20)'
    )
    parser.add_argument(
        '--task',
        choices=['regression', 'classification'],
        default='regression',
        help='Task type for generic data (default: regression)'
    )
    parser.add_argument(
        '--output',
        default='./data/',
        help='Output directory (default: ./data/)'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Generating {args.use_case} data")
    print(f"{'='*60}\n")

    # Generate data based on use case
    if args.use_case == 'demand_forecasting':
        df = generate_demand_forecasting_data(n_samples=args.samples)
        filename = 'demand_forecasting_data'
    elif args.use_case == 'churn_prediction':
        df = generate_churn_prediction_data(n_samples=args.samples)
        filename = 'churn_prediction_data'
    elif args.use_case == 'fraud_detection':
        df = generate_fraud_detection_data(n_samples=args.samples)
        filename = 'fraud_detection_data'
    elif args.use_case == 'price_optimization':
        df = generate_price_optimization_data(n_samples=args.samples)
        filename = 'price_optimization_data'
    else:  # generic
        df = generate_generic_data(
            n_samples=args.samples,
            n_features=args.features,
            task_type=args.task
        )
        filename = f'generic_{args.task}_data'

    # Save data
    save_data(df, args.output, filename)

    # Print summary
    print(f"\nData Summary:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    print(f"\n{'='*60}")
    print("  Data generation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()