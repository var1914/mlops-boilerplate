#!/usr/bin/env python3
"""
Generic Model Training Script for ML Pipeline

This script trains ML models on any tabular dataset and registers them to MLflow.
It serves as a template for adapting the boilerplate to your business use case.

Features:
    - Automatic feature type detection (numeric, categorical)
    - Multiple algorithms (LightGBM, XGBoost, Random Forest)
    - Automatic hyperparameter tuning (optional)
    - MLflow experiment tracking and model registry
    - Support for regression and classification tasks

Usage:
    # Train on generated sample data
    python train_model.py --data data/demand_forecasting_data.csv --target demand

    # Train classification model
    python train_model.py --data data/churn_data.csv --target churned --task classification

    # With custom MLflow tracking
    python train_model.py --data data/my_data.csv --target y --mlflow-url http://localhost:5001

    # Register with custom model name
    python train_model.py --data data/my_data.csv --target y --model-name my_custom_model
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

warnings.filterwarnings('ignore')


class GenericModelTrainer:
    """
    Generic model trainer that works with any tabular dataset.

    This class:
    1. Automatically detects feature types
    2. Preprocesses data (encoding, scaling)
    3. Trains multiple algorithms
    4. Evaluates and compares models
    5. Registers best model to MLflow
    """

    def __init__(
        self,
        task_type: str = 'regression',
        mlflow_url: str = 'http://localhost:5001',
        experiment_name: str = 'generic_ml_experiment'
    ):
        """
        Initialize the trainer.

        Args:
            task_type: 'regression' or 'classification'
            mlflow_url: MLflow tracking server URL
            experiment_name: Name for MLflow experiment
        """
        self.task_type = task_type
        self.mlflow_url = mlflow_url
        self.experiment_name = experiment_name
        self.label_encoders = {}
        self.scaler = None
        self.feature_columns = []
        self.categorical_columns = []
        self.numeric_columns = []

    def detect_feature_types(self, df: pd.DataFrame, target_col: str, id_cols: List[str] = None) -> None:
        """Automatically detect numeric and categorical features."""
        if id_cols is None:
            id_cols = []

        # Columns to exclude
        exclude_cols = [target_col] + id_cols

        # Detect column types
        for col in df.columns:
            if col in exclude_cols:
                continue

            if df[col].dtype in ['object', 'category', 'bool']:
                self.categorical_columns.append(col)
            elif df[col].dtype in ['datetime64[ns]', 'datetime64']:
                # Extract datetime features
                continue  # Skip datetime for now, would need feature engineering
            else:
                self.numeric_columns.append(col)

        self.feature_columns = self.numeric_columns + self.categorical_columns
        print(f"Detected {len(self.numeric_columns)} numeric features")
        print(f"Detected {len(self.categorical_columns)} categorical features")

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training.

        Args:
            df: Input DataFrame
            target_col: Target column name
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            X: Feature matrix
            y: Target array
        """
        X_df = df[self.feature_columns].copy()
        y = df[target_col].values

        # Encode categorical columns
        for col in self.categorical_columns:
            if fit:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories
                X_df[col] = X_df[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Scale numeric columns
        if fit:
            self.scaler = StandardScaler()
            X_df[self.numeric_columns] = self.scaler.fit_transform(X_df[self.numeric_columns])
        else:
            X_df[self.numeric_columns] = self.scaler.transform(X_df[self.numeric_columns])

        # Handle missing values
        X_df = X_df.fillna(0)

        return X_df.values, y

    def get_models(self) -> Dict:
        """Get dictionary of models to train."""
        models = {}

        try:
            import lightgbm as lgb
            if self.task_type == 'regression':
                models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            else:
                models['lightgbm'] = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
        except ImportError:
            print("LightGBM not installed, skipping...")

        try:
            import xgboost as xgb
            if self.task_type == 'regression':
                models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0
                )
            else:
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
        except ImportError:
            print("XGBoost not installed, skipping...")

        # Always include sklearn models
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        if self.task_type == 'regression':
            models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

        return models

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
        """Evaluate model performance."""
        metrics = {}

        if self.task_type == 'regression':
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            if y_prob is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                except:
                    pass

        return metrics

    def train(
        self,
        df: pd.DataFrame,
        target_col: str,
        id_cols: List[str] = None,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train models on the dataset.

        Args:
            df: Input DataFrame
            target_col: Target column name
            id_cols: ID columns to exclude from features
            test_size: Fraction of data for testing

        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*60}")
        print(f"  Training {self.task_type} models")
        print(f"{'='*60}\n")

        # Detect feature types
        self.detect_feature_types(df, target_col, id_cols)

        # Preprocess data
        X, y = self.preprocess_data(df, target_col, fit=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(self.feature_columns)}")

        # Train and evaluate models
        results = {}
        models = self.get_models()

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)
            y_prob = None
            if self.task_type == 'classification' and hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

            # Evaluate
            metrics = self.evaluate_model(y_test, y_pred, y_prob)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring='neg_root_mean_squared_error' if self.task_type == 'regression' else 'f1_weighted'
            )

            results[name] = {
                'model': model,
                'metrics': metrics,
                'cv_mean': np.mean(np.abs(cv_scores)),
                'cv_std': np.std(np.abs(cv_scores))
            }

            # Print metrics
            print(f"  Metrics: {metrics}")
            print(f"  CV Score: {results[name]['cv_mean']:.4f} (+/- {results[name]['cv_std']:.4f})")

        return results

    def register_to_mlflow(
        self,
        results: Dict,
        model_name: str,
        df: pd.DataFrame
    ) -> str:
        """
        Register the best model to MLflow.

        Args:
            results: Training results dictionary
            model_name: Name for the registered model
            df: Original DataFrame for signature inference

        Returns:
            MLflow run ID
        """
        import mlflow
        import mlflow.sklearn

        print(f"\n{'='*60}")
        print(f"  Registering model to MLflow")
        print(f"{'='*60}\n")

        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_url)
        mlflow.set_experiment(self.experiment_name)

        # Find best model
        if self.task_type == 'regression':
            best_name = min(results.keys(), key=lambda k: results[k]['metrics']['rmse'])
        else:
            best_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1'])

        best_result = results[best_name]
        best_model = best_result['model']

        print(f"Best model: {best_name}")
        print(f"Metrics: {best_result['metrics']}")

        # Start MLflow run
        run_name = f"{model_name}_{best_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param('algorithm', best_name)
            mlflow.log_param('task_type', self.task_type)
            mlflow.log_param('n_features', len(self.feature_columns))
            mlflow.log_param('n_numeric_features', len(self.numeric_columns))
            mlflow.log_param('n_categorical_features', len(self.categorical_columns))

            # Log metrics
            for metric_name, metric_value in best_result['metrics'].items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.log_metric('cv_score_mean', best_result['cv_mean'])
            mlflow.log_metric('cv_score_std', best_result['cv_std'])

            # Log feature importance if available
            if hasattr(best_model, 'feature_importances_'):
                importance = dict(zip(self.feature_columns, best_model.feature_importances_))
                importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
                mlflow.log_dict(importance_sorted, 'feature_importance.json')

            # Log model metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'numeric_columns': self.numeric_columns,
                'categorical_columns': self.categorical_columns,
                'label_encoders': {k: list(v.classes_) for k, v in self.label_encoders.items()},
                'training_timestamp': datetime.now().isoformat()
            }
            mlflow.log_dict(metadata, 'model_metadata.json')

            # Register model
            mlflow.sklearn.log_model(
                best_model,
                'model',
                registered_model_name=model_name
            )

            print(f"\nModel registered: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"View at: {self.mlflow_url}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

            return run.info.run_id

    def promote_to_production(self, model_name: str) -> bool:
        """Promote the latest model version to Production stage."""
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.mlflow_url)
        client = MlflowClient()

        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                print(f"No versions found for model: {model_name}")
                return False

            latest_version = max(versions, key=lambda v: int(v.version))

            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production",
                archive_existing_versions=True
            )

            print(f"Promoted {model_name} v{latest_version.version} to Production")
            return True

        except Exception as e:
            print(f"Failed to promote model: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model and register to MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train regression model
  python train_model.py --data data/demand_data.csv --target demand

  # Train classification model
  python train_model.py --data data/churn_data.csv --target churned --task classification

  # Custom MLflow server
  python train_model.py --data data.csv --target y --mlflow-url http://mlflow:5000

  # Full example with all options
  python train_model.py \\
      --data data/sales_data.csv \\
      --target revenue \\
      --task regression \\
      --id-cols customer_id,date \\
      --model-name sales_predictor \\
      --experiment sales_forecasting \\
      --mlflow-url http://localhost:5001 \\
      --promote
        """
    )

    parser.add_argument('--data', required=True, help='Path to CSV/Parquet data file')
    parser.add_argument('--target', required=True, help='Target column name')
    parser.add_argument('--task', choices=['regression', 'classification'],
                        default='regression', help='Task type')
    parser.add_argument('--id-cols', default='', help='Comma-separated ID columns to exclude')
    parser.add_argument('--model-name', default='generic_model', help='Model name for MLflow registry')
    parser.add_argument('--experiment', default='generic_ml_experiment', help='MLflow experiment name')
    parser.add_argument('--mlflow-url', default='http://localhost:5001', help='MLflow tracking URI')
    parser.add_argument('--promote', action='store_true', help='Promote model to Production after training')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')

    args = parser.parse_args()

    # Load data
    print(f"\nLoading data from {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    # Parse ID columns
    id_cols = [c.strip() for c in args.id_cols.split(',') if c.strip()]

    # Initialize trainer
    trainer = GenericModelTrainer(
        task_type=args.task,
        mlflow_url=args.mlflow_url,
        experiment_name=args.experiment
    )

    # Train models
    results = trainer.train(
        df=df,
        target_col=args.target,
        id_cols=id_cols,
        test_size=args.test_size
    )

    # Register to MLflow
    run_id = trainer.register_to_mlflow(
        results=results,
        model_name=args.model_name,
        df=df
    )

    # Optionally promote to production
    if args.promote:
        trainer.promote_to_production(args.model_name)

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
