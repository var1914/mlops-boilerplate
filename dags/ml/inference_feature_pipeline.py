# inference_feature_pipeline.py - UPDATED for multi-task models
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import redis
from typing import Dict, List, Optional
import logging
import json
import mlflow.pyfunc
import sys
import os

# Add src to path for config imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_settings
from model_training import MLflowModelRegistry

# Task configurations - must match your PREDICTION_CONFIGS
INFERENCE_TASKS = {
    'return_1step': {'type': 'regression', 'description': '15-minute price return'},
    'return_4step': {'type': 'regression', 'description': '1-hour price return'},
    'return_16step': {'type': 'regression', 'description': '4-hour price return'},
    'direction_4step': {'type': 'classification_binary', 'description': '1-hour direction (up/down)'},
    'direction_multi_4step': {'type': 'classification_multi', 'description': '1-hour multi-class direction'},
    'volatility_4step': {'type': 'regression', 'description': '1-hour volatility'},
    'vol_regime_4step': {'type': 'classification_multi', 'description': '1-hour volatility regime'}
}

class InferenceFeatureEngine:
    def __init__(self, db_config=None, redis_config=None):
        """
        Initialize feature engine with optional configs.
        If not provided, loads from centralized settings.
        """
        self.logger = logging.getLogger("inference_features")
        settings = get_settings()
        self.db_config = db_config or settings.database.get_connection_dict()
        redis_cfg = redis_config or settings.redis.get_client_config()
        self.redis_client = redis.Redis(**redis_cfg) if redis_cfg else None
        
    def get_latest_ohlcv(self, symbol: str, periods: int = 200) -> pd.DataFrame:
        """Get latest OHLCV data for feature generation - increased periods for multi-horizon"""
        query = """
        SELECT open_time, open_price, high_price, low_price, close_price, volume, quote_volume
        FROM crypto_data 
        WHERE symbol = %s 
        ORDER BY open_time DESC 
        LIMIT %s
        """
        
        with psycopg2.connect(**self.db_config) as conn:
            df = pd.read_sql(query, conn, params=(symbol, periods))
        
        return df.sort_values('open_time').reset_index(drop=True)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators matching training features"""
        features = df.copy()
        close = df['close_price']
        high = df['high_price'] 
        low = df['low_price']
        open_price = df['open_price']
        volume = df['volume']
        quote_volume = df['quote_volume']
        
        # Moving Averages - match training periods
        for period in [28, 56, 84, 200]:
            features[f'sma_{period}'] = close.rolling(window=period).mean()
            features[f'ema_{period}'] = close.ewm(span=period).mean()
        
        # Price-based features
        features['return_15m'] = close.pct_change()
        features['return_1h'] = close.pct_change(periods=4)
        features['return_4h'] = close.pct_change(periods=16)
        features['return_24h'] = close.pct_change(periods=96)
        features['return_7d'] = close.pct_change(periods=672)
        
        # Volatility
        features['volatility_1h'] = features['return_15m'].rolling(window=4).std()
        features['volatility_24h'] = features['return_15m'].rolling(window=96).std()
        features['volatility_7d'] = features['return_15m'].rolling(window=672).std()
        
        # Price ratios
        features['high_low_ratio'] = high / low
        features['close_open_ratio'] = close / open_price
        features['hl2'] = (high + low) / 2
        features['hlc3'] = (high + low + close) / 3
        features['ohlc4'] = (open_price + high + low + close) / 4
        features['price_position'] = (close - low) / (high - low)
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=56).mean()  # 14 hours
        loss = (-delta.where(delta < 0, 0)).rolling(window=56).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        features['bb_upper'] = sma20 + (std20 * 2)
        features['bb_lower'] = sma20 - (std20 * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (close - features['bb_lower']) / features['bb_width']
        
        # True Range and ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        features['true_range'] = np.maximum(tr1, np.maximum(tr2, tr3))
        features['atr_14'] = features['true_range'].rolling(window=56).mean()
        
        # Volume features
        for period in [28, 56, 96, 672]:
            features[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
        
        features['volume_ratio_7'] = volume / features['volume_sma_28'] 
        features['volume_ratio_24'] = volume / features['volume_sma_96']
        features['vwap'] = (quote_volume / volume).fillna(0)
        features['volume_price_trend'] = features['return_1h'] * features['volume_ratio_24']
        
        # Time features
        if hasattr(features, 'index') and hasattr(features.index, 'hour'):
            features['hour'] = features.index.hour
            features['day_of_week'] = features.index.dayofweek
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        
        return features
    
    def get_inference_features(self, symbol: str, timestamp: Optional[datetime] = None) -> Dict:
        """Get features for prediction"""
        cache_key = f"features:{symbol}:{timestamp or 'latest'}"
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Get latest data
        df = self.get_latest_ohlcv(symbol)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
            
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('datetime')
        
        # Calculate features
        df = self.calculate_technical_indicators(df)
        
        # Get latest row features
        latest_features = df.iloc[-1]
        
        # Convert to dict, handling NaN values
        features_dict = {}
        for col, value in latest_features.items():
            if pd.notna(value) and col not in ['open_time']:
                features_dict[col] = float(value) if isinstance(value, (np.integer, np.floating)) else value
        
        # Cache for 1 minute
        if self.redis_client:
            self.redis_client.setex(cache_key, 60, json.dumps(features_dict, default=str))
        
        return features_dict

class MultiTaskModelPredictor:
    def __init__(self, db_config=None, redis_config=None):
        """
        Initialize model predictor with optional configs.
        If not provided, loads from centralized settings.
        """
        settings = get_settings()
        self.feature_engine = InferenceFeatureEngine(db_config, redis_config)

        # Use MLflow tracking URI from settings if available
        mlflow_uri = settings.mlflow.tracking_uri
        self.mlflow_registry = MLflowModelRegistry(tracking_uri=mlflow_uri)

        self.models = {}  # Cache for loaded models
        self.model_metadata = {}
        self.logger = logging.getLogger("model_predictor")
        
    def load_production_models(self, symbols: List[str], tasks: List[str] = None, model_types: List[str] = None):
        """Load production models for specified symbols and tasks"""
        if tasks is None:
            tasks = list(INFERENCE_TASKS.keys())
        if model_types is None:
            model_types = ['lightgbm', 'xgboost']
        
        loaded_count = 0
        failed_count = 0
        
        for symbol in symbols:
            for task in tasks:
                for model_type in model_types:
                    model_name = f"crypto_{model_type}_{task}_{symbol}"
                    
                    try:
                        prod_version = self.mlflow_registry.get_production_model(model_name)
                        if prod_version:
                            # Load model using MLflow
                            model = mlflow.pyfunc.load_model(prod_version.source)
                            
                            model_key = f"{symbol}_{task}_{model_type}"
                            self.models[model_key] = model
                            self.model_metadata[model_key] = {
                                'registered_name': model_name,
                                'version': prod_version.version,
                                'stage': prod_version.current_stage,
                                'model_uri': prod_version.source,
                                'task_type': INFERENCE_TASKS[task]['type']
                            }
                            loaded_count += 1
                            self.logger.info(f"Loaded {model_name} v{prod_version.version}")
                        else:
                            # Try staging if no production model
                            staging_versions = self.mlflow_registry.client.get_latest_versions(
                                model_name, stages=["Staging"]
                            )
                            if staging_versions:
                                model = mlflow.pyfunc.load_model(staging_versions[0].source)
                                model_key = f"{symbol}_{task}_{model_type}"
                                self.models[model_key] = model
                                self.model_metadata[model_key] = {
                                    'registered_name': model_name,
                                    'version': staging_versions[0].version,
                                    'stage': 'Staging',
                                    'model_uri': staging_versions[0].source,
                                    'task_type': INFERENCE_TASKS[task]['type']
                                }
                                loaded_count += 1
                                self.logger.info(f"Loaded {model_name} v{staging_versions[0].version} from staging")
                    
                    except Exception as e:
                        failed_count += 1
                        self.logger.warning(f"Could not load {model_name}: {e}")
        
        self.logger.info(f"Model loading complete: {loaded_count} loaded, {failed_count} failed")
        return loaded_count, failed_count
    
    def predict_all_tasks(self, symbol: str, timestamp: Optional[datetime] = None) -> Dict:
        """Make predictions for all available tasks for a symbol"""
        # Get features
        try:
            features = self.feature_engine.get_inference_features(symbol, timestamp)
        except Exception as e:
            return {'error': f"Feature extraction failed: {str(e)}"}
        
        if not features:
            return {'error': f"No features available for {symbol}"}
        
        # Convert to DataFrame for model prediction
        feature_df = pd.DataFrame([features])
        
        predictions = {
            'symbol': symbol,
            'timestamp': timestamp or datetime.now(),
            'features_count': len(features),
            'task_predictions': {},
            'ensemble_predictions': {}
        }
        
        # Group models by task for ensemble predictions
        task_predictions = {}
        
        for model_key, model in self.models.items():
            if not model_key.startswith(f"{symbol}_"):
                continue
                
            try:
                # Parse model key: symbol_task_modeltype
                parts = model_key.split('_')
                task = '_'.join(parts[1:-1])  # Handle multi-part task names
                model_type = parts[-1]
                
                if task not in task_predictions:
                    task_predictions[task] = {}
                
                # Get model metadata
                metadata = self.model_metadata.get(model_key, {})
                task_type = metadata.get('task_type', 'regression')
                
                # Make prediction
                pred_result = model.predict(feature_df)
                
                if isinstance(pred_result, np.ndarray):
                    if task_type == 'classification_multi' and pred_result.ndim > 1:
                        # Multi-class prediction - get class probabilities and predicted class
                        prediction = {
                            'class_prediction': int(np.argmax(pred_result[0])),
                            'class_probabilities': pred_result[0].tolist(),
                            'confidence': float(np.max(pred_result[0]))
                        }
                    elif task_type == 'classification_binary' and pred_result.ndim > 1:
                        # Binary classification - get probability and prediction
                        prob = pred_result[0][1] if pred_result.shape[1] > 1 else pred_result[0][0]
                        prediction = {
                            'class_prediction': int(prob > 0.5),
                            'probability': float(prob),
                            'confidence': float(abs(prob - 0.5) * 2)  # Distance from 0.5 * 2
                        }
                    else:
                        # Regression or simple prediction
                        pred_value = pred_result[0] if pred_result.ndim > 0 else pred_result
                        prediction = {
                            'value': float(pred_value),
                            'confidence': self._calculate_regression_confidence(pred_value, task)
                        }
                else:
                    prediction = {
                        'value': float(pred_result),
                        'confidence': self._calculate_regression_confidence(pred_result, task)
                    }
                
                # Add model metadata to prediction
                prediction.update({
                    'model_type': model_type,
                    'model_version': metadata.get('version', 'unknown'),
                    'model_stage': metadata.get('stage', 'unknown')
                })
                
                task_predictions[task][model_type] = prediction
                
            except Exception as e:
                self.logger.error(f"Prediction failed for {model_key}: {e}")
                continue
        
        # Create ensemble predictions for each task
        for task, model_preds in task_predictions.items():
            if not model_preds:
                continue
                
            task_type = INFERENCE_TASKS.get(task, {}).get('type', 'regression')
            
            if task_type == 'regression':
                # Average predictions weighted by confidence
                values = [p['value'] for p in model_preds.values()]
                confidences = [p['confidence'] for p in model_preds.values()]
                
                if confidences and sum(confidences) > 0:
                    weighted_pred = sum(v * c for v, c in zip(values, confidences)) / sum(confidences)
                else:
                    weighted_pred = np.mean(values)
                
                predictions['ensemble_predictions'][task] = {
                    'prediction': float(weighted_pred),
                    'confidence': float(np.mean(confidences)),
                    'model_count': len(model_preds),
                    'std_deviation': float(np.std(values)),
                    'individual_models': model_preds
                }
                
            elif task_type in ['classification_binary', 'classification_multi']:
                # Majority vote with confidence weighting
                if task_type == 'classification_binary':
                    # Average probabilities
                    probs = [p.get('probability', p.get('confidence', 0.5)) for p in model_preds.values()]
                    avg_prob = np.mean(probs)
                    
                    predictions['ensemble_predictions'][task] = {
                        'class_prediction': int(avg_prob > 0.5),
                        'probability': float(avg_prob),
                        'confidence': float(abs(avg_prob - 0.5) * 2),
                        'model_count': len(model_preds),
                        'individual_models': model_preds
                    }
                else:
                    # Multi-class: average probabilities across models
                    all_probs = []
                    for p in model_preds.values():
                        if 'class_probabilities' in p:
                            all_probs.append(p['class_probabilities'])
                    
                    if all_probs:
                        avg_probs = np.mean(all_probs, axis=0)
                        predicted_class = int(np.argmax(avg_probs))
                        
                        predictions['ensemble_predictions'][task] = {
                            'class_prediction': predicted_class,
                            'class_probabilities': avg_probs.tolist(),
                            'confidence': float(np.max(avg_probs)),
                            'model_count': len(model_preds),
                            'individual_models': model_preds
                        }
        
        predictions['task_predictions'] = task_predictions
        return predictions
    
    def _calculate_regression_confidence(self, prediction: float, task: str) -> float:
        """Calculate confidence score for regression predictions"""
        # Simple confidence based on prediction magnitude
        # This could be made more sophisticated with prediction intervals
        base_confidence = 0.7  # Base confidence
        
        if 'return' in task:
            # For return predictions, higher absolute values might be less confident
            magnitude_penalty = min(0.3, abs(prediction) * 5)
            return max(0.1, base_confidence - magnitude_penalty)
        elif 'volatility' in task:
            # For volatility, very low or very high values might be less confident
            if prediction < 0.001 or prediction > 0.1:
                return 0.5
            return base_confidence
        
        return base_confidence
    
    def get_model_status(self) -> Dict:
        """Get status of all loaded models"""
        status = {
            'total_models': len(self.models),
            'models_by_symbol': {},
            'models_by_task': {},
            'models_by_stage': {'Production': 0, 'Staging': 0, 'None': 0}
        }
        
        for model_key, metadata in self.model_metadata.items():
            parts = model_key.split('_')
            symbol = parts[0]
            task = '_'.join(parts[1:-1])
            model_type = parts[-1]
            stage = metadata.get('stage', 'None')
            
            # Count by symbol
            if symbol not in status['models_by_symbol']:
                status['models_by_symbol'][symbol] = 0
            status['models_by_symbol'][symbol] += 1
            
            # Count by task
            if task not in status['models_by_task']:
                status['models_by_task'][task] = 0
            status['models_by_task'][task] += 1
            
            # Count by stage
            status['models_by_stage'][stage] += 1
        
        return status