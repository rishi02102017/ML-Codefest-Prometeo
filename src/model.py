"""
ML Model Module - Project Kassandra
Stock price prediction model with ensemble approach.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import pickle
import json
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] XGBoost not available, using alternatives")

warnings.filterwarnings('ignore')


class StockPricePredictor:
    """
    Ensemble model for stock price prediction.
    
    Features:
    - Multiple base models (XGBoost, Random Forest, Gradient Boosting)
    - Ensemble averaging for robust predictions
    - Walk-forward validation for realistic backtesting
    - Feature importance analysis
    """
    
    def __init__(self, 
                 model_type: str = 'ensemble',
                 use_scaling: bool = True):
        """
        Initialize the predictor.
        
        Args:
            model_type: 'ensemble', 'xgboost', 'random_forest', or 'gradient_boosting'
            use_scaling: Whether to scale features
        """
        self.model_type = model_type
        self.use_scaling = use_scaling
        self.scaler = RobustScaler() if use_scaling else None
        self.models = {}
        self.feature_columns = []
        self.is_trained = False
        self.training_metrics = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the base models."""
        
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )
        
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        self.models['ridge'] = Ridge(alpha=1.0)
        
        print(f"[OK] Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_split: float = 0.15) -> Dict:
        """
        Train the model(s) on the provided data.
        
        Args:
            X: Feature matrix
            y: Target values
            validation_split: Proportion for validation
            
        Returns:
            Dictionary with training metrics
        """
        print("\n[TRAIN] Training Stock Price Predictor...")
        print(f"[INFO] Training samples: {len(X)}")
        print(f"[INFO] Features: {len(X.columns)}")
        
        self.feature_columns = list(X.columns)
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove any remaining NaN in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train each model
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            
            # Validate
            val_pred = model.predict(X_val)
            
            model_scores[name] = {
                'mae': mean_absolute_error(y_val, val_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'mape': mean_absolute_percentage_error(y_val, val_pred) * 100,
                'r2': r2_score(y_val, val_pred)
            }
            
            print(f"    MAE: ${model_scores[name]['mae']:.2f}, "
                  f"MAPE: {model_scores[name]['mape']:.2f}%")
        
        self.is_trained = True
        self.training_metrics = model_scores
        
        # Print ensemble performance
        if self.model_type == 'ensemble':
            ensemble_pred = self._ensemble_predict(X_val)
            ensemble_metrics = {
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, ensemble_pred)),
                'mape': mean_absolute_percentage_error(y_val, ensemble_pred) * 100,
                'r2': r2_score(y_val, ensemble_pred)
            }
            self.training_metrics['ensemble'] = ensemble_metrics
            print(f"\n  [ENSEMBLE] MAE: ${ensemble_metrics['mae']:.2f}, "
                  f"MAPE: {ensemble_metrics['mape']:.2f}%")
        
        print("\n[OK] Training complete!")
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure correct features
        X = X[self.feature_columns] if all(col in X.columns for col in self.feature_columns) else X
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale
        if self.use_scaling:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        if self.model_type == 'ensemble':
            return self._ensemble_predict(X_scaled)
        else:
            return self.models[self.model_type].predict(X_scaled)
    
    def _ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble prediction by averaging all models."""
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        # Simple average ensemble
        return np.mean(predictions, axis=0)
    
    def predict_next_day(self, X_latest: pd.DataFrame) -> Dict:
        """
        Predict the next trading day's closing price.
        
        Args:
            X_latest: Latest feature row
            
        Returns:
            Dictionary with prediction and confidence metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        for name, model in self.models.items():
            X_proc = X_latest.copy()
            X_proc = X_proc.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            if self.use_scaling:
                X_scaled = self.scaler.transform(X_proc)
            else:
                X_scaled = X_proc.values
            
            predictions[name] = model.predict(X_scaled)[0]
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Confidence based on agreement between models
        pred_std = np.std(list(predictions.values()))
        confidence = max(0, 1 - (pred_std / ensemble_pred)) * 100
        
        return {
            'predicted_price': ensemble_pred,
            'model_predictions': predictions,
            'prediction_std': pred_std,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        importance_data = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, col in enumerate(self.feature_columns):
                    importance_data.append({
                        'feature': col,
                        'model': name,
                        'importance': model.feature_importances_[i]
                    })
        
        df = pd.DataFrame(importance_data)
        
        if len(df) > 0:
            # Aggregate importance across models
            avg_importance = df.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('importance', ascending=False)
            return avg_importance
        
        return df
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series,
                                  n_splits: int = 5) -> Dict:
        """
        Perform walk-forward validation (time series cross-validation).
        
        This is the PROPER way to validate time series models,
        ensuring no temporal leakage.
        
        Args:
            X: Feature matrix
            y: Target values
            n_splits: Number of time series splits
            
        Returns:
            Dictionary with validation metrics
        """
        print(f"\n[VALIDATION] Walk-Forward Validation ({n_splits} splits)...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_metrics = []
        all_predictions = []
        all_actuals = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle missing values in features
            X_train = X_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
            X_val = X_val.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Handle NaN in target - skip folds with NaN targets
            if y_train.isna().any() or y_val.isna().any():
                # Filter out NaN samples
                train_mask = ~y_train.isna()
                val_mask = ~y_val.isna()
                X_train = X_train[train_mask]
                y_train = y_train[train_mask]
                X_val = X_val[val_mask]
                y_val = y_val[val_mask]
                
                if len(y_val) == 0:
                    print(f"  Fold {fold + 1}: Skipped (no valid samples)")
                    continue
            
            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train and predict with each model
            fold_preds = []
            for name, model in self.models.items():
                model_clone = self._clone_model(model)
                model_clone.fit(X_train_scaled, y_train)
                pred = model_clone.predict(X_val_scaled)
                fold_preds.append(pred)
            
            # Ensemble prediction
            ensemble_pred = np.mean(fold_preds, axis=0)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            mape = mean_absolute_percentage_error(y_val, ensemble_pred) * 100
            
            fold_metrics.append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            })
            
            all_predictions.extend(ensemble_pred)
            all_actuals.extend(y_val.values)
            
            print(f"  Fold {fold + 1}: MAE=${mae:.2f}, MAPE={mape:.2f}%")
        
        # Overall metrics
        overall_metrics = {
            'mean_mae': np.mean([m['mae'] for m in fold_metrics]),
            'mean_rmse': np.mean([m['rmse'] for m in fold_metrics]),
            'mean_mape': np.mean([m['mape'] for m in fold_metrics]),
            'std_mae': np.std([m['mae'] for m in fold_metrics]),
            'fold_metrics': fold_metrics
        }
        
        print(f"\n  [OVERALL] MAE=${overall_metrics['mean_mae']:.2f} +/- "
              f"${overall_metrics['std_mae']:.2f}")
        
        return overall_metrics
    
    def _clone_model(self, model):
        """Create a clone of a model with same parameters."""
        from sklearn.base import clone
        return clone(model)
    
    def save(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_metrics': self.training_metrics,
            'model_type': self.model_type,
            'use_scaling': self.use_scaling
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"[SAVE] Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.feature_columns = save_data['feature_columns']
        self.training_metrics = save_data['training_metrics']
        self.model_type = save_data['model_type']
        self.use_scaling = save_data['use_scaling']
        self.is_trained = True
        
        print(f"[LOAD] Model loaded from {filepath}")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Calculate comprehensive prediction metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2': r2_score(y_true, y_pred),
        'max_error': np.max(np.abs(y_true - y_pred)),
        'mean_error': np.mean(y_pred - y_true),  # Bias
        'correlation': np.corrcoef(y_true, y_pred)[0, 1]
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prev: np.ndarray) -> float:
    """
    Calculate directional accuracy (did we predict up/down correctly?).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_prev: Previous day values
        
    Returns:
        Directional accuracy percentage
    """
    actual_direction = np.sign(y_true - y_prev)
    pred_direction = np.sign(y_pred - y_prev)
    
    return np.mean(actual_direction == pred_direction) * 100


if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import fetch_stock_data
    from feature_engineering import FeatureEngineer
    
    # Fetch and prepare data
    df = fetch_stock_data('TSLA', '2023-01-01', '2024-01-01')
    
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    
    train_df, test_df = fe.prepare_train_test_split(df_features)
    
    X_train, y_train = fe.get_feature_matrix(train_df)
    X_test, y_test = fe.get_feature_matrix(test_df)
    
    # Train model
    predictor = StockPricePredictor(model_type='ensemble')
    metrics = predictor.train(X_train, y_train)
    
    # Test predictions
    predictions = predictor.predict(X_test)
    test_metrics = calculate_metrics(y_test.values, predictions)
    
    print("\n[METRICS] Test Set:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print("\n[TOP FEATURES]")
    print(importance.head(10))
