"""
Prediction Logger Module - Project Kassandra
Logs predictions and generates CSV reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional


class PredictionLogger:
    """
    Logs predictions and exports to CSV for evaluation.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.predictions = []
        self.metrics_history = []
    
    def log_prediction(self, 
                       date: datetime,
                       ticker: str,
                       actual_price: float,
                       predicted_price: float,
                       model_predictions: Dict[str, float] = None,
                       confidence: float = None,
                       additional_data: Dict = None):
        """
        Log a single prediction.
        
        Args:
            date: Date of the prediction
            ticker: Stock ticker
            actual_price: Actual closing price
            predicted_price: Predicted closing price
            model_predictions: Individual model predictions
            confidence: Prediction confidence score
            additional_data: Any additional data to log
        """
        record = {
            'Date': pd.to_datetime(date),
            'Ticker': ticker,
            'Actual_Close': actual_price,
            'Predicted_Close': predicted_price,
            'Absolute_Error': abs(actual_price - predicted_price),
            'Percentage_Error': abs(actual_price - predicted_price) / actual_price * 100,
            'Prediction_Direction': 'Correct' if (predicted_price - actual_price) * (actual_price - self._get_prev_actual()) >= 0 else 'Incorrect',
            'Confidence': confidence,
            'Timestamp': datetime.now().isoformat()
        }
        
        # Add individual model predictions
        if model_predictions:
            for model_name, pred in model_predictions.items():
                record[f'Pred_{model_name}'] = pred
        
        # Add additional data
        if additional_data:
            record.update(additional_data)
        
        self.predictions.append(record)
    
    def _get_prev_actual(self) -> float:
        """Get previous actual price for direction calculation."""
        if len(self.predictions) > 0:
            return self.predictions[-1].get('Actual_Close', 0)
        return 0
    
    def log_batch_predictions(self, 
                               dates: List[datetime],
                               ticker: str,
                               actual_prices: np.ndarray,
                               predicted_prices: np.ndarray):
        """
        Log multiple predictions at once.
        
        Args:
            dates: List of dates
            ticker: Stock ticker
            actual_prices: Array of actual prices
            predicted_prices: Array of predicted prices
        """
        for i, (date, actual, predicted) in enumerate(zip(dates, actual_prices, predicted_prices)):
            prev_actual = actual_prices[i-1] if i > 0 else actual
            
            record = {
                'Date': pd.to_datetime(date),
                'Ticker': ticker,
                'Actual_Close': actual,
                'Predicted_Close': predicted,
                'Absolute_Error': abs(actual - predicted),
                'Percentage_Error': abs(actual - predicted) / actual * 100 if actual > 0 else 0,
                'Timestamp': datetime.now().isoformat()
            }
            
            self.predictions.append(record)
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate summary metrics from logged predictions.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.predictions:
            return {}
        
        df = pd.DataFrame(self.predictions)
        
        actuals = df['Actual_Close'].values
        predictions = df['Predicted_Close'].values
        
        # Basic metrics
        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mape = np.mean(np.abs(actuals - predictions) / actuals) * 100
        
        # Directional accuracy
        if len(df) > 1:
            actual_direction = np.sign(np.diff(actuals))
            pred_direction = np.sign(predictions[1:] - actuals[:-1])
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = None
        
        metrics = {
            'total_predictions': len(df),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'max_error': df['Absolute_Error'].max(),
            'min_error': df['Absolute_Error'].min(),
            'avg_percentage_error': df['Percentage_Error'].mean(),
            'directional_accuracy': directional_accuracy,
            'prediction_bias': np.mean(predictions - actuals)  # Positive = overestimate
        }
        
        return metrics
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export predictions to CSV file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to the exported file
        """
        if not self.predictions:
            print("[WARN] No predictions to export")
            return None
        
        df = pd.DataFrame(self.predictions)
        
        # Ensure required columns are present and formatted
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Select columns for output
        output_cols = ['Date', 'Actual_Close', 'Predicted_Close']
        
        # Add optional columns if present
        for col in ['Absolute_Error', 'Percentage_Error', 'Confidence', 'Ticker']:
            if col in df.columns:
                output_cols.append(col)
        
        df_export = df[output_cols].copy()
        
        # Round numerical columns
        for col in ['Actual_Close', 'Predicted_Close', 'Absolute_Error']:
            if col in df_export.columns:
                df_export[col] = df_export[col].round(2)
        if 'Percentage_Error' in df_export.columns:
            df_export['Percentage_Error'] = df_export['Percentage_Error'].round(4)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            ticker = df['Ticker'].iloc[0] if 'Ticker' in df.columns else 'STOCK'
            filename = f"predictions_{ticker}_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        df_export.to_csv(filepath, index=False)
        
        print(f"[EXPORT] Predictions exported to {filepath}")
        return str(filepath)
    
    def export_competition_format(self, ticker: str, filename: str = None) -> str:
        """
        Export predictions in competition-required format.
        
        Columns: Date, Actual Closing Price, Predicted Closing Price
        
        Args:
            ticker: Stock ticker
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        if not self.predictions:
            print("[WARN] No predictions to export")
            return None
        
        df = pd.DataFrame(self.predictions)
        
        # Format for competition
        df_competition = pd.DataFrame({
            'Date': pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d'),
            'Actual Closing Price': df['Actual_Close'].round(2),
            'Predicted Closing Price': df['Predicted_Close'].round(2)
        })
        
        if filename is None:
            filename = f"prediction_log_{ticker}.csv"
        
        filepath = self.output_dir / filename
        df_competition.to_csv(filepath, index=False)
        
        print(f"[EXPORT] Competition format exported to {filepath}")
        return str(filepath)
    
    def export_metrics_report(self, filename: str = None) -> str:
        """
        Export metrics report as text file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("[WARN] No metrics to export")
            return None
        
        if filename is None:
            filename = "metrics_report.txt"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PROJECT KASSANDRA - PREDICTION METRICS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("EVALUATION METRICS\n")
            f.write("-" * 40 + "\n")
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"[EXPORT] Metrics report exported to {filepath}")
        return str(filepath)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get predictions as DataFrame."""
        return pd.DataFrame(self.predictions)
    
    def clear(self):
        """Clear all logged predictions."""
        self.predictions = []
        self.metrics_history = []


class FeatureExporter:
    """
    Exports processed features to CSV for reproducibility.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_features(self, df: pd.DataFrame, ticker: str, 
                        filename: str = None) -> str:
        """
        Export processed features to CSV.
        
        Args:
            df: DataFrame with processed features
            ticker: Stock ticker
            filename: Output filename
            
        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"features_{ticker}_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Round numerical columns for cleaner output
        df_export = df.copy()
        for col in df_export.select_dtypes(include=[np.number]).columns:
            df_export[col] = df_export[col].round(6)
        
        df_export.to_csv(filepath, index=False)
        
        print(f"[EXPORT] Features exported to {filepath}")
        return str(filepath)


if __name__ == "__main__":
    # Test the logger
    logger = PredictionLogger()
    
    # Log some sample predictions
    dates = pd.date_range('2024-01-01', periods=10)
    actuals = np.random.uniform(200, 250, 10)
    predictions = actuals + np.random.normal(0, 5, 10)
    
    logger.log_batch_predictions(dates, 'TSLA', actuals, predictions)
    
    # Calculate and print metrics
    metrics = logger.calculate_metrics()
    print("\n[METRICS]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    
    # Export
    logger.export_to_csv()
    logger.export_competition_format('TSLA')
    logger.export_metrics_report()
