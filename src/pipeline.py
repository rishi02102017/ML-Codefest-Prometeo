"""
Main Pipeline Module - Project Kassandra
End-to-end ML pipeline for stock price prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import warnings

from data_fetcher import UniversalDataFetcher
from feature_engineering import FeatureEngineer, TemporalLeakageChecker
from model import StockPricePredictor, calculate_metrics, directional_accuracy
from prediction_logger import PredictionLogger, FeatureExporter

warnings.filterwarnings('ignore')


class KassandraPipeline:
    """
    Complete ML pipeline for stock price prediction.
    
    Usage:
        pipeline = KassandraPipeline()
        pipeline.run('TSLA', '2023-01-01', '2024-01-01')
        prediction = pipeline.predict_next_day()
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_fetcher = UniversalDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model = StockPricePredictor(model_type='ensemble')
        self.logger = PredictionLogger(output_dir)
        self.feature_exporter = FeatureExporter(output_dir)
        
        self.ticker = None
        self.raw_data = None
        self.featured_data = None
        self.is_trained = False
        self.latest_prediction = None
    
    def run(self, ticker: str, start_date: str, end_date: str,
            export_features: bool = True,
            run_validation: bool = True) -> dict:
        """
        Run the complete pipeline.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            export_features: Whether to export feature CSV
            run_validation: Whether to run walk-forward validation
            
        Returns:
            Dictionary with pipeline results
        """
        print("\n" + "="*70)
        print("PROJECT KASSANDRA - UNIVERSAL SENTIMENT ENGINE")
        print("="*70)
        print(f"Stock: {ticker}")
        print(f"Period: {start_date} to {end_date}")
        print("="*70 + "\n")
        
        self.ticker = ticker
        results = {}
        
        # Step 1: Fetch Data
        print("\n" + "-"*50)
        print("STEP 1: DATA ACQUISITION")
        print("-"*50)
        
        self.raw_data = self.data_fetcher.fetch_all(
            ticker, start_date, end_date,
            include_trends=True,
            include_news=True,
            include_reddit=True,
            include_wiki=True
        )
        
        results['data_points'] = len(self.raw_data)
        results['date_range'] = f"{self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}"
        
        # Step 2: Feature Engineering
        print("\n" + "-"*50)
        print("STEP 2: FEATURE ENGINEERING")
        print("-"*50)
        
        self.featured_data = self.feature_engineer.create_features(self.raw_data)
        results['total_features'] = len(self.feature_engineer.feature_columns)
        
        # Export features if requested
        if export_features:
            features_path = self.feature_exporter.export_features(
                self.featured_data, ticker, f"features_{ticker}.csv"
            )
            results['features_csv'] = features_path
        
        # Step 3: Temporal Leakage Check
        print("\n" + "-"*50)
        print("STEP 3: TEMPORAL LEAKAGE CHECK")
        print("-"*50)
        
        checker = TemporalLeakageChecker()
        leakage_report = checker.check_feature_leakage(
            self.featured_data, 
            self.feature_engineer.feature_columns
        )
        
        print(f"[CHECK] Leakage Check: {leakage_report['status']}")
        if leakage_report['warnings']:
            print(f"[WARN] Warnings: {leakage_report['warnings']}")
        
        results['leakage_check'] = leakage_report['status']
        
        # Step 4: Train/Test Split
        print("\n" + "-"*50)
        print("STEP 4: TRAIN/TEST SPLIT")
        print("-"*50)
        
        train_df, test_df = self.feature_engineer.prepare_train_test_split(
            self.featured_data, test_ratio=0.2, gap_days=1
        )
        
        # Validate split
        split_report = checker.validate_train_test_split(train_df, test_df)
        print(f"[CHECK] Split Validation: {split_report['status']}")
        
        X_train, y_train = self.feature_engineer.get_feature_matrix(train_df)
        X_test, y_test = self.feature_engineer.get_feature_matrix(test_df)
        
        # Step 5: Model Training
        print("\n" + "-"*50)
        print("STEP 5: MODEL TRAINING")
        print("-"*50)
        
        training_metrics = self.model.train(X_train, y_train)
        results['training_metrics'] = training_metrics
        self.is_trained = True
        
        # Step 6: Walk-Forward Validation (optional)
        if run_validation:
            print("\n" + "-"*50)
            print("STEP 6: WALK-FORWARD VALIDATION")
            print("-"*50)
            
            X_all, y_all = self.feature_engineer.get_feature_matrix(self.featured_data)
            validation_metrics = self.model.walk_forward_validation(X_all, y_all, n_splits=5)
            results['validation_metrics'] = validation_metrics
        
        # Step 7: Test Set Evaluation
        print("\n" + "-"*50)
        print("STEP 7: TEST SET EVALUATION")
        print("-"*50)
        
        test_predictions = self.model.predict(X_test)
        test_metrics = calculate_metrics(y_test.values, test_predictions)
        
        print(f"[RESULT] Test Set Performance:")
        print(f"   MAE: ${test_metrics['mae']:.2f}")
        print(f"   RMSE: ${test_metrics['rmse']:.2f}")
        print(f"   MAPE: {test_metrics['mape']:.2f}%")
        print(f"   R2: {test_metrics['r2']:.4f}")
        
        results['test_metrics'] = test_metrics
        
        # Log predictions
        self.logger.log_batch_predictions(
            test_df['Date'].values,
            ticker,
            y_test.values,
            test_predictions
        )
        
        # Step 8: Feature Importance
        print("\n" + "-"*50)
        print("STEP 8: FEATURE IMPORTANCE")
        print("-"*50)
        
        importance = self.model.get_feature_importance()
        print("[TOP 10] Important Features:")
        for _, row in importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        results['feature_importance'] = importance.to_dict('records')
        
        # Step 9: Export Results
        print("\n" + "-"*50)
        print("STEP 9: EXPORT RESULTS")
        print("-"*50)
        
        self.logger.export_competition_format(ticker)
        self.logger.export_metrics_report()
        
        # Save model
        model_path = self.output_dir / f"model_{ticker}.pkl"
        self.model.save(str(model_path))
        results['model_path'] = str(model_path)
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        
        return results
    
    def predict_next_day(self) -> dict:
        """
        Predict the next trading day's closing price.
        
        Returns:
            Dictionary with prediction details
        """
        if not self.is_trained:
            raise ValueError("Pipeline must be run first before making predictions")
        
        # Get the latest data point for prediction
        latest_row = self.featured_data.iloc[[-1]]
        X_latest = self.feature_engineer.get_prediction_features(self.featured_data)
        
        # Make prediction
        prediction = self.model.predict_next_day(X_latest)
        
        # Add context
        prediction['ticker'] = self.ticker
        prediction['last_known_date'] = str(latest_row['Date'].values[0])[:10]
        prediction['last_close'] = float(latest_row['Close'].values[0])
        prediction['predicted_change'] = prediction['predicted_price'] - prediction['last_close']
        prediction['predicted_change_pct'] = (prediction['predicted_change'] / prediction['last_close']) * 100
        prediction['signal'] = 'BUY' if prediction['predicted_change'] > 0 else 'SELL'
        
        self.latest_prediction = prediction
        
        return prediction
    
    def get_live_data(self) -> dict:
        """
        Get live/current data for dashboard.
        
        Returns:
            Dictionary with current market and sentiment data
        """
        if self.raw_data is None:
            return {}
        
        latest = self.raw_data.iloc[-1]
        
        data = {
            'ticker': self.ticker,
            'date': str(latest['Date'])[:10],
            'close': float(latest['Close']),
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'volume': int(latest['Volume']),
            'last_updated': datetime.now().isoformat()
        }
        
        # Add sentiment data if available
        if 'Google_Trends' in latest:
            data['google_trends'] = float(latest['Google_Trends']) if pd.notna(latest['Google_Trends']) else None
        if 'News_Sentiment' in latest:
            data['news_sentiment'] = float(latest['News_Sentiment']) if pd.notna(latest['News_Sentiment']) else None
        if 'Reddit_Sentiment' in latest:
            data['reddit_sentiment'] = float(latest['Reddit_Sentiment']) if pd.notna(latest['Reddit_Sentiment']) else None
        if 'Wiki_Views' in latest:
            data['wiki_views'] = int(latest['Wiki_Views']) if pd.notna(latest['Wiki_Views']) else None
        
        return data


def main():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Project Kassandra - Universal Sentiment Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --ticker TSLA --start 2023-01-01 --end 2024-01-01
  python pipeline.py --ticker AAPL --start 2023-01-01 --end 2024-01-01 --no-validation
        """
    )
    
    parser.add_argument('--ticker', '-t', type=str, required=True,
                       help='Stock ticker symbol (e.g., TSLA, AAPL)')
    parser.add_argument('--start', '-s', type=str, required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                       help='Output directory (default: outputs)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip walk-forward validation')
    parser.add_argument('--no-export', action='store_true',
                       help='Skip feature CSV export')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = KassandraPipeline(output_dir=args.output)
    results = pipeline.run(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        export_features=not args.no_export,
        run_validation=not args.no_validation
    )
    
    # Make next-day prediction
    print("\n" + "="*70)
    print("NEXT TRADING DAY PREDICTION")
    print("="*70)
    
    prediction = pipeline.predict_next_day()
    
    print(f"\n[PREDICTION] Predicted Closing Price: ${prediction['predicted_price']:.2f}")
    print(f"[INFO] Last Known Close: ${prediction['last_close']:.2f}")
    print(f"[INFO] Predicted Change: ${prediction['predicted_change']:.2f} ({prediction['predicted_change_pct']:.2f}%)")
    print(f"[SIGNAL] {prediction['signal']}")
    print(f"[CONFIDENCE] {prediction['confidence']:.1f}%")
    
    return results, prediction


if __name__ == "__main__":
    main()
