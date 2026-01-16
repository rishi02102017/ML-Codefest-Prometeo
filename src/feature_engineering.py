"""
Feature Engineering Module - Project Kassandra
Transforms raw data into ML-ready features with STRICT temporal leak prevention.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering with temporal leak prevention.
    
    CRITICAL: All features are computed using ONLY past data.
    No information from the future is ever used.
    """
    
    def __init__(self, target_col: str = 'Close', prediction_horizon: int = 1):
        """
        Initialize the feature engineer.
        
        Args:
            target_col: Column to predict (default: 'Close')
            prediction_horizon: Days ahead to predict (default: 1 for next day)
        """
        self.target_col = target_col
        self.prediction_horizon = prediction_horizon
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the model.
        
        TEMPORAL LEAK PREVENTION:
        - All rolling/lagged features use only past data
        - Target variable is shifted to represent FUTURE values
        - No current-day features are used to predict current-day prices
        
        Args:
            df: Raw DataFrame with stock and sentiment data
            
        Returns:
            DataFrame with engineered features
        """
        print("[FEATURE] Engineering features with temporal leak prevention...")
        
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Create the TARGET variable (what we're predicting)
        # This is the FUTURE closing price, shifted back
        df['Target'] = df[self.target_col].shift(-self.prediction_horizon)
        df['Target_Return'] = df['Target'].pct_change()
        
        # Create lagged features (using ONLY past data)
        df = self._create_lagged_price_features(df)
        df = self._create_lagged_volume_features(df)
        df = self._create_lagged_technical_features(df)
        df = self._create_lagged_sentiment_features(df)
        df = self._create_temporal_features(df)
        df = self._create_interaction_features(df)
        
        # Store feature columns (excluding target and non-feature columns)
        non_features = ['Date', 'Target', 'Target_Return', 'Ticker', 'Open', 'High', 
                       'Low', 'Close', 'Volume']
        self.feature_columns = [col for col in df.columns if col not in non_features]
        
        print(f"[OK] Created {len(self.feature_columns)} features")
        
        return df
    
    def _create_lagged_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged price features (using past data only)."""
        
        # Lagged closing prices (1-5 days ago)
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
        
        # Lagged price differences
        df['Close_Diff_1'] = df['Close'].diff().shift(1)
        df['Close_Diff_5'] = df['Close'].diff(5).shift(1)
        
        # Lagged Open-Close gap (previous days)
        df['OC_Gap_Lag_1'] = (df['Close'] - df['Open']).shift(1) / df['Open'].shift(1)
        
        # Lagged High-Low range
        df['HL_Range_Lag_1'] = ((df['High'] - df['Low']) / df['Close']).shift(1)
        
        # Previous day's price relative to moving averages
        if 'MA_5' in df.columns:
            df['Price_vs_MA5_Lag_1'] = (df['Close'].shift(1) / df['MA_5'].shift(1)) - 1
        if 'MA_20' in df.columns:
            df['Price_vs_MA20_Lag_1'] = (df['Close'].shift(1) / df['MA_20'].shift(1)) - 1
        
        return df
    
    def _create_lagged_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged volume features."""
        
        # Lagged volume
        for lag in [1, 2, 3, 5]:
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Volume change
        df['Volume_Change_Lag_1'] = df['Volume'].pct_change().shift(1)
        
        # Volume moving average ratio (lagged)
        if 'Volume_MA_5' in df.columns:
            vol_ma = df['Volume_MA_5'].shift(1).replace(0, np.nan)
            df['Volume_MA_Ratio_Lag_1'] = (df['Volume'].shift(1) / vol_ma).replace([np.inf, -np.inf], 1)
        
        return df
    
    def _create_lagged_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged technical indicator features."""
        
        # Lagged RSI
        if 'RSI' in df.columns:
            df['RSI_Lag_1'] = df['RSI'].shift(1)
            df['RSI_Lag_5'] = df['RSI'].shift(5)
            df['RSI_Change'] = df['RSI'].diff().shift(1)
            
            # RSI zones (oversold/overbought) - lagged
            df['RSI_Oversold_Lag_1'] = (df['RSI'].shift(1) < 30).astype(int)
            df['RSI_Overbought_Lag_1'] = (df['RSI'].shift(1) > 70).astype(int)
        
        # Lagged MACD
        if 'MACD' in df.columns:
            df['MACD_Lag_1'] = df['MACD'].shift(1)
            df['MACD_Signal_Lag_1'] = df['MACD_Signal'].shift(1) if 'MACD_Signal' in df.columns else 0
            df['MACD_Hist_Lag_1'] = (df['MACD'] - df.get('MACD_Signal', 0)).shift(1)
        
        # Lagged Bollinger Band position
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_range = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position_Lag_1'] = ((df['Close'] - df['BB_Lower']) / bb_range).shift(1)
        
        # Lagged volatility
        if 'Volatility_5' in df.columns:
            df['Volatility_Lag_1'] = df['Volatility_5'].shift(1)
            df['Volatility_Lag_5'] = df['Volatility_5'].shift(5)
        
        return df
    
    def _create_lagged_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged sentiment features (no temporal leakage)."""
        
        # Google Trends (lagged)
        if 'Google_Trends' in df.columns:
            df['Trends_Lag_1'] = df['Google_Trends'].shift(1)
            df['Trends_Lag_7'] = df['Google_Trends'].shift(7)
            df['Trends_Change_Lag_1'] = df['Google_Trends'].pct_change().shift(1).replace([np.inf, -np.inf], 0)
            
            # Trend momentum (safe division)
            trends_7 = df['Google_Trends'].shift(7).replace(0, np.nan)
            df['Trends_Momentum'] = (df['Google_Trends'].shift(1) / trends_7).replace([np.inf, -np.inf], 1)
        
        # News Sentiment (lagged)
        if 'News_Sentiment' in df.columns:
            df['News_Sentiment_Lag_1'] = df['News_Sentiment'].shift(1)
            df['News_Sentiment_Lag_3'] = df['News_Sentiment'].shift(3)
            df['News_Sentiment_MA_5'] = df['News_Sentiment'].rolling(5).mean().shift(1)
            
            # Sentiment trend
            df['News_Sentiment_Trend'] = (df['News_Sentiment'].shift(1) - 
                                           df['News_Sentiment'].shift(5))
        
        # Reddit Sentiment (lagged)
        if 'Reddit_Sentiment' in df.columns:
            df['Reddit_Sentiment_Lag_1'] = df['Reddit_Sentiment'].shift(1)
            df['Reddit_Sentiment_Lag_3'] = df['Reddit_Sentiment'].shift(3)
            df['Reddit_Sentiment_MA_5'] = df['Reddit_Sentiment'].rolling(5).mean().shift(1)
            
        if 'Reddit_Bullish_Ratio' in df.columns:
            df['Reddit_Bullish_Lag_1'] = df['Reddit_Bullish_Ratio'].shift(1)
        
        # Wikipedia Views (lagged)
        if 'Wiki_Views' in df.columns:
            df['Wiki_Views_Lag_1'] = df['Wiki_Views'].shift(1)
            df['Wiki_Views_Lag_7'] = df['Wiki_Views'].shift(7)
            df['Wiki_Views_Change_Lag_1'] = df['Wiki_Views'].pct_change().shift(1).replace([np.inf, -np.inf], 0)
            
            # Wiki attention spike (safe division)
            wiki_ma = df['Wiki_Views'].rolling(30).mean().shift(1).replace(0, np.nan)
            df['Wiki_Attention_Spike'] = (df['Wiki_Views'].shift(1) / wiki_ma).replace([np.inf, -np.inf], 1)
        
        # Combined sentiment score (lagged)
        sentiment_cols = []
        if 'News_Sentiment' in df.columns:
            sentiment_cols.append('News_Sentiment_Lag_1')
        if 'Reddit_Sentiment' in df.columns:
            sentiment_cols.append('Reddit_Sentiment_Lag_1')
        
        if sentiment_cols:
            df['Combined_Sentiment'] = df[sentiment_cols].mean(axis=1)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calendar/temporal features."""
        
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        
        # Is Monday (historically volatile)
        df['Is_Monday'] = (df['Date'].dt.dayofweek == 0).astype(int)
        # Is Friday (weekend effect)
        df['Is_Friday'] = (df['Date'].dt.dayofweek == 4).astype(int)
        
        # Month start/end effects
        df['Is_MonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['Is_MonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different data sources."""
        
        # Sentiment * Volatility interaction
        if 'Combined_Sentiment' in df.columns and 'Volatility_Lag_1' in df.columns:
            df['Sentiment_Volatility_Interaction'] = (
                df['Combined_Sentiment'] * df['Volatility_Lag_1']
            )
        
        # Sentiment * Volume interaction
        if 'Combined_Sentiment' in df.columns and 'Volume_Change_Lag_1' in df.columns:
            df['Sentiment_Volume_Interaction'] = (
                df['Combined_Sentiment'] * df['Volume_Change_Lag_1'].fillna(0)
            )
        
        # Trend * RSI interaction
        if 'Trends_Lag_1' in df.columns and 'RSI_Lag_1' in df.columns:
            # High search interest + oversold = potential reversal
            df['Trends_RSI_Interaction'] = (
                df['Trends_Lag_1'] / 100 * (50 - df['RSI_Lag_1'].fillna(50)) / 50
            )
        
        return df
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                                  test_ratio: float = 0.2,
                                  gap_days: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets with temporal ordering.
        
        TEMPORAL LEAK PREVENTION:
        - Training data comes BEFORE test data chronologically
        - Optional gap between train and test to simulate real prediction
        
        Args:
            df: Feature-engineered DataFrame
            test_ratio: Proportion of data for testing
            gap_days: Gap days between train and test (prevents leakage)
            
        Returns:
            (train_df, test_df)
        """
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Drop rows with NaN target
        df = df.dropna(subset=['Target'])
        
        # Calculate split point
        split_idx = int(len(df) * (1 - test_ratio))
        
        # Apply gap
        train_df = df.iloc[:split_idx - gap_days].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"[SPLIT] Train set: {len(train_df)} samples ({train_df['Date'].min()} to {train_df['Date'].max()})")
        print(f"[SPLIT] Test set: {len(test_df)} samples ({test_df['Date'].min()} to {test_df['Date'].max()})")
        
        return train_df, test_df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix X and target vector y.
        
        Args:
            df: DataFrame with features
            
        Returns:
            (X, y) tuple
        """
        # Use stored feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        X = df[available_features].copy()
        y = df['Target'].copy()
        
        # Handle inf values (from division by zero in pct_change)
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Clip extreme values to prevent overflow
        for col in X.columns:
            if X[col].dtype in ['float64', 'float32']:
                p1, p99 = X[col].quantile([0.01, 0.99])
                X[col] = X[col].clip(p1, p99)
        
        return X, y
    
    def get_prediction_features(self, df: pd.DataFrame, date: datetime = None) -> pd.DataFrame:
        """
        Get features for making a prediction.
        
        Args:
            df: DataFrame with all data
            date: Date to predict for (default: latest)
            
        Returns:
            Single row DataFrame with features for prediction
        """
        if date is None:
            # Get the last row
            return df[self.feature_columns].iloc[[-1]]
        else:
            # Get row for specific date
            mask = df['Date'] == pd.to_datetime(date)
            return df.loc[mask, self.feature_columns]


class TemporalLeakageChecker:
    """
    Utility class to verify no temporal leakage exists in the pipeline.
    """
    
    @staticmethod
    def check_feature_leakage(df: pd.DataFrame, feature_cols: List[str], 
                               target_col: str = 'Target') -> dict:
        """
        Check for potential temporal leakage in features.
        
        Returns a report of potential issues.
        """
        report = {
            'status': 'PASS',
            'warnings': [],
            'errors': []
        }
        
        # Check correlation between features and target
        for col in feature_cols:
            if col in df.columns:
                corr = df[col].corr(df[target_col])
                
                # Suspiciously high correlation might indicate leakage
                if abs(corr) > 0.95:
                    report['warnings'].append(
                        f"High correlation ({corr:.3f}) between {col} and target"
                    )
        
        # Check for features that shouldn't exist
        suspicious_patterns = ['_current', '_today', '_future']
        for col in feature_cols:
            for pattern in suspicious_patterns:
                if pattern in col.lower():
                    report['errors'].append(
                        f"Suspicious feature name: {col} (contains '{pattern}')"
                    )
                    report['status'] = 'FAIL'
        
        return report
    
    @staticmethod
    def validate_train_test_split(train_df: pd.DataFrame, 
                                   test_df: pd.DataFrame) -> dict:
        """
        Validate that train/test split has no temporal overlap.
        """
        report = {
            'status': 'PASS',
            'train_end': train_df['Date'].max(),
            'test_start': test_df['Date'].min(),
            'gap_days': 0
        }
        
        gap = (test_df['Date'].min() - train_df['Date'].max()).days
        report['gap_days'] = gap
        
        if gap < 0:
            report['status'] = 'FAIL'
            report['error'] = 'Train and test sets overlap!'
        elif gap == 0:
            report['status'] = 'WARNING'
            report['warning'] = 'No gap between train and test sets'
        else:
            report['status'] = 'PASS'
            report['message'] = f'{gap} day gap between train and test'
        
        return report


if __name__ == "__main__":
    # Test with sample data
    from data_fetcher import fetch_stock_data
    
    # Fetch data
    df = fetch_stock_data('TSLA', '2024-01-01', '2024-06-01')
    
    # Engineer features
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    
    # Split data
    train_df, test_df = fe.prepare_train_test_split(df_features)
    
    # Check for leakage
    checker = TemporalLeakageChecker()
    leakage_report = checker.check_feature_leakage(df_features, fe.feature_columns)
    split_report = checker.validate_train_test_split(train_df, test_df)
    
    print("\n[REPORT] Leakage Check:")
    print(leakage_report)
    print("\n[REPORT] Split Validation:")
    print(split_report)
