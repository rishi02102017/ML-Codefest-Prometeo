#!/usr/bin/env python3
"""
Quick Run Script - Project Kassandra
Simplified interface to run the prediction pipeline.

Usage:
    python run.py TSLA 2023-01-01 2024-01-01
    python run.py AAPL --start 2023-01-01 --end 2024-01-01
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import KassandraPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Project Kassandra - Universal Sentiment Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py TSLA                              # Run with TSLA, last 1 year
  python run.py AAPL --start 2023-01-01           # Custom start date
  python run.py GOOGL -s 2023-01-01 -e 2024-01-01 # Full date range
        """
    )
    
    parser.add_argument('ticker', type=str, nargs='?', default='TSLA',
                       help='Stock ticker symbol (default: TSLA)')
    parser.add_argument('--start', '-s', type=str, 
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date YYYY-MM-DD (default: 1 year ago)')
    parser.add_argument('--end', '-e', type=str,
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick mode: skip validation')
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    print(f"""
================================================================================
                         PROJECT KASSANDRA
                    Universal Sentiment Engine
================================================================================
  Stock: {ticker}
  Period: {args.start} to {args.end}
================================================================================
    """)
    
    # Run pipeline
    pipeline = KassandraPipeline()
    results = pipeline.run(
        ticker=ticker,
        start_date=args.start,
        end_date=args.end,
        export_features=True,
        run_validation=not args.quick
    )
    
    # Get prediction
    prediction = pipeline.predict_next_day()
    
    # Display results
    change_sign = '+' if prediction['predicted_change'] >= 0 else ''
    pct_sign = '+' if prediction['predicted_change_pct'] >= 0 else ''
    
    print(f"""
================================================================================
                         PREDICTION RESULTS
================================================================================

  Predicted Next Day Close:    ${prediction['predicted_price']:>10.2f}
  Last Known Close:            ${prediction['last_close']:>10.2f}
  Predicted Change:            ${change_sign}{prediction['predicted_change']:>9.2f} ({pct_sign}{prediction['predicted_change_pct']:.2f}%)
  Signal:                      {prediction['signal']:<10}
  Confidence:                  {prediction['confidence']:>10.1f}%

================================================================================

Output files saved to: outputs/
""")
    
    return results, prediction


if __name__ == "__main__":
    main()
