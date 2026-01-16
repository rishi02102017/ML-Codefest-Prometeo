#!/usr/bin/env python3
"""
Data Check Script - Project Kassandra (Phase 1 Deliverable)
Fetches 1 year of TSLA historical data and alternative feature data.

Run: python data_check.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data_fetcher import (
    StockDataFetcher,
    GoogleTrendsFetcher,
    NewsSentimentFetcher,
    RedditSentimentFetcher,
    WikipediaViewsFetcher,
    UniversalDataFetcher
)


def main():
    """Main data check function."""
    print("\n" + "="*70)
    print("PROJECT KASSANDRA - DATA CHECK SCRIPT")
    print("="*70)
    print("This script verifies data fetching capabilities for all sources.\n")
    
    # Configuration
    TICKER = "TSLA"
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Stock: {TICKER}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("-"*70 + "\n")
    
    results = {}
    
    # 1. Stock Data
    print("1. FETCHING STOCK DATA (yfinance)")
    print("-"*40)
    try:
        stock_fetcher = StockDataFetcher()
        stock_data = stock_fetcher.fetch(TICKER, START_DATE, END_DATE)
        results['stock'] = stock_data
        print(f"   [OK] Retrieved {len(stock_data)} trading days")
        print(f"   Columns: {list(stock_data.columns[:8])}...")
        print(f"   Date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
        print(f"   Latest close: ${stock_data['Close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        results['stock'] = None
    
    print()
    
    # 2. Google Trends
    print("2. FETCHING GOOGLE TRENDS DATA")
    print("-"*40)
    try:
        trends_fetcher = GoogleTrendsFetcher()
        trends_data = trends_fetcher.fetch("Tesla", START_DATE, END_DATE)
        results['trends'] = trends_data
        if not trends_data.empty:
            print(f"   [OK] Retrieved {len(trends_data)} data points")
            print(f"   Average interest: {trends_data['Google_Trends'].mean():.1f}")
        else:
            print("   [WARN] No data returned (API might be rate-limited)")
    except Exception as e:
        print(f"   [WARN] {e}")
        results['trends'] = None
    
    print()
    
    # 3. News Sentiment
    print("3. FETCHING NEWS SENTIMENT")
    print("-"*40)
    try:
        news_fetcher = NewsSentimentFetcher()
        news_data = news_fetcher.fetch(TICKER, "Tesla", START_DATE, END_DATE)
        results['news'] = news_data
        if not news_data.empty:
            print(f"   [OK] Analyzed sentiment for date range")
            print(f"   Average sentiment: {news_data['News_Sentiment'].mean():.3f}")
        else:
            print("   [WARN] Limited news data available")
    except Exception as e:
        print(f"   [WARN] {e}")
        results['news'] = None
    
    print()
    
    # 4. Reddit Sentiment
    print("4. FETCHING REDDIT SENTIMENT")
    print("-"*40)
    try:
        reddit_fetcher = RedditSentimentFetcher()
        reddit_data = reddit_fetcher.fetch(TICKER, START_DATE, END_DATE)
        results['reddit'] = reddit_data
        if not reddit_data.empty:
            print(f"   [OK] Analyzed sentiment for date range")
            print(f"   Average sentiment: {reddit_data['Reddit_Sentiment'].mean():.3f}")
        else:
            print("   [WARN] Limited Reddit data available")
    except Exception as e:
        print(f"   [WARN] {e}")
        results['reddit'] = None
    
    print()
    
    # 5. Wikipedia Views
    print("5. FETCHING WIKIPEDIA VIEWS")
    print("-"*40)
    try:
        wiki_fetcher = WikipediaViewsFetcher()
        wiki_data = wiki_fetcher.fetch("Tesla,_Inc.", START_DATE, END_DATE)
        results['wiki'] = wiki_data
        if not wiki_data.empty:
            print(f"   [OK] Retrieved {len(wiki_data)} data points")
            print(f"   Average daily views: {wiki_data['Wiki_Views'].mean():,.0f}")
        else:
            print("   [WARN] No Wikipedia data returned")
    except Exception as e:
        print(f"   [WARN] {e}")
        results['wiki'] = None
    
    print()
    
    # 6. Combined Data Fetch
    print("6. FETCHING COMBINED DATA (UniversalDataFetcher)")
    print("-"*40)
    try:
        fetcher = UniversalDataFetcher()
        combined_data = fetcher.fetch_all(TICKER, START_DATE, END_DATE)
        results['combined'] = combined_data
        print(f"   [OK] Complete")
        print(f"   Total features: {len(combined_data.columns)}")
        print(f"   Trading days: {len(combined_data)}")
    except Exception as e:
        print(f"   [ERROR] {e}")
        results['combined'] = None
    
    # Summary
    print("\n" + "="*70)
    print("DATA CHECK SUMMARY")
    print("="*70)
    
    checks = [
        ("Stock Data (yfinance)", results.get('stock') is not None and len(results['stock']) > 0),
        ("Google Trends", results.get('trends') is not None and len(results['trends']) > 0),
        ("News Sentiment", results.get('news') is not None and len(results['news']) > 0),
        ("Reddit Sentiment", results.get('reddit') is not None and len(results['reddit']) > 0),
        ("Wikipedia Views", results.get('wiki') is not None and len(results['wiki']) > 0),
        ("Combined Data", results.get('combined') is not None and len(results['combined']) > 0),
    ]
    
    for name, passed in checks:
        status = "[PASS]" if passed else "[PARTIAL]"
        print(f"   {status} {name}")
    
    # Save sample data
    if results.get('combined') is not None:
        output_path = "outputs/data_check_sample.csv"
        os.makedirs("outputs", exist_ok=True)
        results['combined'].head(50).to_csv(output_path, index=False)
        print(f"\nSample data saved to: {output_path}")
    
    print("\n" + "="*70)
    print("DATA CHECK COMPLETE")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    main()
