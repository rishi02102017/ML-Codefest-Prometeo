"""
Data Fetcher Module - Project Kassandra
Fetches historical stock data and alternative sentiment data from various sources.
All data is fetched LIVE - no local CSV files or synthetic data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import requests
from bs4 import BeautifulSoup
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import warnings
import random

warnings.filterwarnings('ignore')


class StockDataFetcher:
    """Fetches historical stock price data using multiple sources."""
    
    def __init__(self):
        pass
    
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data using multiple methods.
        Tries yahooquery first (uses curl-cffi to avoid blocks), then yfinance.
        """
        print(f"[DATA] Fetching stock data for {ticker} from {start_date} to {end_date}...")
        
        df = None
        
        # Method 1: Try yahooquery (most reliable, uses curl-cffi)
        df = self._fetch_yahooquery(ticker, start_date, end_date)
        
        # Method 2: Try yfinance as fallback
        if df is None or df.empty:
            df = self._fetch_yfinance(ticker, start_date, end_date)
        
        # Method 3: Try pandas-datareader as last resort
        if df is None or df.empty:
            df = self._fetch_pandas_datareader(ticker, start_date, end_date)
        
        # Validate we have real data
        if df is None or df.empty:
            raise ValueError(f"CRITICAL: Could not fetch real data for {ticker}. "
                           "Please check internet connection and try again.")
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Validate data quality
        if len(df) < 10:
            raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows")
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        print(f"[OK] Final dataset: {len(df)} trading days, {len(df.columns)} columns")
        return df
    
    def _fetch_yahooquery(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch using yahooquery library."""
        try:
            from yahooquery import Ticker
            
            print("[INFO] Trying yahooquery...")
            time.sleep(2)
            
            stock = Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df is not None and not df.empty:
                # Reset multi-index if present
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                else:
                    df = df.reset_index()
                
                # Rename columns to standard format
                col_map = {
                    'date': 'Date',
                    'open': 'Open', 
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'adjclose': 'Adj Close'
                }
                df = df.rename(columns=col_map)
                
                # Handle symbol column if present
                if 'symbol' in df.columns:
                    df = df.drop(columns=['symbol'])
                
                if len(df) > 10:
                    print(f"[OK] yahooquery fetched {len(df)} rows")
                    return df
                    
        except Exception as e:
            print(f"[WARN] yahooquery failed: {e}")
        
        return None
    
    def _fetch_yfinance(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch using yfinance library."""
        try:
            import yfinance as yf
            
            print("[INFO] Trying yfinance...")
            
            for attempt in range(3):
                try:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    
                    df = yf.download(ticker, start=start_date, end=end_date, 
                                    progress=False, auto_adjust=True)
                    
                    if df is not None and not df.empty and len(df) > 10:
                        df = df.reset_index()
                        print(f"[OK] yfinance fetched {len(df)} rows")
                        return df
                        
                except Exception as e:
                    print(f"[WARN] yfinance attempt {attempt + 1} failed: {e}")
                    
        except Exception as e:
            print(f"[WARN] yfinance failed: {e}")
        
        return None
    
    def _fetch_pandas_datareader(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch using pandas-datareader with FRED or other sources."""
        try:
            import pandas_datareader as pdr
            
            print("[INFO] Trying pandas-datareader...")
            time.sleep(2)
            
            # Try stooq as data source
            df = pdr.DataReader(ticker, 'stooq', start=start_date, end=end_date)
            
            if df is not None and not df.empty:
                df = df.reset_index()
                df = df.sort_values('Date')
                print(f"[OK] pandas-datareader fetched {len(df)} rows")
                return df
                
        except Exception as e:
            print(f"[WARN] pandas-datareader failed: {e}")
        
        return None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and format."""
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Standardize column names
        col_map = {
            'date': 'Date', 'DATE': 'Date',
            'open': 'Open', 'OPEN': 'Open',
            'high': 'High', 'HIGH': 'High',
            'low': 'Low', 'LOW': 'Low',
            'close': 'Close', 'CLOSE': 'Close',
            'volume': 'Volume', 'VOLUME': 'Volume',
            'adj close': 'Adj Close', 'Adj Close': 'Adj Close'
        }
        df = df.rename(columns=col_map)
        
        # Ensure Date column exists
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        
        # Convert Date to datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Keep only essential columns
        essential = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available = [c for c in essential if c in df.columns]
        df = df[available].copy()
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Fill small gaps
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the stock data."""
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_Abs'] = df['Close'].diff()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI (with safe division)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss_safe = loss.replace(0, np.nan)
        rs = gain / loss_safe
        rs = rs.replace([np.inf, -np.inf], 100)  # When loss is 0, RSI should be 100
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI'] = df['RSI'].fillna(50)  # Default to neutral
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        bb_mid = df['BB_Middle'].replace(0, np.nan)
        df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / bb_mid).replace([np.inf, -np.inf], 0)
        
        # Volatility
        df['Volatility_5'] = df['Price_Change'].rolling(window=5).std()
        df['Volatility_20'] = df['Price_Change'].rolling(window=20).std()
        
        # Volume features
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        vol_ma_20 = df['Volume_MA_20'].replace(0, np.nan)
        df['Volume_Ratio'] = (df['Volume'] / vol_ma_20).replace([np.inf, -np.inf], 1)
        
        # Daily range
        close_safe = df['Close'].replace(0, np.nan)
        df['Daily_Range'] = ((df['High'] - df['Low']) / close_safe).replace([np.inf, -np.inf], 0)
        df['Daily_Range_MA'] = df['Daily_Range'].rolling(window=10).mean()
        
        return df


class GoogleTrendsFetcher:
    """Fetches Google Trends data for sentiment analysis."""
    
    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=3, backoff_factor=0.5)
    
    def fetch(self, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Google Trends data for a keyword."""
        print(f"[DATA] Fetching Google Trends for '{keyword}'...")
        
        for attempt in range(3):
            try:
                wait_time = (attempt + 1) * 5 + random.uniform(0, 3)
                time.sleep(wait_time)
                
                timeframe = f"{start_date} {end_date}"
                self.pytrends.build_payload([keyword], timeframe=timeframe)
                
                df = self.pytrends.interest_over_time()
                
                if df.empty:
                    continue
                
                df = df.reset_index()
                df = df.rename(columns={keyword: 'Google_Trends', 'date': 'Date'})
                df['Date'] = pd.to_datetime(df['Date'])
                
                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])
                
                df['Trends_Change'] = df['Google_Trends'].pct_change()
                df['Trends_MA_7'] = df['Google_Trends'].rolling(window=7).mean()
                
                print(f"[OK] Fetched {len(df)} data points from Google Trends")
                return df
                
            except Exception as e:
                print(f"[WARN] Google Trends attempt {attempt + 1} failed: {e}")
        
        print("[WARN] Google Trends unavailable")
        return pd.DataFrame()


class NewsSentimentFetcher:
    """Fetches and analyzes news sentiment from RSS feeds."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.rss_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        ]
    
    def fetch(self, ticker: str, company_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch news and analyze sentiment."""
        print(f"[DATA] Fetching news sentiment for {ticker} ({company_name})...")
        
        all_news = []
        search_terms = [ticker.lower(), company_name.lower()]
        
        # Yahoo Finance RSS
        try:
            url = self.rss_sources[0].format(ticker=ticker)
            feed = feedparser.parse(url)
            for entry in feed.entries[:50]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                if title:
                    sentiment = self.vader.polarity_scores(f"{title} {summary}")
                    all_news.append({
                        'source': 'yahoo',
                        'title': title,
                        'sentiment_compound': sentiment['compound'],
                        'sentiment_pos': sentiment['pos'],
                        'sentiment_neg': sentiment['neg'],
                    })
            print(f"[INFO] Yahoo: {len([n for n in all_news if n['source']=='yahoo'])} articles")
        except Exception as e:
            print(f"[WARN] Yahoo RSS: {e}")
        
        # MarketWatch RSS
        try:
            time.sleep(1)
            feed = feedparser.parse(self.rss_sources[1])
            for entry in feed.entries[:100]:
                title = entry.get('title', '')
                if any(term in title.lower() for term in search_terms):
                    sentiment = self.vader.polarity_scores(title)
                    all_news.append({
                        'source': 'marketwatch',
                        'title': title,
                        'sentiment_compound': sentiment['compound'],
                        'sentiment_pos': sentiment['pos'],
                        'sentiment_neg': sentiment['neg'],
                    })
            print(f"[INFO] MarketWatch: {len([n for n in all_news if n['source']=='marketwatch'])} articles")
        except Exception as e:
            print(f"[WARN] MarketWatch RSS: {e}")
        
        # CNBC RSS
        try:
            time.sleep(1)
            feed = feedparser.parse(self.rss_sources[2])
            for entry in feed.entries[:100]:
                title = entry.get('title', '')
                if any(term in title.lower() for term in search_terms):
                    sentiment = self.vader.polarity_scores(title)
                    all_news.append({
                        'source': 'cnbc',
                        'title': title,
                        'sentiment_compound': sentiment['compound'],
                        'sentiment_pos': sentiment['pos'],
                        'sentiment_neg': sentiment['neg'],
                    })
            print(f"[INFO] CNBC: {len([n for n in all_news if n['source']=='cnbc'])} articles")
        except Exception as e:
            print(f"[WARN] CNBC RSS: {e}")
        
        if not all_news:
            print("[WARN] No news articles found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_news)
        print(f"[OK] Total: {len(df)} articles analyzed")
        
        return self._aggregate_daily_sentiment(df, start_date, end_date)
    
    def _aggregate_daily_sentiment(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Aggregate to daily level."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        return pd.DataFrame({
            'Date': dates,
            'News_Sentiment': df['sentiment_compound'].mean() if len(df) > 0 else 0,
            'News_Sentiment_Pos': df['sentiment_pos'].mean() if len(df) > 0 else 0.33,
            'News_Sentiment_Neg': df['sentiment_neg'].mean() if len(df) > 0 else 0.33,
            'News_Count': len(df)
        })


class RedditSentimentFetcher:
    """Fetches sentiment from Reddit using public RSS feeds."""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket', 'options']
    
    def fetch(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Reddit sentiment."""
        print(f"[DATA] Fetching Reddit sentiment for {ticker}...")
        
        all_posts = []
        
        for subreddit in self.subreddits:
            try:
                url = f"https://www.reddit.com/r/{subreddit}/search.rss?q={ticker}&restrict_sr=1&sort=new&limit=100"
                headers = {'User-Agent': 'Mozilla/5.0 ProjectKassandra/1.0'}
                
                response = requests.get(url, headers=headers, timeout=15)
                
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries[:25]:
                        title = entry.get('title', '')
                        content = entry.get('summary', '')
                        sentiment = self.vader.polarity_scores(f"{title} {content}")
                        
                        all_posts.append({
                            'subreddit': subreddit,
                            'title': title,
                            'sentiment_compound': sentiment['compound'],
                            'sentiment_pos': sentiment['pos'],
                            'sentiment_neg': sentiment['neg']
                        })
                    
                    print(f"[INFO] r/{subreddit}: {len([p for p in all_posts if p['subreddit']==subreddit])} posts")
                
                time.sleep(2)
                
            except Exception as e:
                print(f"[WARN] r/{subreddit}: {e}")
        
        if not all_posts:
            print("[WARN] No Reddit posts found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_posts)
        print(f"[OK] Total: {len(df)} Reddit posts")
        
        return self._aggregate_reddit_sentiment(df, start_date, end_date)
    
    def _aggregate_reddit_sentiment(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Aggregate to daily level."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        return pd.DataFrame({
            'Date': dates,
            'Reddit_Sentiment': df['sentiment_compound'].mean() if len(df) > 0 else 0,
            'Reddit_Volume': len(df),
            'Reddit_Bullish_Ratio': (df['sentiment_compound'] > 0.1).mean() if len(df) > 0 else 0.5
        })


class WikipediaViewsFetcher:
    """Fetches Wikipedia page view statistics."""
    
    def __init__(self):
        self.base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
    
    def fetch(self, article_title: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Wikipedia page views."""
        print(f"[DATA] Fetching Wikipedia views for '{article_title}'...")
        
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
            end = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
            
            url = f"{self.base_url}/en.wikipedia/all-access/all-agents/{article_title}/daily/{start}/{end}"
            headers = {'User-Agent': 'ProjectKassandra/1.0'}
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                records = []
                for item in data.get('items', []):
                    date = datetime.strptime(item['timestamp'], '%Y%m%d00')
                    records.append({'Date': date, 'Wiki_Views': item['views']})
                
                df = pd.DataFrame(records)
                
                if not df.empty:
                    df['Wiki_Views_MA_7'] = df['Wiki_Views'].rolling(window=7, min_periods=1).mean()
                    df['Wiki_Views_Change'] = df['Wiki_Views'].pct_change()
                    print(f"[OK] Fetched {len(df)} days of Wikipedia data")
                    return df
                    
        except Exception as e:
            print(f"[WARN] Wikipedia: {e}")
        
        return pd.DataFrame()


class UniversalDataFetcher:
    """Main orchestrator for all data sources."""
    
    COMPANY_INFO = {
        'TSLA': {'name': 'Tesla', 'wiki': 'Tesla,_Inc.'},
        'AAPL': {'name': 'Apple', 'wiki': 'Apple_Inc.'},
        'GOOGL': {'name': 'Google', 'wiki': 'Google'},
        'MSFT': {'name': 'Microsoft', 'wiki': 'Microsoft'},
        'AMZN': {'name': 'Amazon', 'wiki': 'Amazon_(company)'},
        'META': {'name': 'Meta', 'wiki': 'Meta_Platforms'},
        'NVDA': {'name': 'NVIDIA', 'wiki': 'Nvidia'},
        'AMD': {'name': 'AMD', 'wiki': 'AMD'},
        'NFLX': {'name': 'Netflix', 'wiki': 'Netflix'},
        'DIS': {'name': 'Disney', 'wiki': 'The_Walt_Disney_Company'},
        'NKE': {'name': 'Nike', 'wiki': 'Nike,_Inc.'},
        'SBUX': {'name': 'Starbucks', 'wiki': 'Starbucks'},
        '^NSEI': {'name': 'Nifty 50', 'wiki': 'NIFTY_50'},
        '^NSEBANK': {'name': 'Nifty Bank', 'wiki': 'Nifty_Bank'},
    }
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.trends_fetcher = GoogleTrendsFetcher()
        self.news_fetcher = NewsSentimentFetcher()
        self.reddit_fetcher = RedditSentimentFetcher()
        self.wiki_fetcher = WikipediaViewsFetcher()
    
    def get_company_info(self, ticker: str) -> dict:
        """Get company info for a ticker."""
        ticker = ticker.upper()
        if ticker in self.COMPANY_INFO:
            return self.COMPANY_INFO[ticker]
        return {'name': ticker, 'wiki': ticker}
    
    def fetch_all(self, ticker: str, start_date: str, end_date: str,
                  include_trends: bool = True,
                  include_news: bool = True,
                  include_reddit: bool = True,
                  include_wiki: bool = True) -> pd.DataFrame:
        """Fetch all data sources and merge."""
        print("\n" + "="*60)
        print(f"UNIVERSAL DATA FETCHER - {ticker}")
        print(f"Period: {start_date} to {end_date}")
        print("="*60 + "\n")
        
        company_info = self.get_company_info(ticker)
        company_name = company_info['name']
        wiki_title = company_info['wiki']
        
        # Fetch stock data (REQUIRED)
        stock_df = self.stock_fetcher.fetch(ticker, start_date, end_date)
        merged_df = stock_df.copy()
        
        # Optional data sources
        if include_trends:
            try:
                trends_df = self.trends_fetcher.fetch(company_name, start_date, end_date)
                if not trends_df.empty:
                    merged_df = self._merge_on_date(merged_df, trends_df)
            except Exception as e:
                print(f"[WARN] Skipping Trends: {e}")
        
        if include_news:
            try:
                news_df = self.news_fetcher.fetch(ticker, company_name, start_date, end_date)
                if not news_df.empty:
                    merged_df = self._merge_on_date(merged_df, news_df)
            except Exception as e:
                print(f"[WARN] Skipping News: {e}")
        
        if include_reddit:
            try:
                reddit_df = self.reddit_fetcher.fetch(ticker, start_date, end_date)
                if not reddit_df.empty:
                    merged_df = self._merge_on_date(merged_df, reddit_df)
            except Exception as e:
                print(f"[WARN] Skipping Reddit: {e}")
        
        if include_wiki:
            try:
                wiki_df = self.wiki_fetcher.fetch(wiki_title, start_date, end_date)
                if not wiki_df.empty:
                    merged_df = self._merge_on_date(merged_df, wiki_df)
            except Exception as e:
                print(f"[WARN] Skipping Wikipedia: {e}")
        
        merged_df['Ticker'] = ticker
        
        print("\n" + "="*60)
        print("DATA FETCHING COMPLETE")
        print(f"Total features: {len(merged_df.columns)}")
        print(f"Total trading days: {len(merged_df)}")
        print("="*60 + "\n")
        
        return merged_df
    
    def _merge_on_date(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """Merge DataFrames on Date."""
        df1['Date'] = pd.to_datetime(df1['Date'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        return pd.merge(df1, df2, on='Date', how='left')


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Quick function to fetch all data."""
    fetcher = UniversalDataFetcher()
    return fetcher.fetch_all(ticker, start_date, end_date)


if __name__ == "__main__":
    df = fetch_stock_data('AAPL', '2024-01-01', '2025-01-01')
    print(df.head())
