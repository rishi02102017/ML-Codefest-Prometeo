"""
Project Kassandra - Web Application
Flask-based REST API server for the prediction dashboard.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_fetcher import UniversalDataFetcher
from src.feature_engineering import FeatureEngineer
from src.model import StockPricePredictor, calculate_metrics
from src.pipeline import KassandraPipeline


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global pipeline instance
pipeline_instance = None
last_run_time = None


@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get current system status."""
    global pipeline_instance, last_run_time
    
    return jsonify({
        'status': 'ready' if pipeline_instance and pipeline_instance.is_trained else 'idle',
        'last_update': last_run_time.isoformat() if last_run_time else None,
        'model_trained': pipeline_instance.is_trained if pipeline_instance else False
    })


@app.route('/api/run', methods=['POST'])
def run_pipeline():
    """Run the prediction pipeline."""
    global pipeline_instance, last_run_time
    
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'TSLA').upper()
        start_date = data.get('start_date', (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Initialize and run pipeline
        pipeline_instance = KassandraPipeline()
        results = pipeline_instance.run(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            export_features=True,
            run_validation=False
        )
        
        # Get prediction
        prediction = pipeline_instance.predict_next_day()
        last_run_time = datetime.now()
        
        # Get live data
        live_data = pipeline_instance.get_live_data()
        
        # Prepare response
        response = {
            'success': True,
            'ticker': ticker,
            'prediction': {
                'price': float(prediction['predicted_price']),
                'last_close': float(prediction['last_close']),
                'change': float(prediction['predicted_change']),
                'change_pct': float(prediction['predicted_change_pct']),
                'signal': prediction['signal'],
                'confidence': float(prediction['confidence']),
                'model_predictions': {k: float(v) for k, v in prediction['model_predictions'].items()}
            },
            'metrics': {
                'mae': float(results.get('test_metrics', {}).get('mae', 0)),
                'rmse': float(results.get('test_metrics', {}).get('rmse', 0)),
                'mape': float(results.get('test_metrics', {}).get('mape', 0)),
                'r2': float(results.get('test_metrics', {}).get('r2', 0))
            },
            'sentiment': {
                'news': float(live_data.get('news_sentiment', 0) or 0),
                'reddit': float(live_data.get('reddit_sentiment', 0) or 0),
                'google_trends': float(live_data.get('google_trends', 0) or 0),
                'wiki_views': int(live_data.get('wiki_views', 0) or 0)
            },
            'market_data': {
                'open': float(live_data.get('open', 0) or 0),
                'high': float(live_data.get('high', 0) or 0),
                'low': float(live_data.get('low', 0) or 0),
                'close': float(live_data.get('close', 0) or 0),
                'volume': int(live_data.get('volume', 0) or 0)
            },
            'feature_importance': convert_to_native(results.get('feature_importance', [])[:10]),
            'last_update': last_run_time.isoformat()
        }
        
        # Convert all numpy types to native Python types
        response = convert_to_native(response)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/price-history')
def get_price_history():
    """Get historical price data for charting."""
    global pipeline_instance
    
    if not pipeline_instance or pipeline_instance.raw_data is None:
        return jsonify({'error': 'No data available. Run the pipeline first.'}), 400
    
    df = pipeline_instance.raw_data.copy()
    
    # Convert to JSON-serializable format
    history = []
    for _, row in df.iterrows():
        history.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume'])
        })
    
    return jsonify({'history': history})


@app.route('/api/predictions-log')
def get_predictions_log():
    """Get prediction log data."""
    global pipeline_instance
    
    if not pipeline_instance or not pipeline_instance.logger.predictions:
        return jsonify({'predictions': []})
    
    df = pipeline_instance.logger.get_dataframe()
    
    predictions = []
    for _, row in df.iterrows():
        predictions.append({
            'date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date'])[:10],
            'actual': round(float(row['Actual_Close']), 2),
            'predicted': round(float(row['Predicted_Close']), 2),
            'error': round(float(row['Absolute_Error']), 2),
            'error_pct': round(float(row['Percentage_Error']), 2)
        })
    
    return jsonify({'predictions': predictions})


@app.route('/api/download/<filename>')
def download_file(filename):
    """Download generated files."""
    return send_from_directory('outputs', filename, as_attachment=True)


if __name__ == '__main__':
    # Ensure directories exist
    Path('outputs').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    Path('static/css').mkdir(parents=True, exist_ok=True)
    Path('static/js').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("PROJECT KASSANDRA - Web Server")
    print("="*60)
    print("Starting server at http://localhost:8080")
    print("Debug mode: ON (auto-reload enabled)")
    print("="*60 + "\n")
    
    # debug=True enables auto-reload when files change
    app.run(host='0.0.0.0', port=8080, debug=True)
