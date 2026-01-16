/**
 * Project Kassandra - Dashboard Application
 * Professional stock prediction dashboard with real-time updates
 */

class KassandraDashboard {
    constructor() {
        this.priceChart = null;
        this.featureChart = null;
        this.priceHistory = [];
        this.isLoading = false;
        
        this.init();
    }
    
    init() {
        this.setupDateDefaults();
        this.setupEventListeners();
        this.initCharts();
        this.checkStatus();
    }
    
    setupDateDefaults() {
        // Competition timeline: 5 years of data (2021-01-15 to 2026-01-15)
        const startDate = new Date('2021-01-15');
        const endDate = new Date('2026-01-15');
        
        document.getElementById('startDate').value = this.formatDate(startDate);
        document.getElementById('endDate').value = this.formatDate(endDate);
    }
    
    formatDate(date) {
        return date.toISOString().split('T')[0];
    }
    
    setupEventListeners() {
        // Run pipeline button
        document.getElementById('runPipeline').addEventListener('click', () => {
            this.runPipeline();
        });
        
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleNavigation(item.dataset.section);
            });
        });
        
        // Chart range buttons
        document.querySelectorAll('.chart-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.updateChartRange(parseInt(btn.dataset.range));
            });
        });
        
        // Download CSV
        document.getElementById('downloadCSV').addEventListener('click', () => {
            window.location.href = '/api/download/prediction_log_' + document.getElementById('tickerInput').value.toUpperCase() + '.csv';
        });
        
        // Enter key on ticker input
        document.getElementById('tickerInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.runPipeline();
            }
        });
    }
    
    handleNavigation(section) {
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.section === section);
        });
        
        document.getElementById('dashboardSection').style.display = 
            section === 'dashboard' ? 'block' : 'none';
        document.getElementById('predictionsSection').style.display = 
            section === 'predictions' ? 'block' : 'none';
            
        if (section === 'predictions') {
            this.loadPredictionsLog();
        }
    }
    
    initCharts() {
        // Price Chart
        const priceCtx = document.getElementById('priceChart').getContext('2d');
        this.priceChart = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Close Price',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: '#1a2332',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        borderColor: '#2a3548',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return '$' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(42, 53, 72, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(42, 53, 72, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b',
                            callback: function(value) {
                                return '$' + value;
                            }
                        }
                    }
                }
            }
        });
        
        // Feature Importance Chart
        const featureCtx = document.getElementById('featureChart').getContext('2d');
        this.featureChart = new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Importance',
                    data: [],
                    backgroundColor: [
                        '#3b82f6', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b',
                        '#ef4444', '#ec4899', '#6366f1', '#14b8a6', '#84cc16'
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(42, 53, 72, 0.5)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#64748b'
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: {
                                size: 11
                            }
                        }
                    }
                }
            }
        });
    }
    
    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            this.updateStatusIndicator(data.status);
            
            if (data.last_update) {
                this.updateTimestamp(data.last_update);
            }
        } catch (error) {
            console.error('Status check failed:', error);
        }
    }
    
    updateStatusIndicator(status) {
        const indicator = document.getElementById('systemStatus');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');
        
        dot.className = 'status-dot ' + status;
        
        const statusTexts = {
            'idle': 'System Idle',
            'running': 'Processing...',
            'ready': 'Model Ready'
        };
        
        text.textContent = statusTexts[status] || 'Unknown';
    }
    
    async runPipeline() {
        if (this.isLoading) return;
        
        const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;
        
        if (!ticker) {
            this.showToast('Please enter a stock ticker', 'error');
            return;
        }
        
        this.setLoading(true);
        this.updateStatusIndicator('running');
        
        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ticker: ticker,
                    start_date: startDate,
                    end_date: endDate
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.updateDashboard(data);
                await this.loadPriceHistory();
                this.updateStatusIndicator('ready');
                this.showToast('Analysis complete for ' + ticker, 'success');
            } else {
                throw new Error(data.error || 'Pipeline failed');
            }
        } catch (error) {
            this.showToast('Error: ' + error.message, 'error');
            this.updateStatusIndicator('idle');
        } finally {
            this.setLoading(false);
        }
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        const btn = document.getElementById('runPipeline');
        const btnText = btn.querySelector('.btn-text');
        const btnLoader = btn.querySelector('.btn-loader');
        
        btn.disabled = loading;
        btnText.style.display = loading ? 'none' : 'inline';
        btnLoader.style.display = loading ? 'inline-flex' : 'none';
    }
    
    updateDashboard(data) {
        // Prediction Hero
        document.getElementById('predictedPrice').textContent = '$' + data.prediction.price.toFixed(2);
        document.getElementById('lastCloseValue').textContent = '$' + data.prediction.last_close.toFixed(2);
        document.getElementById('confidenceValue').textContent = data.prediction.confidence.toFixed(1) + '%';
        
        // Change
        const changeEl = document.getElementById('predictedChange');
        const changeValue = data.prediction.change;
        const changePct = data.prediction.change_pct;
        changeEl.innerHTML = `
            <span class="change-value">${changeValue >= 0 ? '+' : ''}$${changeValue.toFixed(2)}</span>
            <span class="change-percent">(${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%)</span>
        `;
        changeEl.className = 'prediction-change ' + (changeValue >= 0 ? 'positive' : 'negative');
        
        // Signal
        const signalBadge = document.getElementById('signalBadge');
        const signal = data.prediction.signal.toLowerCase();
        signalBadge.className = 'prediction-signal ' + signal;
        signalBadge.querySelector('.signal-text').textContent = data.prediction.signal;
        
        // Metrics
        document.getElementById('maeValue').textContent = '$' + data.metrics.mae.toFixed(2);
        document.getElementById('rmseValue').textContent = '$' + data.metrics.rmse.toFixed(2);
        document.getElementById('mapeValue').textContent = data.metrics.mape.toFixed(2);
        document.getElementById('r2Value').textContent = data.metrics.r2.toFixed(4);
        
        // Sentiment
        this.updateSentimentBar('news', data.sentiment.news);
        this.updateSentimentBar('reddit', data.sentiment.reddit);
        
        const trendsValue = data.sentiment.google_trends || 0;
        document.getElementById('trendsValue').textContent = trendsValue.toFixed(0);
        document.getElementById('trendsSentimentBar').style.width = trendsValue + '%';
        
        // Market Data
        document.getElementById('marketOpen').textContent = '$' + data.market_data.open.toFixed(2);
        document.getElementById('marketHigh').textContent = '$' + data.market_data.high.toFixed(2);
        document.getElementById('marketLow').textContent = '$' + data.market_data.low.toFixed(2);
        document.getElementById('marketClose').textContent = '$' + data.market_data.close.toFixed(2);
        document.getElementById('marketVolume').textContent = this.formatVolume(data.market_data.volume);
        
        // Model Predictions
        const modelGrid = document.getElementById('modelPredictions');
        const models = data.prediction.model_predictions;
        modelGrid.innerHTML = Object.entries(models).map(([name, value]) => `
            <div class="model-item">
                <span class="model-name">${this.formatModelName(name)}</span>
                <span class="model-value">$${value.toFixed(2)}</span>
            </div>
        `).join('');
        
        // Feature Importance
        if (data.feature_importance && data.feature_importance.length > 0) {
            const labels = data.feature_importance.map(f => f.feature);
            const values = data.feature_importance.map(f => f.importance);
            
            this.featureChart.data.labels = labels;
            this.featureChart.data.datasets[0].data = values;
            this.featureChart.update();
        }
        
        // Timestamp
        this.updateTimestamp(data.last_update);
    }
    
    updateSentimentBar(type, value) {
        const bar = document.getElementById(type + 'SentimentBar');
        const valueEl = document.getElementById(type + 'SentimentValue');
        
        // Convert -1 to 1 range to 0-100%
        const percentage = ((value + 1) / 2) * 100;
        bar.style.width = percentage + '%';
        bar.className = 'sentiment-bar ' + (value > 0.1 ? 'positive' : value < -0.1 ? 'negative' : '');
        valueEl.textContent = value.toFixed(3);
    }
    
    formatVolume(volume) {
        if (volume >= 1000000000) {
            return (volume / 1000000000).toFixed(2) + 'B';
        } else if (volume >= 1000000) {
            return (volume / 1000000).toFixed(2) + 'M';
        } else if (volume >= 1000) {
            return (volume / 1000).toFixed(2) + 'K';
        }
        return volume.toString();
    }
    
    formatModelName(name) {
        const names = {
            'xgboost': 'XGBoost',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boost',
            'ridge': 'Ridge Regression'
        };
        return names[name] || name;
    }
    
    updateTimestamp(isoString) {
        const date = new Date(isoString);
        const formatted = date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        document.querySelector('.timestamp-value').textContent = formatted;
    }
    
    async loadPriceHistory() {
        try {
            const response = await fetch('/api/price-history');
            const data = await response.json();
            
            if (data.history) {
                this.priceHistory = data.history;
                this.updateChartRange(30);
            }
        } catch (error) {
            console.error('Failed to load price history:', error);
        }
    }
    
    updateChartRange(days) {
        const recentData = this.priceHistory.slice(-days);
        
        this.priceChart.data.labels = recentData.map(d => d.date);
        this.priceChart.data.datasets[0].data = recentData.map(d => d.close);
        this.priceChart.update();
    }
    
    async loadPredictionsLog() {
        try {
            const response = await fetch('/api/predictions-log');
            const data = await response.json();
            
            const tbody = document.querySelector('#predictionsTable tbody');
            
            if (data.predictions && data.predictions.length > 0) {
                tbody.innerHTML = data.predictions.map(p => `
                    <tr>
                        <td>${p.date}</td>
                        <td>$${p.actual.toFixed(2)}</td>
                        <td>$${p.predicted.toFixed(2)}</td>
                        <td>$${p.error.toFixed(2)}</td>
                        <td>${p.error_pct.toFixed(2)}%</td>
                    </tr>
                `).join('');
            } else {
                tbody.innerHTML = '<tr><td colspan="5" class="empty-state">Run analysis to view predictions</td></tr>';
            }
        } catch (error) {
            console.error('Failed to load predictions log:', error);
        }
    }
    
    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = 'toast ' + type;
        toast.innerHTML = `<div class="toast-message">${message}</div>`;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new KassandraDashboard();
});
