"""
Mock data generator for demonstration purposes
Generates realistic stock data when external APIs are not available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """
    Generate realistic mock stock data for demonstration
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1mo, 3mo, 6mo, 1y)
    
    Returns:
        DataFrame with OHLCV data
    """
    # Determine number of days based on period
    period_days = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365
    }
    
    num_days = period_days.get(period, 90)
    
    # Generate dates (trading days only)
    end_date = datetime.now()
    dates = []
    current_date = end_date - timedelta(days=num_days)
    
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Base price depends on ticker (for realism)
    base_prices = {
        'AAPL': 175.0,
        'GOOGL': 140.0,
        'MSFT': 380.0,
        'AMZN': 150.0,
        'TSLA': 250.0,
        'META': 350.0,
        'JPM': 150.0,
        'BAC': 32.0,
        'GS': 380.0,
        'V': 250.0,
        'JNJ': 160.0,
        'UNH': 520.0,
        'PFE': 30.0,
        'ABBV': 165.0,
        'NKE': 110.0,
        'DIS': 95.0
    }
    
    base_price = base_prices.get(ticker, 100.0)
    
    # Generate price series with realistic properties
    np.random.seed(hash(ticker) % 2**32)  # Consistent data for same ticker
    
    # Trend component (slight upward bias for most stocks)
    trend = np.linspace(0, 0.15, len(dates))
    
    # Random walk component
    returns = np.random.normal(0.0005, 0.02, len(dates))
    price_multiplier = np.exp(np.cumsum(returns) + trend)
    
    close_prices = base_price * price_multiplier
    
    # Generate OHLC data
    data = []
    for i, date in enumerate(dates):
        close = close_prices[i]
        
        # Daily volatility
        daily_volatility = np.random.uniform(0.005, 0.025)
        
        open_price = close * (1 + np.random.normal(0, daily_volatility))
        high_price = max(open_price, close) * (1 + abs(np.random.normal(0, daily_volatility)))
        low_price = min(open_price, close) * (1 - abs(np.random.normal(0, daily_volatility)))
        
        # Volume (higher volume on volatile days)
        base_volume = 50_000_000
        volume_multiplier = 1 + abs(close - open_price) / close
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    
    return df


def get_mock_stock_info(ticker: str) -> dict:
    """Generate mock stock information"""
    stock_info = {
        'AAPL': {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 2_800_000_000_000,
            'peRatio': 29.5,
            'dividendYield': 0.005,
            '52WeekHigh': 199.62,
            '52WeekLow': 164.08
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'name': 'Alphabet Inc.',
            'sector': 'Technology',
            'industry': 'Internet Content & Information',
            'marketCap': 1_750_000_000_000,
            'peRatio': 26.8,
            'dividendYield': 0.0,
            '52WeekHigh': 153.78,
            '52WeekLow': 121.46
        },
        'MSFT': {
            'symbol': 'MSFT',
            'name': 'Microsoft Corporation',
            'sector': 'Technology',
            'industry': 'Software',
            'marketCap': 2_900_000_000_000,
            'peRatio': 35.2,
            'dividendYield': 0.008,
            '52WeekHigh': 420.82,
            '52WeekLow': 362.90
        },
        'TSLA': {
            'symbol': 'TSLA',
            'name': 'Tesla Inc.',
            'sector': 'Consumer Cyclical',
            'industry': 'Auto Manufacturers',
            'marketCap': 790_000_000_000,
            'peRatio': 78.5,
            'dividendYield': 0.0,
            '52WeekHigh': 299.29,
            '52WeekLow': 138.80
        }
    }
    
    # Default info for unknown tickers
    default_info = {
        'symbol': ticker,
        'name': f'{ticker} Corporation',
        'sector': 'Unknown',
        'industry': 'Unknown',
        'marketCap': 100_000_000_000,
        'peRatio': 20.0,
        'dividendYield': 0.02,
        '52WeekHigh': 150.0,
        '52WeekLow': 80.0
    }
    
    return stock_info.get(ticker, default_info)
