"""
Stock Data Module
Fetches and processes stock market data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from mock_data import generate_mock_stock_data, get_mock_stock_info


class StockDataFetcher:
    """Fetches and caches stock market data"""
    
    def __init__(self):
        self.cache = {}
        self.info_cache = {}
        
    def get_stock_data(self, ticker: str, period: str = "3mo") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        # Check cache first
        cache_key = (ticker, period)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                # Fallback to mock data
                print(f"Using mock data for {ticker}")
                df = generate_mock_stock_data(ticker, period)
                self.cache[cache_key] = df
                return df
                
            self.cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            # Fallback to mock data
            print(f"Using mock data for {ticker}")
            df = generate_mock_stock_data(ticker, period)
            self.cache[cache_key] = df
            return df

    def fetch_batch_data(self, tickers: List[str], period: str = "3mo"):
        """
        Fetch data for multiple tickers in one go to populate cache.

        Args:
            tickers: List of stock ticker symbols
            period: Time period
        """
        # Filter out tickers already in cache
        tickers_to_fetch = [t for t in tickers if (t, period) not in self.cache]

        if not tickers_to_fetch:
            return

        try:
            # Join tickers with space
            tickers_str = " ".join(tickers_to_fetch)
            data = yf.download(tickers_str, period=period, group_by='ticker', progress=False)

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    # MultiIndex columns: (Ticker, Price)
                    for ticker in data.columns.levels[0]:
                        # Extract data for this ticker
                        ticker_data = data[ticker]
                        # Check if data is valid (not all NaNs or empty)
                        if not ticker_data.empty:
                            self.cache[(ticker, period)] = ticker_data
                elif len(tickers_to_fetch) == 1:
                    # Single ticker result
                    self.cache[(tickers_to_fetch[0], period)] = data

        except Exception as e:
            print(f"Batch fetch failed: {e}")
            # Individual fetch will handle fallbacks later
            
    def get_current_price(self, ticker: str) -> Optional[float]:
        """Get current stock price"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None
            
    def get_stock_info(self, ticker: str) -> Dict:
        """Get stock information"""
        # Check cache first
        if ticker in self.info_cache:
            return self.info_cache[ticker]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if not info or 'symbol' not in info:
                # Fallback to mock data
                result = get_mock_stock_info(ticker)
                self.info_cache[ticker] = result
                return result
            
            result = {
                'symbol': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'marketCap': info.get('marketCap', 0),
                'peRatio': info.get('trailingPE', 0),
                'dividendYield': info.get('dividendYield', 0),
                '52WeekHigh': info.get('fiftyTwoWeekHigh', 0),
                '52WeekLow': info.get('fiftyTwoWeekLow', 0),
            }
            self.info_cache[ticker] = result
            return result
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            # Fallback to mock data
            result = get_mock_stock_info(ticker)
            self.info_cache[ticker] = result
            return result
            
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if df is None or df.empty:
            return {}
            
        indicators = {}
        
        # Simple Moving Averages
        if len(df) >= 20:
            indicators['SMA_20'] = df['Close'].rolling(window=20).mean().iloc[-1]
        if len(df) >= 50:
            indicators['SMA_50'] = df['Close'].rolling(window=50).mean().iloc[-1]
            
        # Relative Strength Index (RSI)
        if len(df) >= 14:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = (100 - (100 / (1 + rs))).iloc[-1]
            
        # Volatility
        if len(df) >= 20:
            indicators['Volatility'] = df['Close'].pct_change().rolling(window=20).std().iloc[-1]
            
        # Current metrics
        indicators['Current_Price'] = df['Close'].iloc[-1]
        indicators['Volume'] = df['Volume'].iloc[-1]
        
        # Price change
        if len(df) >= 2:
            indicators['Daily_Change'] = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
            
        return indicators


class BiweeklyInvestmentStrategy:
    """Implements biweekly investment strategy logic"""
    
    def __init__(self, investment_amount: float = 1000.0):
        self.investment_amount = investment_amount
        self.portfolio = {}
        self.transaction_history = []
        
    def calculate_biweekly_allocation(self, 
                                     agent_decisions: List,
                                     sentiment: float) -> Dict:
        """
        Calculate optimal biweekly investment allocation
        
        Args:
            agent_decisions: List of agent decisions
            sentiment: Market sentiment score
            
        Returns:
            Allocation strategy dictionary
        """
        # Count agent recommendations
        buy_votes = sum(1 for d in agent_decisions if d.action == "BUY")
        sell_votes = sum(1 for d in agent_decisions if d.action == "SELL")
        hold_votes = sum(1 for d in agent_decisions if d.action == "HOLD")
        
        total_votes = len(agent_decisions)
        buy_confidence = buy_votes / total_votes if total_votes > 0 else 0
        
        # Adjust allocation based on sentiment and agent consensus
        allocation_factor = buy_confidence * sentiment
        
        allocation = {
            'action': 'BUY' if buy_votes > max(sell_votes, hold_votes) else 
                     'SELL' if sell_votes > max(buy_votes, hold_votes) else 'HOLD',
            'amount': self.investment_amount * allocation_factor,
            'confidence': allocation_factor,
            'agent_consensus': {
                'buy': buy_votes,
                'sell': sell_votes,
                'hold': hold_votes
            }
        }
        
        return allocation
        
    def get_next_investment_date(self) -> datetime:
        """Calculate next biweekly investment date (every 14 days from today)"""
        today = datetime.now()
        # Simply add 14 days for true biweekly schedule
        next_date = today + timedelta(days=14)
        return next_date
        
    def days_until_next_investment(self) -> int:
        """Days remaining until next investment"""
        next_date = self.get_next_investment_date()
        today = datetime.now()
        delta = next_date - today
        return max(0, delta.days)
