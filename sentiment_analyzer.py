"""
Sentiment Analysis Module
Analyzes market sentiment from multiple sources
"""

from textblob import TextBlob
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta


class SentimentAnalyzer:
    """Analyzes sentiment for stocks using multiple signals"""
    
    def __init__(self):
        self.sentiment_history = {}
        
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment from text (news, social media)
        Returns sentiment score [0, 1] where 0=negative, 0.5=neutral, 1=positive
        """
        if not text:
            return 0.5
            
        blob = TextBlob(text)
        # Normalize polarity from [-1, 1] to [0, 1]
        normalized_sentiment = (blob.sentiment.polarity + 1) / 2
        return normalized_sentiment
        
    def analyze_price_momentum(self, prices: List[float]) -> float:
        """
        Analyze sentiment from price momentum
        Returns sentiment score [0, 1]
        """
        if len(prices) < 2:
            return 0.5
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Use moving average momentum
        if len(returns) >= 5:
            short_ma = np.mean(returns[-5:])
            long_ma = np.mean(returns)
            momentum = (short_ma - long_ma) * 100
        else:
            momentum = np.mean(returns) * 100
            
        # Normalize to [0, 1]
        sentiment = 1 / (1 + np.exp(-momentum))
        return sentiment
        
    def analyze_volume_trend(self, volumes: List[float]) -> float:
        """
        Analyze sentiment from volume trends
        Returns sentiment score [0, 1]
        """
        if len(volumes) < 2:
            return 0.5
            
        recent_avg = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        historical_avg = np.mean(volumes)
        
        if historical_avg == 0:
            return 0.5
            
        volume_ratio = recent_avg / historical_avg
        
        # High volume can indicate strong sentiment (positive or negative)
        # We'll consider increasing volume as positive
        sentiment = min(volume_ratio, 2.0) / 2.0
        return sentiment
        
    def calculate_composite_sentiment(self, 
                                     news_texts: List[str] = None,
                                     prices: List[float] = None,
                                     volumes: List[float] = None) -> Dict:
        """
        Calculate composite sentiment from multiple sources
        Returns detailed sentiment breakdown
        """
        sentiments = {}
        weights = {}
        
        # Text sentiment
        if news_texts:
            text_sentiments = [self.analyze_text(text) for text in news_texts]
            sentiments['text'] = np.mean(text_sentiments) if text_sentiments else 0.5
            weights['text'] = 0.3
        else:
            sentiments['text'] = 0.5
            weights['text'] = 0.0
            
        # Price momentum sentiment
        if prices and len(prices) > 1:
            sentiments['momentum'] = self.analyze_price_momentum(prices)
            weights['momentum'] = 0.4
        else:
            sentiments['momentum'] = 0.5
            weights['momentum'] = 0.0
            
        # Volume sentiment
        if volumes and len(volumes) > 1:
            sentiments['volume'] = self.analyze_volume_trend(volumes)
            weights['volume'] = 0.3
        else:
            sentiments['volume'] = 0.5
            weights['volume'] = 0.0
            
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Default equal weights if no data
            return {
                'composite': 0.5,
                'components': sentiments,
                'confidence': 0.0
            }
            
        # Calculate weighted composite
        composite = sum(sentiments[k] * weights[k] for k in sentiments)
        
        # Calculate confidence based on data availability
        confidence = min(total_weight / 1.0, 1.0)
        
        return {
            'composite': composite,
            'components': sentiments,
            'weights': weights,
            'confidence': confidence
        }
        
    def get_sentiment_signal(self, composite_sentiment: float) -> str:
        """Convert sentiment score to trading signal"""
        if composite_sentiment >= 0.65:
            return "BULLISH"
        elif composite_sentiment >= 0.45:
            return "NEUTRAL"
        else:
            return "BEARISH"


def generate_mock_news(ticker: str, sentiment_bias: float = 0.5) -> List[str]:
    """
    Generate mock news headlines for demonstration
    In production, this would fetch real news from APIs
    """
    positive_templates = [
        f"{ticker} reports strong quarterly earnings, beating expectations",
        f"Analysts upgrade {ticker} price target on robust fundamentals",
        f"{ticker} announces innovative product launch, stock surges",
        f"Major institutional investors increase {ticker} holdings",
        f"{ticker} expands into new markets, growth prospects strong"
    ]
    
    negative_templates = [
        f"{ticker} faces regulatory scrutiny, shares decline",
        f"Analysts downgrade {ticker} on weak guidance",
        f"{ticker} misses revenue estimates, concerns grow",
        f"Competition intensifies for {ticker}, market share at risk",
        f"{ticker} announces layoffs amid restructuring efforts"
    ]
    
    neutral_templates = [
        f"{ticker} trading in line with market averages",
        f"Investors await {ticker} earnings announcement",
        f"{ticker} maintains steady performance amid volatility",
        f"Mixed analyst opinions on {ticker} outlook",
        f"{ticker} holds annual shareholder meeting"
    ]
    
    # Select news based on sentiment bias
    if sentiment_bias > 0.6:
        return np.random.choice(positive_templates, size=3, replace=False).tolist()
    elif sentiment_bias < 0.4:
        return np.random.choice(negative_templates, size=3, replace=False).tolist()
    else:
        # Mix of different sentiments
        news = []
        news.append(np.random.choice(positive_templates))
        news.append(np.random.choice(neutral_templates))
        news.append(np.random.choice(negative_templates))
        return news
