"""
Sentiment Divergence Analysis Module
Detects and analyzes divergence between news sentiment and social sentiment
"""

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class SentimentDivergenceAnalyzer:
    """
    Analyzes divergence between professional news sentiment and retail social sentiment
    Implements the principle: News vs Social divergence creates trading opportunities
    """
    
    def __init__(self):
        self.divergence_threshold = 0.15  # 15% difference triggers divergence signal
        
    def calculate_divergence(self, news_sentiment: float, social_sentiment: float) -> Dict:
        """
        Calculate divergence between news and social sentiment
        
        Args:
            news_sentiment: Professional news sentiment score [0, 1]
            social_sentiment: Social media sentiment score [0, 1]
            
        Returns:
            Dictionary with divergence analysis
        """
        divergence = news_sentiment - social_sentiment
        divergence_magnitude = abs(divergence)
        
        # Determine divergence type
        if divergence_magnitude < self.divergence_threshold:
            divergence_type = "ALIGNED"
            signal = "STRONG"  # Both aligned = strong signal
            expected_direction = self._get_aligned_direction(news_sentiment, social_sentiment)
        elif divergence > self.divergence_threshold:
            divergence_type = "NEWS_BULLISH_SOCIAL_BEARISH"
            signal = "BUY"  # News positive, social negative = price likely to rise
            expected_direction = "UP"
        else:
            divergence_type = "NEWS_BEARISH_SOCIAL_BULLISH"
            signal = "SELL"  # News negative, social positive = momentum fade likely
            expected_direction = "DOWN"
        
        # Calculate confidence
        confidence = self._calculate_divergence_confidence(
            news_sentiment, social_sentiment, divergence_magnitude
        )
        
        return {
            'divergence': divergence,
            'divergence_magnitude': divergence_magnitude,
            'divergence_type': divergence_type,
            'signal': signal,
            'expected_direction': expected_direction,
            'confidence': confidence,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'is_divergent': divergence_magnitude >= self.divergence_threshold
        }
    
    def _get_aligned_direction(self, news: float, social: float) -> str:
        """Determine direction when sentiments are aligned"""
        avg_sentiment = (news + social) / 2
        if avg_sentiment > 0.6:
            return "UP"
        elif avg_sentiment < 0.4:
            return "DOWN"
        else:
            return "NEUTRAL"
    
    def _calculate_divergence_confidence(self, news: float, social: float, magnitude: float) -> float:
        """
        Calculate confidence in divergence signal
        Higher confidence when:
        - Divergence magnitude is large
        - One sentiment is extreme while other is moderate
        """
        # Base confidence from magnitude
        magnitude_confidence = min(magnitude / 0.5, 1.0)
        
        # Boost confidence if sentiments are extreme
        extremeness = max(abs(news - 0.5), abs(social - 0.5)) * 2
        
        # Combined confidence
        confidence = 0.6 * magnitude_confidence + 0.4 * extremeness
        
        return min(confidence, 1.0)
    
    def get_trading_recommendation(self, divergence_analysis: Dict, 
                                   current_price: float,
                                   technical_signal: str) -> Dict:
        """
        Generate trading recommendation based on sentiment divergence
        
        Args:
            divergence_analysis: Result from calculate_divergence
            current_price: Current stock price
            technical_signal: Technical analysis signal ('BUY', 'SELL', 'HOLD')
            
        Returns:
            Trading recommendation with position sizing
        """
        signal = divergence_analysis['signal']
        confidence = divergence_analysis['confidence']
        
        # Combine with technical signal
        if signal == "STRONG" and divergence_analysis['expected_direction'] == "UP":
            action = "BUY"
            position_size = 1.0  # Full position
        elif signal == "STRONG" and divergence_analysis['expected_direction'] == "DOWN":
            action = "SELL"
            position_size = 1.0
        elif signal == "BUY":
            action = "BUY"
            position_size = 0.75  # 75% position due to divergence
        elif signal == "SELL":
            action = "SELL"
            position_size = 0.75
        else:
            action = "HOLD"
            position_size = 0.5
        
        # Adjust based on technical confirmation
        if technical_signal == action:
            confidence *= 1.2  # 20% boost for confirmation
            position_size = min(position_size * 1.1, 1.0)
        elif technical_signal == "HOLD":
            confidence *= 0.9  # Slight reduction
        else:
            confidence *= 0.7  # Conflicting signals
            position_size *= 0.7
        
        confidence = min(confidence, 1.0)
        
        return {
            'action': action,
            'position_size': position_size,
            'confidence': confidence,
            'reasoning': self._generate_reasoning(divergence_analysis, technical_signal),
            'expected_move': divergence_analysis['expected_direction'],
            'risk_level': self._calculate_risk_level(divergence_analysis, confidence)
        }
    
    def _generate_reasoning(self, divergence: Dict, technical: str) -> str:
        """Generate human-readable reasoning for the recommendation"""
        if divergence['divergence_type'] == "ALIGNED":
            direction = divergence['expected_direction']
            sentiment_level = "bullish" if divergence['news_sentiment'] > 0.6 else \
                            "bearish" if divergence['news_sentiment'] < 0.4 else "neutral"
            return f"News and social sentiment are aligned ({sentiment_level}), suggesting {direction} movement. " \
                   f"Technical signal: {technical}."
        
        elif divergence['divergence_type'] == "NEWS_BULLISH_SOCIAL_BEARISH":
            return f"News sentiment is positive ({divergence['news_sentiment']:.2f}) while social sentiment is " \
                   f"negative ({divergence['social_sentiment']:.2f}). Price likely to rise as retail catches up. " \
                   f"Technical signal: {technical}."
        
        else:
            return f"Social sentiment is high ({divergence['social_sentiment']:.2f}) but news sentiment is " \
                   f"weak ({divergence['news_sentiment']:.2f}). Momentum fade likely. " \
                   f"Technical signal: {technical}."
    
    def _calculate_risk_level(self, divergence: Dict, confidence: float) -> str:
        """Calculate risk level for the trade"""
        if confidence > 0.75 and not divergence['is_divergent']:
            return "LOW"
        elif confidence > 0.6:
            return "MODERATE"
        else:
            return "HIGH"
    
    def analyze_sentiment_trend(self, historical_divergence: List[Dict]) -> Dict:
        """
        Analyze trend in sentiment divergence over time
        
        Args:
            historical_divergence: List of divergence analyses over time
            
        Returns:
            Trend analysis
        """
        if len(historical_divergence) < 2:
            return {
                'trend': 'INSUFFICIENT_DATA',
                'trend_strength': 0.0,
                'recommendation': 'HOLD'
            }
        
        # Extract divergence values
        divergences = [d['divergence'] for d in historical_divergence]
        
        # Calculate trend
        if len(divergences) >= 3:
            # Linear regression slope with error handling
            try:
                x = np.arange(len(divergences))
                y = np.array(divergences)
                slope = np.polyfit(x, y, 1)[0]
            except (np.linalg.LinAlgError, ValueError) as e:
                # Fallback to simple comparison if polyfit fails
                slope = (divergences[-1] - divergences[0]) / len(divergences)
            
            if slope > 0.05:
                trend = "INCREASING_DIVERGENCE"
                recommendation = "WAIT"  # Divergence growing, wait for reversal
            elif slope < -0.05:
                trend = "DECREASING_DIVERGENCE"
                recommendation = "ENTER"  # Divergence narrowing, good entry
            else:
                trend = "STABLE"
                recommendation = "MONITOR"
            
            trend_strength = min(abs(slope) * 10, 1.0)
        else:
            # Simple comparison
            if divergences[-1] > divergences[0]:
                trend = "INCREASING"
                recommendation = "WAIT"
            else:
                trend = "DECREASING"
                recommendation = "ENTER"
            
            trend_strength = abs(divergences[-1] - divergences[0])
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'recommendation': recommendation,
            'current_divergence': divergences[-1],
            'average_divergence': np.mean(divergences)
        }


class RetailLeadingIndicator:
    """
    Analyzes retail sentiment as a leading indicator
    Principle: Retail sentiment leads fundamental news by 3-5 days
    """
    
    def __init__(self):
        self.lead_time_days = 4  # Average lead time
        self.signal_threshold = 0.2  # 20% change triggers signal
        
    def detect_leading_signal(self, social_sentiment: float, 
                              social_trend: str,
                              news_sentiment: float) -> Dict:
        """
        Detect if retail sentiment is leading the market
        
        Args:
            social_sentiment: Current social sentiment
            social_trend: Trend in social sentiment ('INCREASING', 'DECREASING', 'STABLE')
            news_sentiment: Current news sentiment
            
        Returns:
            Leading indicator analysis
        """
        # Calculate sentiment gap
        sentiment_gap = social_sentiment - news_sentiment
        
        # Determine if retail is leading
        is_leading = abs(sentiment_gap) > self.signal_threshold
        
        if not is_leading:
            return {
                'is_leading': False,
                'signal': 'WAIT',
                'expected_news_shift': 'NONE',
                'confidence': 0.0,
                'days_to_convergence': 0
            }
        
        # Predict news sentiment shift
        if sentiment_gap > self.signal_threshold:
            expected_shift = "BULLISH"
            signal = "BUY_EARLY"
            action = "News sentiment likely to improve in 3-5 days"
        else:
            expected_shift = "BEARISH"
            signal = "SELL_EARLY"
            action = "News sentiment likely to deteriorate in 3-5 days"
        
        # Calculate confidence based on trend strength
        if social_trend == "INCREASING" and sentiment_gap > 0:
            confidence = 0.8
        elif social_trend == "DECREASING" and sentiment_gap < 0:
            confidence = 0.8
        else:
            confidence = 0.6
        
        return {
            'is_leading': True,
            'signal': signal,
            'expected_news_shift': expected_shift,
            'confidence': confidence,
            'days_to_convergence': self.lead_time_days,
            'sentiment_gap': sentiment_gap,
            'action': action
        }


class MultiSourceSentimentAggregator:
    """
    Aggregates sentiment from multiple sources with intelligent weighting
    """
    
    def __init__(self):
        self.divergence_analyzer = SentimentDivergenceAnalyzer()
        self.leading_indicator = RetailLeadingIndicator()
        
    def aggregate_sentiments(self, 
                            news_sentiment: float,
                            social_sentiment: float,
                            analyst_sentiment: float,
                            technical_sentiment: float) -> Dict:
        """
        Aggregate all sentiment sources with dynamic weighting
        
        Args:
            news_sentiment: Professional news sentiment [0, 1]
            social_sentiment: Social media sentiment [0, 1]
            analyst_sentiment: Analyst ratings/targets sentiment [0, 1]
            technical_sentiment: Technical analysis sentiment [0, 1]
            
        Returns:
            Aggregated sentiment analysis
        """
        # Base weights
        weights = {
            'news': 0.25,
            'social': 0.20,
            'analyst': 0.30,
            'technical': 0.25
        }
        
        # Calculate divergence
        divergence = self.divergence_analyzer.calculate_divergence(
            news_sentiment, social_sentiment
        )
        
        # Adjust weights based on divergence
        if divergence['is_divergent']:
            # Increase weight of the sentiment that's more extreme
            if abs(news_sentiment - 0.5) > abs(social_sentiment - 0.5):
                weights['news'] += 0.10
                weights['social'] -= 0.05
                weights['technical'] -= 0.05
            else:
                weights['social'] += 0.10
                weights['news'] -= 0.05
                weights['technical'] -= 0.05
        
        # Calculate weighted composite
        composite = (
            weights['news'] * news_sentiment +
            weights['social'] * social_sentiment +
            weights['analyst'] * analyst_sentiment +
            weights['technical'] * technical_sentiment
        )
        
        # Calculate confidence
        # Higher confidence when all sources agree
        sentiments = [news_sentiment, social_sentiment, analyst_sentiment, technical_sentiment]
        variance = np.var(sentiments)
        agreement = 1.0 - min(variance * 4, 1.0)
        
        return {
            'composite_sentiment': composite,
            'agreement_score': agreement,
            'confidence': agreement,
            'divergence_analysis': divergence,
            'weights': weights,
            'individual_sentiments': {
                'news': news_sentiment,
                'social': social_sentiment,
                'analyst': analyst_sentiment,
                'technical': technical_sentiment
            }
        }
