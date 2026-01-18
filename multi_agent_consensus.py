"""
Enhanced Multi-Agent Consensus Module
Implements improved consensus mechanisms with reliability scoring and signal quality
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


# Agent type constants for consistency
AGENT_TYPE_BULL = "Bull"
AGENT_TYPE_BEAR = "Bear"
AGENT_TYPE_TECHNICAL = "Technical"


class SignalReliability(Enum):
    """Signal reliability levels"""
    VERY_HIGH = "very_high"  # ~85%+ reliability
    HIGH = "high"  # ~70%+ reliability
    MODERATE = "moderate"  # ~55%+ reliability
    LOW = "low"  # <55% reliability


@dataclass
class EnhancedAgentDecision:
    """Enhanced agent decision with additional metadata"""
    agent_id: str
    agent_type: str  # 'Bull', 'Bear', 'Technical', 'Sentiment', 'Fundamental'
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    weight: float  # Importance weight for this agent


class MultiAgentConsensus:
    """
    Enhanced multi-agent consensus system with reliability scoring
    
    Principle: 3+ agents agree (Bull + Bear + Technical) = ~70% signal reliability
    More agents agree = higher reliability
    """
    
    def __init__(self):
        self.min_consensus_agents = 3
        self.high_consensus_threshold = 0.7  # 70% agreement
        self.very_high_consensus_threshold = 0.85  # 85% agreement
        
        # Agent type weights
        self.agent_weights = {
            'Bull': 1.0,
            'Bear': 1.0,
            'Technical': 1.2,  # Slightly higher weight
            'Sentiment': 0.9,
            'Fundamental': 1.1,
            'Contrarian': 0.8
        }
        
    def calculate_consensus(self, decisions: List[EnhancedAgentDecision]) -> Dict:
        """
        Calculate consensus from multiple agent decisions
        
        Args:
            decisions: List of agent decisions
            
        Returns:
            Consensus analysis with reliability scoring
        """
        if not decisions:
            return {
                'consensus_action': 'HOLD',
                'reliability': SignalReliability.LOW,
                'reliability_score': 0.0,
                'confidence': 0.0,
                'agent_agreement': 0.0,
                'signal_quality': 'POOR'
            }
        
        # Count votes by action
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        weighted_votes = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        
        for decision in decisions:
            weight = self.agent_weights.get(decision.agent_type, 1.0)
            votes[decision.action] += 1
            weighted_votes[decision.action] += weight * decision.confidence
        
        # Determine consensus action
        consensus_action = max(weighted_votes, key=weighted_votes.get)
        
        # Calculate agreement ratio
        consensus_count = votes[consensus_action]
        total_agents = len(decisions)
        agreement_ratio = consensus_count / total_agents
        
        # Calculate weighted confidence
        consensus_confidence = weighted_votes[consensus_action] / sum(weighted_votes.values())
        
        # Determine reliability
        reliability_info = self._calculate_reliability(
            consensus_count, total_agents, agreement_ratio, decisions, consensus_action
        )
        
        # Calculate signal quality
        signal_quality = self._assess_signal_quality(
            reliability_info['reliability_score'],
            agreement_ratio,
            consensus_confidence
        )
        
        return {
            'consensus_action': consensus_action,
            'reliability': reliability_info['reliability'],
            'reliability_score': reliability_info['reliability_score'],
            'confidence': consensus_confidence,
            'agent_agreement': agreement_ratio,
            'signal_quality': signal_quality,
            'votes': votes,
            'weighted_votes': weighted_votes,
            'total_agents': total_agents,
            'consensus_agents': consensus_count,
            'agent_types_in_consensus': reliability_info['agent_types']
        }
    
    def _calculate_reliability(self, consensus_count: int, 
                               total_agents: int,
                               agreement_ratio: float,
                               decisions: List[EnhancedAgentDecision],
                               consensus_action: str) -> Dict:
        """
        Calculate signal reliability based on multi-agent consensus
        
        Rules:
        - 3+ agents same direction = ~70% reliability
        - 4+ agents same direction = ~80% reliability
        - 5+ agents same direction = ~85% reliability
        - Must include diverse agent types for high reliability
        """
        # Get agent types that agree with consensus
        consensus_agent_types = [
            d.agent_type for d in decisions if d.action == consensus_action
        ]
        unique_types = len(set(consensus_agent_types))
        
        # Base reliability from consensus count
        if consensus_count >= 5:
            base_reliability = 0.85
            reliability_level = SignalReliability.VERY_HIGH
        elif consensus_count >= 4:
            base_reliability = 0.80
            reliability_level = SignalReliability.HIGH
        elif consensus_count >= 3:
            base_reliability = 0.70
            reliability_level = SignalReliability.HIGH
        elif consensus_count >= 2:
            base_reliability = 0.55
            reliability_level = SignalReliability.MODERATE
        else:
            base_reliability = 0.40
            reliability_level = SignalReliability.LOW
        
        # Boost reliability if diverse agent types agree
        if unique_types >= 3:
            diversity_boost = 0.10
        elif unique_types >= 2:
            diversity_boost = 0.05
        else:
            diversity_boost = 0.0
        
        # Check for Bull + Bear + Technical consensus (optimal)
        has_bull_bear_tech = all(
            t in consensus_agent_types 
            for t in [AGENT_TYPE_BULL, AGENT_TYPE_BEAR, AGENT_TYPE_TECHNICAL]
        )
        
        if has_bull_bear_tech:
            diversity_boost += 0.05
        
        # Final reliability score
        reliability_score = min(base_reliability + diversity_boost, 1.0)
        
        # Update reliability level based on final score
        if reliability_score >= 0.85:
            reliability_level = SignalReliability.VERY_HIGH
        elif reliability_score >= 0.70:
            reliability_level = SignalReliability.HIGH
        elif reliability_score >= 0.55:
            reliability_level = SignalReliability.MODERATE
        else:
            reliability_level = SignalReliability.LOW
        
        return {
            'reliability': reliability_level,
            'reliability_score': reliability_score,
            'agent_types': consensus_agent_types,
            'unique_types': unique_types,
            'has_bull_bear_tech': has_bull_bear_tech
        }
    
    def _assess_signal_quality(self, reliability_score: float,
                               agreement_ratio: float,
                               confidence: float) -> str:
        """
        Assess overall signal quality
        
        Returns: 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
        """
        # Composite quality score
        quality_score = (
            0.4 * reliability_score +
            0.3 * agreement_ratio +
            0.3 * confidence
        )
        
        if quality_score >= 0.80:
            return 'EXCELLENT'
        elif quality_score >= 0.65:
            return 'GOOD'
        elif quality_score >= 0.50:
            return 'FAIR'
        else:
            return 'POOR'
    
    def generate_position_sizing(self, consensus: Dict) -> Dict:
        """
        Generate position sizing recommendation based on consensus quality
        
        Args:
            consensus: Result from calculate_consensus
            
        Returns:
            Position sizing recommendation
        """
        action = consensus['consensus_action']
        reliability_score = consensus['reliability_score']
        signal_quality = consensus['signal_quality']
        
        # Base position size by action
        if action == 'BUY':
            base_size = 1.0
        elif action == 'SELL':
            base_size = -1.0
        else:
            base_size = 0.0
        
        # Adjust by reliability
        if reliability_score >= 0.85:
            size_multiplier = 1.0  # Full position
            conviction = 'VERY_HIGH'
        elif reliability_score >= 0.70:
            size_multiplier = 0.85  # 85% position
            conviction = 'HIGH'
        elif reliability_score >= 0.55:
            size_multiplier = 0.60  # 60% position
            conviction = 'MODERATE'
        else:
            size_multiplier = 0.30  # 30% position
            conviction = 'LOW'
        
        position_size = base_size * size_multiplier
        
        return {
            'action': action,
            'position_size': position_size,
            'position_pct': abs(position_size) * 100,
            'conviction': conviction,
            'reliability_score': reliability_score,
            'signal_quality': signal_quality,
            'recommendation': self._generate_recommendation(
                action, conviction, signal_quality, reliability_score
            )
        }
    
    def _generate_recommendation(self, action: str, conviction: str,
                                signal_quality: str, reliability: float) -> str:
        """Generate human-readable recommendation"""
        action_text = action.lower()
        
        if signal_quality == 'EXCELLENT' and conviction in ['VERY_HIGH', 'HIGH']:
            return f"Strong {action_text} signal with {reliability*100:.0f}% reliability. " \
                   f"High conviction - recommended full position."
        
        elif signal_quality in ['GOOD', 'EXCELLENT']:
            return f"{action_text.capitalize()} signal with {reliability*100:.0f}% reliability. " \
                   f"Good quality - recommended {conviction.lower()} conviction position."
        
        elif signal_quality == 'FAIR':
            return f"Moderate {action_text} signal ({reliability*100:.0f}% reliability). " \
                   f"Fair quality - consider reduced position or wait for confirmation."
        
        else:
            return f"Weak {action_text} signal ({reliability*100:.0f}% reliability). " \
                   f"Poor quality - minimal position or hold recommended."


class DisagreementAnalyzer:
    """
    Analyzes cases where agents disagree
    Principle: Disagreement = lower conviction, suitable for smaller position
    """
    
    def __init__(self):
        self.high_disagreement_threshold = 0.4  # 40% variance in votes
        
    def analyze_disagreement(self, decisions: List[EnhancedAgentDecision]) -> Dict:
        """
        Analyze the nature and implications of agent disagreement
        
        Args:
            decisions: List of agent decisions
            
        Returns:
            Disagreement analysis
        """
        if len(decisions) < 2:
            return {
                'disagreement_level': 'NONE',
                'confidence_reduction': 0.0,
                'market_regime': 'UNCERTAIN'
            }
        
        # Count votes
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for d in decisions:
            votes[d.action] += 1
        
        total = len(decisions)
        max_votes = max(votes.values())
        
        # Calculate disagreement
        disagreement_ratio = 1.0 - (max_votes / total)
        
        if disagreement_ratio < 0.2:
            disagreement_level = 'LOW'
            confidence_reduction = 0.0
            market_regime = 'TRENDING'
        elif disagreement_ratio < 0.4:
            disagreement_level = 'MODERATE'
            confidence_reduction = 0.2
            market_regime = 'MIXED'
        else:
            disagreement_level = 'HIGH'
            confidence_reduction = 0.4
            market_regime = 'CHOPPY'
        
        # Identify conflicting agent types
        buy_agents = [d.agent_type for d in decisions if d.action == 'BUY']
        sell_agents = [d.agent_type for d in decisions if d.action == 'SELL']
        
        return {
            'disagreement_level': disagreement_level,
            'disagreement_ratio': disagreement_ratio,
            'confidence_reduction': confidence_reduction,
            'market_regime': market_regime,
            'vote_distribution': votes,
            'buy_agent_types': buy_agents,
            'sell_agent_types': sell_agents,
            'recommendation': self._generate_disagreement_recommendation(
                disagreement_level, market_regime
            )
        }
    
    def _generate_disagreement_recommendation(self, level: str, regime: str) -> str:
        """Generate recommendation for disagreement scenarios"""
        if level == 'HIGH':
            return f"High agent disagreement detected. Market regime: {regime}. " \
                   f"Recommend minimal position (20-30%) or wait for clarity."
        elif level == 'MODERATE':
            return f"Moderate agent disagreement. Market regime: {regime}. " \
                   f"Recommend reduced position (50-60%) with tight stops."
        else:
            return f"Low agent disagreement. Market regime: {regime}. " \
                   f"Good consensus - normal position sizing applies."


class SentimentExtremeDetector:
    """
    Detects sentiment extremes where emotions override logic
    Principle: Multi-agent consensus failure at extremes = human emotions dominating
    """
    
    def __init__(self):
        self.extreme_threshold = 0.85  # 85% sentiment = extreme
        
    def detect_extreme(self, sentiment_score: float,
                      social_sentiment: float,
                      consensus_action: str) -> Dict:
        """
        Detect if sentiment is at extreme levels
        
        Args:
            sentiment_score: Overall sentiment score [0, 1]
            social_sentiment: Social media sentiment [0, 1]
            consensus_action: Agent consensus action
            
        Returns:
            Extreme sentiment analysis
        """
        is_extreme_bullish = sentiment_score > self.extreme_threshold
        is_extreme_bearish = sentiment_score < (1 - self.extreme_threshold)
        
        social_extreme_bull = social_sentiment > self.extreme_threshold
        social_extreme_bear = social_sentiment < (1 - self.extreme_threshold)
        
        if is_extreme_bullish or social_extreme_bull:
            extreme_type = 'EXTREME_BULLISH'
            risk = 'HIGH'
            warning = "Sentiment at extreme bullish levels. Risk of reversal high. " \
                     "Consider contrarian positioning or reduced exposure."
            
            if consensus_action == 'BUY':
                recommendation = 'REDUCE_SIZE'
            else:
                recommendation = 'CONSIDER_CONTRARIAN'
                
        elif is_extreme_bearish or social_extreme_bear:
            extreme_type = 'EXTREME_BEARISH'
            risk = 'HIGH'
            warning = "Sentiment at extreme bearish levels. Possible capitulation. " \
                     "Consider contrarian opportunity or wait for stabilization."
            
            if consensus_action == 'SELL':
                recommendation = 'REDUCE_SIZE'
            else:
                recommendation = 'CONSIDER_CONTRARIAN'
        else:
            extreme_type = 'NORMAL'
            risk = 'MODERATE'
            warning = "Sentiment within normal range. Standard position sizing applies."
            recommendation = 'NORMAL'
        
        return {
            'extreme_type': extreme_type,
            'risk_level': risk,
            'warning': warning,
            'recommendation': recommendation,
            'sentiment_score': sentiment_score,
            'social_sentiment': social_sentiment,
            'is_extreme': is_extreme_bullish or is_extreme_bearish or \
                         social_extreme_bull or social_extreme_bear
        }
