"""
Biweekly Signal Generator
Generates BUY/HOLD/SELL signals based on comprehensive multi-factor analysis
"""

from typing import Dict, List
from datetime import datetime, timedelta
import numpy as np


class BiweeklySignalGenerator:
    """
    Generates actionable signals every 2 weeks for biweekly investment strategy
    """
    
    def __init__(self):
        self.signal_history = []
        self.last_signal_date = None
        
    def generate_signal(self, 
                       multi_agent_consensus: Dict,
                       sentiment_divergence: Dict,
                       prisoners_dilemma: Dict,
                       technical_signal: str,
                       analyst_sentiment: float,
                       current_price: float) -> Dict:
        """
        Generate comprehensive trading signal
        
        Args:
            multi_agent_consensus: Multi-agent consensus analysis
            sentiment_divergence: Sentiment divergence analysis
            prisoners_dilemma: Prisoner's dilemma capex analysis
            technical_signal: Technical analysis signal
            analyst_sentiment: Analyst sentiment score
            current_price: Current stock price
            
        Returns:
            Comprehensive signal with all factors
        """
        # Extract key metrics
        consensus_action = multi_agent_consensus.get('consensus_action', 'HOLD')
        reliability_score = multi_agent_consensus.get('reliability_score', 0.5)
        signal_quality = multi_agent_consensus.get('signal_quality', 'FAIR')
        
        divergence_signal = sentiment_divergence.get('signal', 'HOLD')
        divergence_confidence = sentiment_divergence.get('confidence', 0.5)
        
        pd_recommendation = prisoners_dilemma.get('recommendation', 'HOLD')
        pd_risk_level = prisoners_dilemma.get('risk_level', 'MODERATE')
        
        # Calculate composite signal
        signal_scores = {
            'BUY': 0.0,
            'SELL': 0.0,
            'HOLD': 0.0
        }
        
        # Multi-agent consensus (40% weight)
        signal_scores[consensus_action] += 0.4 * reliability_score
        
        # Sentiment divergence (25% weight)
        if divergence_signal == 'STRONG':
            # When aligned, use direction from divergence analysis
            expected_dir = sentiment_divergence.get('expected_direction', 'NEUTRAL')
            if expected_dir == 'UP':
                signal_scores['BUY'] += 0.25 * divergence_confidence
            elif expected_dir == 'DOWN':
                signal_scores['SELL'] += 0.25 * divergence_confidence
            else:
                signal_scores['HOLD'] += 0.25 * divergence_confidence
        else:
            signal_scores[divergence_signal] += 0.25 * divergence_confidence
        
        # Prisoner's dilemma (20% weight)
        pd_weight = 0.2
        if pd_risk_level == 'HIGH':
            signal_scores['SELL'] += pd_weight * 0.8
            signal_scores['HOLD'] += pd_weight * 0.2
        elif pd_risk_level in ['LOW', 'MODERATE']:
            if pd_recommendation == 'BUY':
                signal_scores['BUY'] += pd_weight * 0.8
            elif pd_recommendation == 'SELL':
                signal_scores['SELL'] += pd_weight * 0.8
            else:
                signal_scores['HOLD'] += pd_weight
        
        # Technical signal (15% weight)
        tech_weight = 0.15
        signal_scores[technical_signal] += tech_weight
        
        # Determine final signal
        final_signal = max(signal_scores, key=signal_scores.get)
        signal_strength = signal_scores[final_signal]
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            reliability_score, divergence_confidence, 
            signal_quality, signal_strength
        )
        
        # Generate recommendation with position sizing
        recommendation = self._generate_recommendation(
            final_signal, confidence, signal_quality, 
            pd_risk_level, current_price
        )
        
        # Record signal
        signal_data = {
            'signal': final_signal,
            'confidence': confidence,
            'signal_strength': signal_strength,
            'signal_quality': signal_quality,
            'recommendation': recommendation,
            'position_size': recommendation['position_size'],
            'factors': {
                'multi_agent': {
                    'action': consensus_action,
                    'reliability': reliability_score,
                    'quality': signal_quality
                },
                'sentiment_divergence': {
                    'signal': divergence_signal,
                    'confidence': divergence_confidence,
                    'is_divergent': sentiment_divergence.get('is_divergent', False)
                },
                'prisoners_dilemma': {
                    'recommendation': pd_recommendation,
                    'risk_level': pd_risk_level,
                    'analysis': prisoners_dilemma.get('analysis', 'UNKNOWN')
                },
                'technical': technical_signal,
                'analyst_sentiment': analyst_sentiment
            },
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price
        }
        
        self.signal_history.append(signal_data)
        self.last_signal_date = datetime.now()
        
        return signal_data
    
    def _calculate_overall_confidence(self, reliability: float, 
                                     divergence_conf: float,
                                     quality: str, 
                                     strength: float) -> float:
        """Calculate overall confidence score"""
        # Base confidence from signal strength
        base_conf = strength
        
        # Boost from reliability
        reliability_boost = reliability * 0.2
        
        # Boost from divergence confidence
        divergence_boost = divergence_conf * 0.1
        
        # Quality adjustment
        quality_multiplier = {
            'EXCELLENT': 1.2,
            'GOOD': 1.1,
            'FAIR': 1.0,
            'POOR': 0.8
        }.get(quality, 1.0)
        
        confidence = (base_conf + reliability_boost + divergence_boost) * quality_multiplier
        
        return min(confidence, 1.0)
    
    def _generate_recommendation(self, signal: str, confidence: float,
                                quality: str, risk_level: str,
                                current_price: float) -> Dict:
        """
        Generate detailed recommendation with position sizing
        """
        # Base position size
        if signal == 'BUY':
            base_size = 1.0
            action_text = "Initiate long position"
        elif signal == 'SELL':
            base_size = -0.5  # Half size for sells (more conservative)
            action_text = "Reduce exposure or take profits"
        else:
            base_size = 0.0
            action_text = "Maintain current position"
        
        # Conviction levels as constants
        VERY_HIGH_THRESHOLD = 0.80
        HIGH_THRESHOLD = 0.65
        MODERATE_THRESHOLD = 0.50
        
        # Adjust by confidence
        if confidence >= VERY_HIGH_THRESHOLD:
            size_mult = 1.0
            conviction = "VERY HIGH"
        elif confidence >= HIGH_THRESHOLD:
            size_mult = 0.85
            conviction = "HIGH"
        elif confidence >= MODERATE_THRESHOLD:
            size_mult = 0.60
            conviction = "MODERATE"
        else:
            size_mult = 0.35
            conviction = "LOW"
        
        # Adjust by risk level
        if risk_level == 'HIGH':
            size_mult *= 0.7
        elif risk_level == 'LOW':
            size_mult *= 1.1
        
        position_size = base_size * size_mult
        
        # Generate stop loss and target
        if signal == 'BUY':
            stop_loss = current_price * 0.92  # 8% stop loss
            target_price = current_price * 1.15  # 15% target
        elif signal == 'SELL':
            stop_loss = current_price * 1.08  # 8% stop (for shorts)
            target_price = current_price * 0.85  # 15% target
        else:
            stop_loss = None
            target_price = None
        
        return {
            'action': signal,
            'action_text': action_text,
            'position_size': abs(position_size),
            'position_pct': abs(position_size) * 100,
            'conviction': conviction,
            'confidence': confidence,
            'signal_quality': quality,
            'risk_level': risk_level,
            'stop_loss': stop_loss,
            'target_price': target_price,
            'risk_reward_ratio': 1.875 if signal in ['BUY', 'SELL'] else 0,  # 15%/8%
            'recommendation_text': self._generate_text(
                signal, conviction, confidence, quality, risk_level
            )
        }
    
    def _generate_text(self, signal: str, conviction: str,
                      confidence: float, quality: str, 
                      risk_level: str) -> str:
        """Generate human-readable recommendation text"""
        action = signal.lower()
        
        text = f"{conviction} conviction {action} signal ({confidence*100:.0f}% confidence). "
        text += f"Signal quality: {quality}. "
        text += f"Risk level: {risk_level}. "
        
        if conviction in ['VERY HIGH', 'HIGH'] and quality in ['EXCELLENT', 'GOOD']:
            text += f"Strong consensus across multiple factors. "
            text += f"Recommended position: {conviction.lower()} conviction."
        elif quality == 'FAIR' or conviction == 'MODERATE':
            text += f"Moderate signal strength. Consider reduced position sizing."
        else:
            text += f"Weak signal. Minimal position or wait for better setup."
        
        return text
    
    def should_generate_new_signal(self) -> bool:
        """
        Check if it's time to generate a new biweekly signal
        
        Returns:
            True if 14 days have passed since last signal
        """
        if self.last_signal_date is None:
            return True
        
        days_since_last = (datetime.now() - self.last_signal_date).days
        return days_since_last >= 14
    
    def get_days_to_next_signal(self) -> int:
        """Get number of days until next signal generation"""
        if self.last_signal_date is None:
            return 0
        
        days_since_last = (datetime.now() - self.last_signal_date).days
        return max(14 - days_since_last, 0)
    
    def get_signal_history(self, limit: int = 10) -> List[Dict]:
        """Get recent signal history"""
        return self.signal_history[-limit:] if self.signal_history else []
    
    def get_signal_performance_summary(self) -> Dict:
        """Get summary of signal performance (for future backtesting)"""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_confidence': 0.0
            }
        
        buy_count = sum(1 for s in self.signal_history if s['signal'] == 'BUY')
        sell_count = sum(1 for s in self.signal_history if s['signal'] == 'SELL')
        hold_count = sum(1 for s in self.signal_history if s['signal'] == 'HOLD')
        
        # Safe calculation of average confidence
        confidences = [s.get('confidence', 0) for s in self.signal_history]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        total = len(self.signal_history)
        
        return {
            'total_signals': total,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'avg_confidence': avg_confidence,
            'buy_ratio': buy_count / total if total > 0 else 0,
            'sell_ratio': sell_count / total if total > 0 else 0,
            'hold_ratio': hold_count / total if total > 0 else 0
        }


class TacticalAdjustment:
    """
    Weekly tactical adjustments based on sentiment divergence changes
    """
    
    def __init__(self):
        self.last_check = None
        
    def check_for_adjustment(self, current_signal: str,
                            sentiment_divergence: Dict,
                            previous_divergence: Dict = None) -> Dict:
        """
        Check if tactical adjustment is needed based on sentiment changes
        
        Args:
            current_signal: Current biweekly signal
            sentiment_divergence: Current sentiment divergence
            previous_divergence: Previous sentiment divergence (if available)
            
        Returns:
            Adjustment recommendation
        """
        if previous_divergence is None:
            return {
                'adjust': False,
                'recommendation': 'MAINTAIN',
                'reason': 'Insufficient historical data'
            }
        
        # Check for significant divergence changes
        current_div = sentiment_divergence.get('divergence_magnitude', 0)
        previous_div = previous_divergence.get('divergence_magnitude', 0)
        
        div_change = abs(current_div - previous_div)
        
        # Significant change threshold: 20%
        if div_change > 0.20:
            if current_signal == 'BUY' and sentiment_divergence.get('signal') == 'SELL':
                return {
                    'adjust': True,
                    'recommendation': 'REDUCE_POSITION',
                    'adjustment_pct': 0.3,  # Reduce by 30%
                    'reason': 'Sentiment divergence shifted against position'
                }
            elif current_signal == 'SELL' and sentiment_divergence.get('signal') == 'BUY':
                return {
                    'adjust': True,
                    'recommendation': 'COVER_POSITION',
                    'adjustment_pct': 0.3,
                    'reason': 'Sentiment divergence shifted against position'
                }
            elif current_signal == 'HOLD' and sentiment_divergence.get('signal') == 'STRONG':
                return {
                    'adjust': True,
                    'recommendation': 'CONSIDER_ENTRY',
                    'adjustment_pct': 0.5,
                    'reason': 'Strong aligned sentiment emerged'
                }
        
        return {
            'adjust': False,
            'recommendation': 'MAINTAIN',
            'reason': 'No significant divergence change'
        }
    
    def should_check(self) -> bool:
        """Check if weekly review is due"""
        if self.last_check is None:
            return True
        
        days_since_check = (datetime.now() - self.last_check).days
        return days_since_check >= 7
