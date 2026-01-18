"""
Game Theory Stock Agent
Multi-agent system for stock market analysis using game-theoretic principles
Enhanced with Bull, Bear, Technical, Sentiment, and Fundamental agents
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class AgentStrategy(Enum):
    """Agent decision strategies"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    CONTRARIAN = "contrarian"


class AgentType(Enum):
    """Agent specialized types"""
    BULL = "Bull"
    BEAR = "Bear"
    TECHNICAL = "Technical"
    SENTIMENT = "Sentiment"
    FUNDAMENTAL = "Fundamental"
    CONTRARIAN = "Contrarian"


@dataclass
class AgentDecision:
    """Represents an agent's investment decision"""
    agent_id: str
    strategy: AgentStrategy
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float
    allocation: float  # percentage
    agent_type: str = "Balanced"  # Agent type for consensus analysis


class GameTheoryAgent:
    """Individual AI agent with game-theoretic decision making"""
    
    def __init__(self, agent_id: str, strategy: AgentStrategy, risk_tolerance: float = 0.5, agent_type: str = "Balanced"):
        self.agent_id = agent_id
        self.strategy = strategy
        self.risk_tolerance = risk_tolerance
        self.agent_type = agent_type
        self.payoff_history = []
        
    def calculate_payoff(self, action: str, market_sentiment: float, 
                        other_agents_actions: List[str]) -> float:
        """
        Calculate expected payoff based on game theory
        Uses concepts from Nash equilibrium and prisoner's dilemma
        """
        base_payoff = 0.0
        
        # Market sentiment impact
        if action == "BUY":
            base_payoff = market_sentiment * 10
        elif action == "SELL":
            base_payoff = (1 - market_sentiment) * 10
        else:  # HOLD
            base_payoff = 5.0
            
        # Game theory: coordination bonus (Nash equilibrium concept)
        buy_count = sum(1 for a in other_agents_actions if a == "BUY")
        sell_count = sum(1 for a in other_agents_actions if a == "SELL")
        
        if action == "BUY" and buy_count > len(other_agents_actions) / 2:
            base_payoff += 3.0  # Coordination bonus
        elif action == "SELL" and sell_count > len(other_agents_actions) / 2:
            base_payoff += 3.0
            
        # Contrarian strategy bonus
        if self.strategy == AgentStrategy.CONTRARIAN:
            if action == "BUY" and sell_count > buy_count:
                base_payoff += 5.0
            elif action == "SELL" and buy_count > sell_count:
                base_payoff += 5.0
                
        return base_payoff * self.risk_tolerance
        
    def decide(self, market_data: Dict, sentiment: float, 
               other_agents_actions: List[str]) -> AgentDecision:
        """Make investment decision based on strategy and game theory"""
        actions = ["BUY", "SELL", "HOLD"]
        payoffs = []
        
        for action in actions:
            payoff = self.calculate_payoff(action, sentiment, other_agents_actions)
            payoffs.append(payoff)
            
        # Strategy-specific adjustments
        if self.strategy == AgentStrategy.AGGRESSIVE:
            payoffs[0] *= 1.5  # Prefer buying
        elif self.strategy == AgentStrategy.CONSERVATIVE:
            payoffs[2] *= 1.5  # Prefer holding
        elif self.strategy == AgentStrategy.BALANCED:
            pass  # No adjustment
            
        best_action_idx = np.argmax(payoffs)
        best_action = actions[best_action_idx]
        confidence = payoffs[best_action_idx] / (sum(payoffs) + 1e-6)
        
        # Calculate allocation based on confidence and risk tolerance
        allocation = confidence * self.risk_tolerance * 100
        
        return AgentDecision(
            agent_id=self.agent_id,
            strategy=self.strategy,
            action=best_action,
            confidence=confidence,
            allocation=min(allocation, 100.0),
            agent_type=self.agent_type
        )


class MultiAgentSystem:
    """Manages multiple agents and their interactions"""
    
    def __init__(self):
        self.agents: List[GameTheoryAgent] = []
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize diverse set of agents with different strategies"""
        strategies_config = [
            ("Agent_Alpha_Bull", AgentStrategy.AGGRESSIVE, 0.8, "Bull"),
            ("Agent_Beta_Bear", AgentStrategy.CONSERVATIVE, 0.3, "Bear"),
            ("Agent_Gamma_Technical", AgentStrategy.BALANCED, 0.5, "Technical"),
            ("Agent_Delta_Contrarian", AgentStrategy.CONTRARIAN, 0.6, "Contrarian"),
            ("Agent_Epsilon_Sentiment", AgentStrategy.BALANCED, 0.5, "Sentiment"),
            ("Agent_Zeta_Fundamental", AgentStrategy.BALANCED, 0.6, "Fundamental"),
        ]
        
        for agent_id, strategy, risk, agent_type in strategies_config:
            self.agents.append(GameTheoryAgent(agent_id, strategy, risk, agent_type))
            
    def run_simulation(self, market_data: Dict, sentiment: float) -> List[AgentDecision]:
        """
        Run multi-agent simulation
        Implements iterative best response for Nash equilibrium
        """
        # Initial round: agents make decisions without knowing others
        decisions = []
        for agent in self.agents:
            decision = agent.decide(market_data, sentiment, [])
            decisions.append(decision)
            
        # Iterative refinement (best response dynamics)
        for iteration in range(3):
            new_decisions = []
            actions = [d.action for d in decisions]
            
            for i, agent in enumerate(self.agents):
                # Get actions of other agents
                other_actions = actions[:i] + actions[i+1:]
                decision = agent.decide(market_data, sentiment, other_actions)
                new_decisions.append(decision)
                
            decisions = new_decisions
            
        return decisions
        
    def calculate_nash_equilibrium(self, decisions: List[AgentDecision]) -> Dict:
        """
        Analyze if current decisions represent a Nash equilibrium
        Returns equilibrium metrics
        """
        actions = [d.action for d in decisions]
        buy_ratio = actions.count("BUY") / len(actions)
        sell_ratio = actions.count("SELL") / len(actions)
        hold_ratio = actions.count("HOLD") / len(actions)
        
        # Calculate stability score (how close to equilibrium)
        variance = np.var([buy_ratio, sell_ratio, hold_ratio])
        stability = 1.0 - min(variance * 3, 1.0)
        
        return {
            "buy_ratio": buy_ratio,
            "sell_ratio": sell_ratio,
            "hold_ratio": hold_ratio,
            "stability_score": stability,
            "is_equilibrium": stability > 0.7
        }
        
    def get_consensus_decision(self, decisions: List[AgentDecision]) -> str:
        """Get weighted consensus from all agents"""
        vote_weights = {"BUY": 0, "SELL": 0, "HOLD": 0}
        
        for decision in decisions:
            vote_weights[decision.action] += decision.confidence
            
        consensus = max(vote_weights, key=vote_weights.get)
        return consensus
