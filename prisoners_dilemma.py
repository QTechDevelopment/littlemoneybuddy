"""
Prisoner's Dilemma AI Capex Analysis Module
Analyzes AI spending patterns using game theory to identify valuation risks
"""

import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CompanyCapex:
    """Represents a company's capex data"""
    ticker: str
    name: str
    revenue: float
    capex: float
    market_cap: float
    is_ai_heavy: bool
    capex_revenue_ratio: float


class PrisonersDilemmaCapexAnalyzer:
    """
    Analyzes AI capex spending patterns using Prisoner's Dilemma framework
    
    Principle: When multiple mega-caps overspend on AI capex simultaneously,
    it creates a prisoner's dilemma situation leading to valuation compression
    """
    
    def __init__(self):
        self.ai_heavy_threshold = 0.15  # 15% capex/revenue ratio
        self.overspending_threshold = 0.20  # 20% capex/revenue ratio
        self.mega_cap_threshold = 100_000_000_000  # $100B market cap
        
        # AI-heavy sector tickers
        self.ai_heavy_sectors = ['Technology', 'Communication Services']
        self.mega_cap_ai_tickers = [
            'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AAPL', 'TSLA'
        ]
        
    def get_company_capex_data(self, ticker: str) -> Optional[CompanyCapex]:
        """
        Fetch company's capex and revenue data
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CompanyCapex object or None if data unavailable
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            financials = stock.financials
            cashflow = stock.cashflow
            
            # Get revenue
            if financials is not None and not financials.empty:
                if 'Total Revenue' in financials.index:
                    revenue = financials.loc['Total Revenue'].iloc[0]
                else:
                    revenue = info.get('totalRevenue', 0)
            else:
                revenue = info.get('totalRevenue', 0)
            
            # Get capex
            if cashflow is not None and not cashflow.empty:
                if 'Capital Expenditure' in cashflow.index:
                    capex = abs(cashflow.loc['Capital Expenditure'].iloc[0])
                else:
                    capex = 0
            else:
                capex = 0
            
            # Calculate capex/revenue ratio
            capex_ratio = capex / revenue if revenue > 0 else 0
            
            # Determine if AI-heavy
            sector = info.get('sector', '')
            is_ai_heavy = (
                sector in self.ai_heavy_sectors or 
                ticker in self.mega_cap_ai_tickers or
                capex_ratio > self.ai_heavy_threshold
            )
            
            return CompanyCapex(
                ticker=ticker,
                name=info.get('longName', ticker),
                revenue=revenue,
                capex=capex,
                market_cap=info.get('marketCap', 0),
                is_ai_heavy=is_ai_heavy,
                capex_revenue_ratio=capex_ratio
            )
            
        except Exception as e:
            print(f"Error fetching capex data for {ticker}: {e}")
            return None
    
    def analyze_competitive_capex(self, ticker: str, 
                                   competitors: List[str] = None) -> Dict:
        """
        Analyze capex patterns among competitors using Prisoner's Dilemma
        
        Args:
            ticker: Target stock ticker
            competitors: List of competitor tickers (auto-detected if None)
            
        Returns:
            Prisoner's Dilemma analysis
        """
        # Get target company data
        target_data = self.get_company_capex_data(ticker)
        
        if target_data is None:
            return {
                'analysis': 'INSUFFICIENT_DATA',
                'risk_level': 'UNKNOWN',
                'recommendation': 'HOLD',
                'confidence': 0.0
            }
        
        # Auto-detect competitors if not provided
        if competitors is None:
            if ticker in self.mega_cap_ai_tickers:
                competitors = [t for t in self.mega_cap_ai_tickers if t != ticker]
            else:
                competitors = []
        
        # Get competitor data
        competitor_data = []
        for comp_ticker in competitors:
            data = self.get_company_capex_data(comp_ticker)
            if data is not None:
                competitor_data.append(data)
        
        if not competitor_data:
            return {
                'analysis': 'NO_COMPETITOR_DATA',
                'risk_level': 'LOW',
                'recommendation': 'NEUTRAL',
                'confidence': 0.3,
                'target_capex_ratio': target_data.capex_revenue_ratio
            }
        
        # Analyze Prisoner's Dilemma situation
        analysis = self._apply_prisoners_dilemma(target_data, competitor_data)
        
        return analysis
    
    def _apply_prisoners_dilemma(self, target: CompanyCapex, 
                                 competitors: List[CompanyCapex]) -> Dict:
        """
        Apply Prisoner's Dilemma framework to capex analysis
        
        Scenarios:
        1. Both cooperate (low capex) = moderate returns for all
        2. Both defect (high capex) = low returns for all (worst outcome)
        3. One defects, one cooperates = defector wins, cooperator loses
        
        In AI race: Overspending = defect, Disciplined = cooperate
        """
        # Count overspending competitors
        overspending_competitors = [
            c for c in competitors 
            if c.capex_revenue_ratio > self.overspending_threshold
        ]
        
        # Calculate average competitor capex ratio
        avg_competitor_capex = np.mean([c.capex_revenue_ratio for c in competitors])
        
        # Determine target's position
        target_overspending = target.capex_revenue_ratio > self.overspending_threshold
        target_disciplined = target.capex_revenue_ratio < self.ai_heavy_threshold
        
        # Analyze scenario
        num_overspending = len(overspending_competitors)
        total_competitors = len(competitors)
        overspending_ratio = num_overspending / total_competitors if total_competitors > 0 else 0
        
        # Scenario 1: Multiple competitors overspending (Mutual Defection)
        if overspending_ratio > 0.5:
            if target_overspending:
                # Target also overspending = worst outcome
                analysis_type = "MUTUAL_DEFECTION"
                risk_level = "HIGH"
                recommendation = "SELL"
                reasoning = "Multiple competitors overspending on AI capex. Prisoner's dilemma " \
                           "mutual defection scenario - all players suffer from valuation compression."
                confidence = 0.8
                valuation_risk = "COMPRESSION"
                
            else:
                # Target disciplined while others overspend = potential winner
                analysis_type = "DISCIPLINED_SPENDER"
                risk_level = "LOW"
                recommendation = "BUY"
                reasoning = "Company maintains disciplined capex while competitors overspend. " \
                           "Positioned to benefit from competitor inefficiency."
                confidence = 0.75
                valuation_risk = "EXPANSION"
        
        # Scenario 2: Few competitors overspending
        elif overspending_ratio > 0 and overspending_ratio <= 0.5:
            if target_overspending:
                # Target overspending with minority = risky
                analysis_type = "AGGRESSIVE_INVESTOR"
                risk_level = "MODERATE_HIGH"
                recommendation = "HOLD"
                reasoning = "Company overspending on AI capex. Risk of diminishing returns if " \
                           "market doesn't validate investment thesis."
                confidence = 0.6
                valuation_risk = "NEUTRAL"
                
            else:
                # Target disciplined = good position
                analysis_type = "BALANCED_APPROACH"
                risk_level = "LOW"
                recommendation = "BUY"
                reasoning = "Disciplined capex approach while maintaining competitiveness. " \
                           "Lower risk profile compared to aggressive spenders."
                confidence = 0.7
                valuation_risk = "STABLE"
        
        # Scenario 3: No overspending (Mutual Cooperation)
        else:
            analysis_type = "MUTUAL_COOPERATION"
            risk_level = "LOW"
            recommendation = "BUY"
            reasoning = "Industry showing disciplined capex management. Mutual cooperation " \
                       "scenario where all players benefit from sustainable spending."
            confidence = 0.85
            valuation_risk = "STABLE"
        
        # Calculate competitive positioning score
        positioning_score = self._calculate_positioning_score(
            target.capex_revenue_ratio,
            avg_competitor_capex
        )
        
        return {
            'analysis': analysis_type,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'confidence': confidence,
            'valuation_risk': valuation_risk,
            'target_capex_ratio': target.capex_revenue_ratio,
            'competitor_avg_capex_ratio': avg_competitor_capex,
            'overspending_competitors': num_overspending,
            'total_competitors': total_competitors,
            'overspending_ratio': overspending_ratio,
            'competitive_positioning': positioning_score,
            'is_ai_beneficiary': not target_overspending and target.is_ai_heavy
        }
    
    def _calculate_positioning_score(self, target_ratio: float, 
                                     competitor_avg: float) -> float:
        """
        Calculate competitive positioning score [0, 1]
        Higher is better (disciplined spending with competitiveness)
        """
        if competitor_avg == 0:
            return 0.5
        
        # Ideal scenario: spend less than competitors but not too little
        ratio_diff = competitor_avg - target_ratio
        
        if ratio_diff > 0.05:
            # Spending significantly less = good positioning
            score = 0.8
        elif ratio_diff < -0.05:
            # Spending significantly more = poor positioning
            score = 0.3
        else:
            # Similar spending = neutral
            score = 0.5
        
        # Adjust for absolute spending level
        if target_ratio > self.overspending_threshold:
            score *= 0.7  # Penalize overspending
        elif target_ratio < 0.05:
            score *= 0.8  # Penalize underspending
        
        return min(max(score, 0), 1)
    
    def identify_ai_beneficiaries(self, ticker: str) -> Dict:
        """
        Identify if stock is an AI beneficiary rather than AI investor
        
        AI Beneficiaries: Benefit from AI without heavy capex
        Examples: Software companies, AI service providers
        
        Returns:
            Analysis of AI beneficiary status
        """
        company_data = self.get_company_capex_data(ticker)
        
        if company_data is None:
            return {
                'is_beneficiary': False,
                'confidence': 0.0,
                'category': 'UNKNOWN'
            }
        
        # Criteria for AI beneficiary:
        # 1. In tech/AI sector
        # 2. Low capex/revenue ratio
        # 3. Revenue growth potential from AI
        
        is_beneficiary = (
            company_data.is_ai_heavy and 
            company_data.capex_revenue_ratio < self.ai_heavy_threshold
        )
        
        if is_beneficiary:
            category = "AI_BENEFICIARY"
            confidence = 0.7
            reasoning = "Company positioned to benefit from AI adoption without heavy capital " \
                       "expenditure. Lower risk, stable margin profile."
        elif company_data.capex_revenue_ratio > self.overspending_threshold:
            category = "AI_INVESTOR"
            confidence = 0.8
            reasoning = "Heavy AI infrastructure investor. Higher risk, potential for higher returns " \
                       "if investments pay off."
        else:
            category = "BALANCED"
            confidence = 0.6
            reasoning = "Balanced approach to AI investments. Moderate risk/reward profile."
        
        return {
            'is_beneficiary': is_beneficiary,
            'confidence': confidence,
            'category': category,
            'reasoning': reasoning,
            'capex_ratio': company_data.capex_revenue_ratio,
            'recommendation': "BUY" if is_beneficiary else "HOLD"
        }


class NashEquilibriumCapexAnalyzer:
    """
    Analyzes whether competitor capex spending has reached Nash equilibrium
    """
    
    def __init__(self):
        self.convergence_threshold = 0.05  # 5% difference = equilibrium
        
    def analyze_equilibrium(self, competitor_capex_ratios: List[float]) -> Dict:
        """
        Determine if competitor capex ratios have converged to Nash equilibrium
        
        Args:
            competitor_capex_ratios: List of capex/revenue ratios for competitors
            
        Returns:
            Nash equilibrium analysis
        """
        if len(competitor_capex_ratios) < 2:
            return {
                'is_equilibrium': False,
                'convergence_score': 0.0,
                'stability': 'UNKNOWN',
                'recommendation': 'HOLD'
            }
        
        # Calculate variance
        variance = np.var(competitor_capex_ratios)
        std_dev = np.std(competitor_capex_ratios)
        mean_ratio = np.mean(competitor_capex_ratios)
        
        # Coefficient of variation
        cv = std_dev / mean_ratio if mean_ratio > 0 else 1.0
        
        # Convergence score (lower variance = higher score)
        convergence_score = max(1.0 - cv, 0)
        
        # Determine equilibrium status
        is_equilibrium = cv < self.convergence_threshold
        
        if is_equilibrium:
            stability = "STABLE"
            risk_level = "LOW"
            recommendation = "BUY"
            reasoning = "Competitor capex ratios have converged to Nash equilibrium. " \
                       "Stable competitive environment with lower risk."
        else:
            stability = "UNSTABLE"
            risk_level = "MODERATE"
            recommendation = "HOLD"
            reasoning = "Competitor capex ratios show high variance. Competitive dynamics " \
                       "still evolving, higher uncertainty."
        
        # Identify outliers
        outliers = []
        for i, ratio in enumerate(competitor_capex_ratios):
            z_score = abs(ratio - mean_ratio) / std_dev if std_dev > 0 else 0
            if z_score > 2:
                outliers.append({
                    'index': i,
                    'ratio': ratio,
                    'z_score': z_score,
                    'type': 'MOMENTUM' if ratio > mean_ratio else 'VALUE_TRAP'
                })
        
        return {
            'is_equilibrium': is_equilibrium,
            'convergence_score': convergence_score,
            'stability': stability,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'reasoning': reasoning,
            'mean_capex_ratio': mean_ratio,
            'std_deviation': std_dev,
            'coefficient_of_variation': cv,
            'outliers': outliers,
            'num_outliers': len(outliers)
        }
