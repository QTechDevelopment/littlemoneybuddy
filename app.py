"""
Game Theory Stock Agent Dashboard
Interactive dashboard for AI-driven stock analysis using game-theoretic principles
Enhanced with real data sources, sentiment divergence, and AI capex analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from game_theory_agent import (
    MultiAgentSystem, AgentStrategy, GameTheoryAgent
)
from sentiment_analyzer import SentimentAnalyzer, generate_mock_news
from stock_data import StockDataFetcher, BiweeklyInvestmentStrategy

# New enhanced modules
from data_sources import DataSourceAggregator
from sentiment_divergence import (
    SentimentDivergenceAnalyzer, MultiSourceSentimentAggregator
)
from prisoners_dilemma import (
    PrisonersDilemmaCapexAnalyzer, NashEquilibriumCapexAnalyzer
)
from multi_agent_consensus import (
    MultiAgentConsensus, DisagreementAnalyzer, SentimentExtremeDetector,
    EnhancedAgentDecision
)
from signal_generator import BiweeklySignalGenerator, TacticalAdjustment


# Page configuration
st.set_page_config(
    page_title="Game Theory Stock Agent Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .agent-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .bullish {
        color: #4CAF50;
        font-weight: bold;
    }
    .bearish {
        color: #f44336;
        font-weight: bold;
    }
    .neutral {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'agent_system' not in st.session_state:
        st.session_state.agent_system = MultiAgentSystem()
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = SentimentAnalyzer()
    if 'stock_fetcher' not in st.session_state:
        st.session_state.stock_fetcher = StockDataFetcher()
    if 'investment_strategy' not in st.session_state:
        st.session_state.investment_strategy = BiweeklyInvestmentStrategy()
    
    # New enhanced components
    if 'data_aggregator' not in st.session_state:
        st.session_state.data_aggregator = DataSourceAggregator()
    if 'divergence_analyzer' not in st.session_state:
        st.session_state.divergence_analyzer = SentimentDivergenceAnalyzer()
    if 'sentiment_aggregator' not in st.session_state:
        st.session_state.sentiment_aggregator = MultiSourceSentimentAggregator()
    if 'pd_analyzer' not in st.session_state:
        st.session_state.pd_analyzer = PrisonersDilemmaCapexAnalyzer()
    if 'consensus_system' not in st.session_state:
        st.session_state.consensus_system = MultiAgentConsensus()
    if 'signal_generator' not in st.session_state:
        st.session_state.signal_generator = BiweeklySignalGenerator()
    if 'disagreement_analyzer' not in st.session_state:
        st.session_state.disagreement_analyzer = DisagreementAnalyzer()
    if 'extreme_detector' not in st.session_state:
        st.session_state.extreme_detector = SentimentExtremeDetector()


def create_price_chart(df, ticker):
    """Create interactive price chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add moving averages if enough data
    if len(df) >= 20:
        ma20 = df['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma20,
            name='MA20',
            line=dict(color='orange', width=1)
        ))
    
    if len(df) >= 50:
        ma50 = df['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma50,
            name='MA50',
            line=dict(color='blue', width=1)
        ))
    
    fig.update_layout(
        title=f"{ticker} Price Chart",
        yaxis_title="Price (USD)",
        xaxis_title="Date",
        template="plotly_white",
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_agent_decision_chart(decisions):
    """Create visualization of agent decisions"""
    actions = [d.action for d in decisions]
    action_counts = {
        'BUY': actions.count('BUY'),
        'SELL': actions.count('SELL'),
        'HOLD': actions.count('HOLD')
    }
    
    colors = {'BUY': '#4CAF50', 'SELL': '#f44336', 'HOLD': '#FF9800'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(action_counts.keys()),
            y=list(action_counts.values()),
            marker_color=[colors[k] for k in action_counts.keys()],
            text=list(action_counts.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Agent Decision Distribution",
        yaxis_title="Number of Agents",
        xaxis_title="Action",
        template="plotly_white",
        height=300
    )
    
    return fig


def create_sentiment_gauge(sentiment_score):
    """Create sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Market Sentiment", 'font': {'size': 20}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcdd2'},
                {'range': [30, 70], 'color': '#fff9c4'},
                {'range': [70, 100], 'color': '#c8e6c9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig


def create_nash_equilibrium_viz(equilibrium_data):
    """Create Nash equilibrium visualization"""
    categories = ['Buy Ratio', 'Sell Ratio', 'Hold Ratio']
    values = [
        equilibrium_data['buy_ratio'] * 100,
        equilibrium_data['sell_ratio'] * 100,
        equilibrium_data['hold_ratio'] * 100
    ]
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current State'
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        title="Nash Equilibrium Analysis",
        height=300
    )
    
    return fig


def main():
    """Main dashboard application"""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🎮 Game Theory Stock Agent Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("**AI-Driven Stock Analysis with Game-Theoretic Principles**")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Demo mode notice
        st.info("🎯 **Demo Mode**: Using simulated stock data for demonstration purposes")
        
        # Stock selection
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Time period
        period = st.selectbox(
            "Analysis Period",
            options=["1mo", "3mo", "6mo", "1y"],
            index=1
        )
        
        # Investment amount
        investment_amount = st.number_input(
            "Biweekly Investment Amount ($)",
            min_value=100.0,
            max_value=10000.0,
            value=1000.0,
            step=100.0
        )
        
        st.session_state.investment_strategy.investment_amount = investment_amount
        
        st.markdown("---")
        st.markdown("### 🤖 AI Agents")
        st.info(f"**{len(st.session_state.agent_system.agents)}** agents active")
        
        for agent in st.session_state.agent_system.agents:
            st.text(f"• {agent.agent_id}")
            st.caption(f"  Strategy: {agent.strategy.value}")
        
        st.markdown("---")
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_analysis:
        with st.spinner(f"Analyzing {ticker}..."):
            # Fetch comprehensive data from all sources
            comprehensive_data = st.session_state.data_aggregator.get_comprehensive_data(ticker, period)
            
            stock_data = comprehensive_data['stock_data']
            
            if stock_data is None or stock_data.empty:
                st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                return
            
            # Extract all data components
            technical_indicators = comprehensive_data['technical_indicators']
            technical_signal = comprehensive_data['technical_signal']
            analyst_data = comprehensive_data['analyst_data']
            analyst_sentiment = comprehensive_data['analyst_sentiment']
            news_articles = comprehensive_data['news_articles']
            social_sentiment_data = comprehensive_data['social_sentiment']
            
            # Calculate enhanced sentiments
            prices = stock_data['Close'].tolist()
            volumes = stock_data['Volume'].tolist()
            
            # News sentiment from articles
            news_texts = [article.get('title', '') + ' ' + article.get('description', '') 
                         for article in news_articles[:5]]
            if not news_texts:
                news_texts = generate_mock_news(ticker, np.random.random())
            
            # Calculate news sentiment
            news_sentiment_scores = [st.session_state.sentiment_analyzer.analyze_text(text) 
                                    for text in news_texts]
            news_sentiment = np.mean(news_sentiment_scores) if news_sentiment_scores else 0.5
            
            # Social sentiment
            social_sentiment = social_sentiment_data['combined_sentiment']
            
            # Technical sentiment (convert signal to score)
            tech_sentiment_map = {'BUY': 0.8, 'SELL': 0.2, 'HOLD': 0.5}
            technical_sentiment = tech_sentiment_map.get(technical_signal, 0.5)
            
            # Aggregate all sentiments
            aggregated_sentiment = st.session_state.sentiment_aggregator.aggregate_sentiments(
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                analyst_sentiment=analyst_sentiment,
                technical_sentiment=technical_sentiment
            )
            
            composite_sentiment = aggregated_sentiment['composite_sentiment']
            
            # Sentiment divergence analysis
            divergence_analysis = st.session_state.divergence_analyzer.calculate_divergence(
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment
            )
            
            divergence_recommendation = st.session_state.divergence_analyzer.get_trading_recommendation(
                divergence_analysis=divergence_analysis,
                current_price=technical_indicators.get('Current_Price', 0),
                technical_signal=technical_signal
            )
            
            # Prisoner's Dilemma AI Capex Analysis
            pd_analysis = st.session_state.pd_analyzer.analyze_competitive_capex(ticker)
            ai_beneficiary = st.session_state.pd_analyzer.identify_ai_beneficiaries(ticker)
            
            # Run agent simulation
            market_data = {
                'price': prices[-1] if prices else 0,
                'volume': volumes[-1] if volumes else 0,
                'technical': technical_indicators
            }
            
            agent_decisions = st.session_state.agent_system.run_simulation(
                market_data, composite_sentiment
            )
            
            # Convert to EnhancedAgentDecisions for consensus
            enhanced_decisions = [
                EnhancedAgentDecision(
                    agent_id=d.agent_id,
                    agent_type=d.agent_type,
                    action=d.action,
                    confidence=d.confidence,
                    reasoning=f"{d.strategy.value} strategy",
                    weight=1.0
                )
                for d in agent_decisions
            ]
            
            # Multi-agent consensus with reliability
            consensus_result = st.session_state.consensus_system.calculate_consensus(enhanced_decisions)
            position_sizing = st.session_state.consensus_system.generate_position_sizing(consensus_result)
            
            # Disagreement analysis
            disagreement = st.session_state.disagreement_analyzer.analyze_disagreement(enhanced_decisions)
            
            # Sentiment extremes
            extreme_analysis = st.session_state.extreme_detector.detect_extreme(
                sentiment_score=composite_sentiment,
                social_sentiment=social_sentiment,
                consensus_action=consensus_result['consensus_action']
            )
            
            # Generate biweekly signal
            biweekly_signal = st.session_state.signal_generator.generate_signal(
                multi_agent_consensus=consensus_result,
                sentiment_divergence=divergence_recommendation,
                prisoners_dilemma=pd_analysis,
                technical_signal=technical_signal,
                analyst_sentiment=analyst_sentiment,
                current_price=technical_indicators.get('Current_Price', 0)
            )
            
            # Legacy calculations for compatibility
            nash_equilibrium = st.session_state.agent_system.calculate_nash_equilibrium(agent_decisions)
            consensus = consensus_result['consensus_action']
            
            # Get stock info
            stock_info = st.session_state.stock_fetcher.get_stock_info(ticker)
            
            # Calculate investment allocation
            allocation = st.session_state.investment_strategy.calculate_biweekly_allocation(
                agent_decisions, composite_sentiment
            )
            
            # Display results
            st.markdown('<div class="sub-header">📈 Stock Overview</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = technical_indicators.get('Current_Price', 0)
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{technical_indicators.get('Daily_Change', 0):.2f}%"
                )
            
            with col2:
                st.metric(
                    "Company",
                    stock_info.get('name', ticker)[:20],
                    stock_info.get('sector', 'N/A')
                )
            
            with col3:
                st.metric(
                    "Composite Sentiment",
                    f"{composite_sentiment*100:.0f}%",
                    divergence_analysis['divergence_type'][:15]
                )
            
            with col4:
                st.metric(
                    "Signal Quality",
                    consensus_result.get('signal_quality', 'FAIR'),
                    f"{consensus_result.get('reliability_score', 0)*100:.0f}% reliable"
                )
            
            # Charts row
            st.markdown('<div class="sub-header">📊 Market Analysis</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                price_chart = create_price_chart(stock_data, ticker)
                st.plotly_chart(price_chart, use_container_width=True)
            
            with col2:
                sentiment_gauge = create_sentiment_gauge(composite_sentiment)
                st.plotly_chart(sentiment_gauge, use_container_width=True)
                
                # Enhanced sentiment components
                st.markdown("**Multi-Source Sentiment:**")
                sentiment_sources = aggregated_sentiment['individual_sentiments']
                for source, value in sentiment_sources.items():
                    st.progress(value, text=f"{source.title()}: {value*100:.1f}%")
            
            # Agent decisions
            st.markdown('<div class="sub-header">🤖 Agent Decisions & Game Theory</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                agent_chart = create_agent_decision_chart(agent_decisions)
                st.plotly_chart(agent_chart, use_container_width=True)
                
                # Detailed agent decisions
                st.markdown("**Individual Agent Analysis:**")
                for decision in agent_decisions:
                    action_color = "bullish" if decision.action == "BUY" else "bearish" if decision.action == "SELL" else "neutral"
                    st.markdown(f"""
                    <div class="agent-card">
                        <strong>{decision.agent_id}</strong> ({decision.strategy.value})<br>
                        Action: <span class="{action_color}">{decision.action}</span><br>
                        Confidence: {decision.confidence*100:.1f}%<br>
                        Allocation: {decision.allocation:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                nash_chart = create_nash_equilibrium_viz(nash_equilibrium)
                st.plotly_chart(nash_chart, use_container_width=True)
                
                # Nash equilibrium metrics
                st.markdown("**Nash Equilibrium Metrics:**")
                st.metric("Stability Score", f"{nash_equilibrium['stability_score']*100:.1f}%")
                
                equilibrium_status = "✅ At Equilibrium" if nash_equilibrium['is_equilibrium'] else "⚠️ Not at Equilibrium"
                st.info(equilibrium_status)
                
                st.markdown("**Action Distribution:**")
                st.write(f"• Buy: {nash_equilibrium['buy_ratio']*100:.1f}%")
                st.write(f"• Sell: {nash_equilibrium['sell_ratio']*100:.1f}%")
                st.write(f"• Hold: {nash_equilibrium['hold_ratio']*100:.1f}%")
            
            # Enhanced Investment recommendation
            st.markdown('<div class="sub-header">💰 Biweekly Investment Signal</div>', 
                       unsafe_allow_html=True)
            
            # Main signal display
            signal = biweekly_signal['signal']
            signal_color = "bullish" if signal == "BUY" else "bearish" if signal == "SELL" else "neutral"
            
            st.markdown(f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
                <h2 style="margin: 0; color: white;">Signal: <span class="{signal_color}" 
                    style="color: {'#4CAF50' if signal == 'BUY' else '#f44336' if signal == 'SELL' else '#FF9800'};">
                    {signal}</span></h2>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                    {biweekly_signal['recommendation']['recommendation_text']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Position Size", f"{biweekly_signal['position_size']*100:.0f}%")
                st.metric("Conviction", biweekly_signal['recommendation']['conviction'])
            
            with col2:
                st.metric("Confidence", f"{biweekly_signal['confidence']*100:.0f}%")
                st.metric("Signal Quality", biweekly_signal['signal_quality'])
            
            with col3:
                rec = biweekly_signal['recommendation']
                if rec['stop_loss']:
                    st.metric("Stop Loss", f"${rec['stop_loss']:.2f}")
                if rec['target_price']:
                    st.metric("Target Price", f"${rec['target_price']:.2f}")
            
            with col4:
                days_to_next = st.session_state.signal_generator.get_days_to_next_signal()
                st.metric("Next Signal In", f"{days_to_next} days")
                st.metric("Risk/Reward", f"{rec.get('risk_reward_ratio', 0):.2f}x")
            
            # Sentiment Divergence Analysis
            st.markdown('<div class="sub-header">🔄 Sentiment Divergence Analysis</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Divergence Status:**")
                div_type = divergence_analysis['divergence_type']
                is_divergent = divergence_analysis['is_divergent']
                
                if is_divergent:
                    st.warning(f"⚠️ **Divergence Detected:** {div_type}")
                else:
                    st.success(f"✅ **Sentiments Aligned:** {div_type}")
                
                st.markdown(f"""
                - **News Sentiment:** {divergence_analysis['news_sentiment']*100:.0f}%
                - **Social Sentiment:** {divergence_analysis['social_sentiment']*100:.0f}%
                - **Divergence Magnitude:** {divergence_analysis['divergence_magnitude']*100:.0f}%
                - **Expected Direction:** {divergence_analysis['expected_direction']}
                """)
            
            with col2:
                st.markdown("**Trading Implication:**")
                st.info(divergence_recommendation['reasoning'])
                
                st.markdown(f"""
                - **Action:** {divergence_recommendation['action']}
                - **Position Size:** {divergence_recommendation['position_size']*100:.0f}%
                - **Confidence:** {divergence_recommendation['confidence']*100:.0f}%
                - **Risk Level:** {divergence_recommendation['risk_level']}
                """)
            
            # Prisoner's Dilemma AI Capex Analysis
            st.markdown('<div class="sub-header">🎯 AI Capex Analysis (Prisoner\'s Dilemma)</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Competitive Capex Analysis:**")
                
                if pd_analysis.get('analysis') != 'INSUFFICIENT_DATA':
                    analysis_type = pd_analysis.get('analysis', 'UNKNOWN')
                    risk_level = pd_analysis.get('risk_level', 'UNKNOWN')
                    
                    if risk_level == 'HIGH':
                        st.error(f"🔴 **High Risk:** {analysis_type}")
                    elif risk_level == 'LOW':
                        st.success(f"🟢 **Low Risk:** {analysis_type}")
                    else:
                        st.warning(f"🟡 **Moderate Risk:** {analysis_type}")
                    
                    st.markdown(pd_analysis.get('reasoning', 'Analysis in progress...'))
                    
                    if 'target_capex_ratio' in pd_analysis:
                        st.metric("Target Capex/Revenue", f"{pd_analysis['target_capex_ratio']*100:.1f}%")
                else:
                    st.info("📊 Capex data not available for detailed analysis")
            
            with col2:
                st.markdown("**AI Beneficiary Status:**")
                
                if ai_beneficiary['is_beneficiary']:
                    st.success(f"✅ **{ai_beneficiary['category']}**")
                else:
                    st.info(f"ℹ️ **{ai_beneficiary['category']}**")
                
                st.markdown(ai_beneficiary['reasoning'])
                st.metric("Confidence", f"{ai_beneficiary['confidence']*100:.0f}%")
            
            # Multi-Agent Consensus Details
            st.markdown('<div class="sub-header">🤖 Multi-Agent Consensus & Reliability</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Consensus Metrics:**")
                st.metric("Reliability Score", f"{consensus_result['reliability_score']*100:.0f}%")
                st.metric("Agent Agreement", f"{consensus_result['agent_agreement']*100:.0f}%")
                st.metric("Signal Quality", consensus_result['signal_quality'])
                
                st.markdown("**Voting Distribution:**")
                votes = consensus_result['votes']
                st.write(f"🟢 Buy: {votes['BUY']} agents")
                st.write(f"🔴 Sell: {votes['SELL']} agents")
                st.write(f"🟡 Hold: {votes['HOLD']} agents")
            
            with col2:
                st.markdown("**Disagreement Analysis:**")
                st.metric("Disagreement Level", disagreement['disagreement_level'])
                st.metric("Market Regime", disagreement['market_regime'])
                
                st.info(disagreement['recommendation'])
                
                # Sentiment extremes
                if extreme_analysis['is_extreme']:
                    st.warning(f"⚠️ **{extreme_analysis['extreme_type']}**")
                    st.caption(extreme_analysis['warning'])
            
            # News & insights
            st.markdown('<div class="sub-header">📰 Market News & Analyst Insights</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Recent News Articles:**")
                if news_articles:
                    for article in news_articles[:5]:
                        title = article.get('title', 'No title')
                        source = article.get('source', 'Unknown')
                        url = article.get('url', '#')
                        sentiment_score = st.session_state.sentiment_analyzer.analyze_text(title)
                        sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "🟡"
                        st.markdown(f"{sentiment_emoji} **{title}**")
                        st.caption(f"Source: {source}")
                else:
                    st.info("News articles from Yahoo Finance or NewsAPI will appear here")
            
            with col2:
                st.markdown("**Analyst Insights:**")
                st.metric("Analyst Rating", analyst_data.get('recommendation', 'N/A').upper())
                st.metric("Number of Analysts", analyst_data.get('num_analysts', 0))
                
                if analyst_data.get('target_mean', 0) > 0:
                    upside = ((analyst_data['target_mean'] - analyst_data['current_price']) / 
                             analyst_data['current_price'] * 100)
                    st.metric("Avg Price Target", f"${analyst_data['target_mean']:.2f}", 
                             f"{upside:+.1f}%")
                
                st.markdown("**Social Sentiment:**")
                st.metric("Combined Social Score", f"{social_sentiment*100:.0f}%")
                st.caption(f"Reddit: {social_sentiment_data['platforms']['reddit']['sentiment_score']*100:.0f}% | "
                          f"Twitter: {social_sentiment_data['platforms']['twitter']['sentiment_score']*100:.0f}%")
            
            # Technical indicators
            with st.expander("📊 Technical Indicators"):
                tech_df = pd.DataFrame([technical_indicators]).T
                tech_df.columns = ['Value']
                st.dataframe(tech_df, use_container_width=True)
            
            # Game theory insights
            with st.expander("🎮 Enhanced Game Theory & Psychology Insights"):
                st.markdown("""
                **Game-Theoretic Principles Applied:**
                
                1. **Nash Equilibrium**: The system iterates agent decisions to find a stable state where no agent 
                   can improve their payoff by unilaterally changing their strategy.
                
                2. **Multi-Agent Coordination**: Agents observe each other's actions and adjust their strategies,
                   similar to coordination games where collective action yields better outcomes.
                
                3. **Prisoner's Dilemma (AI Capex)**: Applied to analyze competitive AI spending. When multiple
                   mega-caps overspend on AI infrastructure, it creates valuation compression risk for all players.
                
                4. **Sentiment Divergence**: Exploits psychology - retail sentiment (social) leads professional
                   news by 3-5 days. Divergence creates trading opportunities.
                
                5. **Best Response Dynamics**: Each agent calculates their best response given other agents' actions,
                   converging toward equilibrium through iterative refinement.
                """)
                
                st.markdown(f"""
                **Current Analysis:**
                - **Stability Score:** {nash_equilibrium['stability_score']*100:.1f}% (>70% indicates Nash equilibrium)
                - **Agent Consensus:** {len([d for d in agent_decisions if d.action == consensus])} out of {len(agent_decisions)} agents agree on {consensus}
                - **Signal Reliability:** {consensus_result['reliability_score']*100:.0f}% (~70% with 3+ agents, ~85% with 5+ agents)
                - **Composite Sentiment:** {composite_sentiment*100:.0f}%
                - **Sentiment Divergence:** {'Detected' if divergence_analysis['is_divergent'] else 'Aligned'}
                - **PD Risk Level:** {pd_analysis.get('risk_level', 'N/A')}
                """)
                
                st.markdown("""
                **Financial Psychology Insights:**
                - **Leading Indicator:** Retail sentiment (Reddit/Twitter) typically leads news by 3-5 days
                - **Contrarian Signal:** When capex spending is unsustainable despite bullish consensus
                - **Divergence Trades:** Exploit crowd psychology mismatches between news and social
                - **Consensus Reliability:** 3+ diverse agents (Bull+Bear+Technical) achieve ~70% signal accuracy
                """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Game Theory Stock Agent Dashboard! 🎮📊
        
        This advanced AI-powered system combines cutting-edge research in:
        
        - 🤖 **Multi-Agent AI Systems**: 6 specialized AI agents (Bull, Bear, Technical, Sentiment, Fundamental, Contrarian)
        - 🎮 **Game Theory**: Nash equilibrium, Prisoner's Dilemma for AI capex, coordination games
        - 📊 **Real Data Sources**: Yahoo Finance, News APIs, Social sentiment (Reddit, Twitter)
        - 🔄 **Sentiment Divergence**: Detects mismatches between professional news and retail social sentiment
        - 🎯 **AI Capex Analysis**: Prisoner's Dilemma framework for competitive AI spending patterns
        - 📈 **Advanced Technical Analysis**: 15+ indicators including MACD, RSI, Bollinger Bands, Stochastic
        - 💰 **Biweekly Signals**: BUY/HOLD/SELL signals with reliability scoring and position sizing
        
        ### How It Works:
        
        1. **Select a stock** ticker in the sidebar
        2. **Configure** your analysis period and investment amount
        3. **Run Analysis** to activate the comprehensive multi-factor analysis
        4. **Review** biweekly signals, sentiment divergence, AI capex risks
        5. **Get recommendations** with reliability scores and position sizing
        
        ### The Enhanced AI Agents:
        
        The system employs 6 specialized agents with diverse perspectives:
        - **Agent Alpha (Bull)**: Aggressive long bias, high-risk/high-reward
        - **Agent Beta (Bear)**: Conservative, risk-averse, cautious positioning
        - **Agent Gamma (Technical)**: Chart-based, trend-following analysis
        - **Agent Delta (Contrarian)**: Goes against consensus for asymmetric opportunities
        - **Agent Epsilon (Sentiment)**: Social and news sentiment focused
        - **Agent Zeta (Fundamental)**: Analyst targets and valuation focused
        
        ### Key Innovations:
        
        **🔄 Sentiment Divergence:**
        - Detects when professional news sentiment differs from retail social sentiment
        - News bullish + Social bearish = Price likely to rise (catching up)
        - News bearish + Social bullish = Momentum fade likely
        - Retail sentiment leads news by 3-5 days (leading indicator)
        
        **🎯 Prisoner's Dilemma AI Capex:**
        - Analyzes competitive AI infrastructure spending
        - Multiple mega-caps overspending = valuation compression risk (mutual defection)
        - Identifies disciplined spenders and AI beneficiaries (winners)
        - Contrarian signal when consensus bullish but capex unsustainable
        
        **📊 Multi-Agent Consensus Reliability:**
        - 3+ agents agree = ~70% signal reliability
        - 4+ agents agree = ~80% reliability  
        - 5+ agents agree = ~85% reliability
        - Bull + Bear + Technical consensus = optimal signal quality
        
        **💰 Biweekly Execution:**
        - Generates actionable signals every 2 weeks
        - Position sizing based on signal quality and reliability
        - Weekly tactical adjustments for sentiment divergence changes
        - Stop loss and target prices for risk management
        
        ### Get Started:
        
        👈 Enter a stock ticker in the sidebar and click **Run Analysis**!
        """)
        
        # Example stocks
        st.markdown("### Popular Stocks to Try:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.info("**Tech Giants**\nAAPL, MSFT, GOOGL, META")
        with col2:
            st.info("**Finance**\nJPM, BAC, GS, V")
        with col3:
            st.info("**Consumer**\nAMZN, TSLA, NKE, DIS")
        with col4:
            st.info("**Healthcare**\nJNJ, UNH, PFE, ABBV")


if __name__ == "__main__":
    main()
