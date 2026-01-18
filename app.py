"""
Game Theory Stock Agent Dashboard
Interactive dashboard for AI-driven stock analysis using game-theoretic principles
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
            # Fetch stock data
            stock_data = st.session_state.stock_fetcher.get_stock_data(ticker, period)
            
            if stock_data is None or stock_data.empty:
                st.error(f"❌ Could not fetch data for {ticker}. Please check the ticker symbol.")
                return
            
            stock_info = st.session_state.stock_fetcher.get_stock_info(ticker)
            technical_indicators = st.session_state.stock_fetcher.calculate_technical_indicators(stock_data)
            
            # Generate sentiment analysis
            prices = stock_data['Close'].tolist()
            volumes = stock_data['Volume'].tolist()
            news_texts = generate_mock_news(ticker, np.random.random())
            
            sentiment_result = st.session_state.sentiment_analyzer.calculate_composite_sentiment(
                news_texts=news_texts,
                prices=prices,
                volumes=volumes
            )
            
            composite_sentiment = sentiment_result['composite']
            sentiment_signal = st.session_state.sentiment_analyzer.get_sentiment_signal(composite_sentiment)
            
            # Run agent simulation
            market_data = {
                'price': prices[-1] if prices else 0,
                'volume': volumes[-1] if volumes else 0,
                'technical': technical_indicators
            }
            
            agent_decisions = st.session_state.agent_system.run_simulation(
                market_data, composite_sentiment
            )
            
            # Calculate Nash equilibrium
            nash_equilibrium = st.session_state.agent_system.calculate_nash_equilibrium(agent_decisions)
            consensus = st.session_state.agent_system.get_consensus_decision(agent_decisions)
            
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
                sentiment_class = sentiment_signal.lower()
                st.metric(
                    "Sentiment",
                    sentiment_signal,
                    f"{composite_sentiment*100:.1f}%"
                )
            
            with col4:
                st.metric(
                    "Agent Consensus",
                    consensus,
                    f"{len([d for d in agent_decisions if d.action == consensus])}/{len(agent_decisions)} agents"
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
                
                # Sentiment components
                st.markdown("**Sentiment Components:**")
                for component, value in sentiment_result['components'].items():
                    st.progress(value, text=f"{component.title()}: {value*100:.1f}%")
            
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
            
            # Investment recommendation
            st.markdown('<div class="sub-header">💰 Biweekly Investment Recommendation</div>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                action_color = "bullish" if allocation['action'] == "BUY" else "bearish" if allocation['action'] == "SELL" else "neutral"
                st.markdown(f"**Recommended Action:** <span class='{action_color}'>{allocation['action']}</span>", 
                           unsafe_allow_html=True)
                st.metric("Allocation Amount", f"${allocation['amount']:.2f}")
            
            with col2:
                st.metric("Confidence Level", f"{allocation['confidence']*100:.1f}%")
                next_investment_days = st.session_state.investment_strategy.days_until_next_investment()
                st.metric("Next Investment In", f"{next_investment_days} days")
            
            with col3:
                st.markdown("**Agent Votes:**")
                st.write(f"🟢 Buy: {allocation['agent_consensus']['buy']}")
                st.write(f"🔴 Sell: {allocation['agent_consensus']['sell']}")
                st.write(f"🟡 Hold: {allocation['agent_consensus']['hold']}")
            
            # News & insights
            st.markdown('<div class="sub-header">📰 Market News & Insights</div>', 
                       unsafe_allow_html=True)
            
            st.info("**Simulated News Headlines** (In production, these would be real-time news from APIs)")
            for i, news in enumerate(news_texts, 1):
                sentiment_score = st.session_state.sentiment_analyzer.analyze_text(news)
                sentiment_emoji = "🟢" if sentiment_score > 0.6 else "🔴" if sentiment_score < 0.4 else "🟡"
                st.markdown(f"{sentiment_emoji} {news}")
            
            # Technical indicators
            with st.expander("📊 Technical Indicators"):
                tech_df = pd.DataFrame([technical_indicators]).T
                tech_df.columns = ['Value']
                st.dataframe(tech_df, use_container_width=True)
            
            # Game theory insights
            with st.expander("🎮 Game Theory Insights"):
                st.markdown("""
                **Game-Theoretic Principles Applied:**
                
                1. **Nash Equilibrium**: The system iterates agent decisions to find a stable state where no agent 
                   can improve their payoff by unilaterally changing their strategy.
                
                2. **Multi-Agent Coordination**: Agents observe each other's actions and adjust their strategies,
                   similar to coordination games where collective action yields better outcomes.
                
                3. **Prisoner's Dilemma**: Contrarian agents can benefit from going against the consensus,
                   representing the tension between individual and collective rationality.
                
                4. **Best Response Dynamics**: Each agent calculates their best response given other agents' actions,
                   converging toward equilibrium through iterative refinement.
                """)
                
                st.markdown(f"""
                **Current Analysis:**
                - Stability Score: {nash_equilibrium['stability_score']*100:.1f}% 
                  (>70% indicates Nash equilibrium)
                - Agent consensus on {consensus}: {len([d for d in agent_decisions if d.action == consensus])} out of {len(agent_decisions)} agents
                - Market sentiment: {sentiment_signal} ({composite_sentiment*100:.1f}%)
                """)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Game Theory Stock Agent Dashboard! 🎮📊
        
        This advanced AI-powered system combines cutting-edge research in:
        
        - 🤖 **Multi-Agent AI Systems**: Multiple specialized AI agents with different strategies
        - 🎮 **Game Theory**: Nash equilibrium, best response dynamics, and coordination games
        - 😊 **Sentiment Analysis**: NLP-based analysis of market sentiment from multiple sources
        - 📈 **Technical Analysis**: Price momentum, volume trends, and technical indicators
        - 💰 **Biweekly Investment Strategy**: Optimized allocation for regular investing
        
        ### How It Works:
        
        1. **Select a stock** ticker in the sidebar
        2. **Configure** your analysis period and investment amount
        3. **Run Analysis** to activate the AI agents
        4. **Review** agent decisions, sentiment analysis, and game-theoretic insights
        5. **Get recommendations** for your biweekly investment strategy
        
        ### The AI Agents:
        
        The system employs 5 specialized agents with different strategies:
        - **Agent Alpha** (Aggressive): High-risk, high-reward approach
        - **Agent Beta** (Conservative): Risk-averse, stability-focused
        - **Agent Gamma** (Balanced): Moderate risk tolerance
        - **Agent Delta** (Contrarian): Goes against market consensus
        - **Agent Epsilon** (Balanced): Secondary balanced perspective
        
        Each agent analyzes the market independently and makes decisions based on:
        - Market sentiment (from news, price action, volume)
        - Technical indicators (moving averages, RSI, volatility)
        - Game-theoretic payoffs (Nash equilibrium, coordination)
        - Other agents' decisions (multi-agent interaction)
        
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
