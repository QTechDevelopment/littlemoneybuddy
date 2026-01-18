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


# Page configuration - Terminal Style
st.set_page_config(
    page_title="GTSA Terminal // Game Theory Stock Agent",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Retro Monochrome Terminal CSS
st.markdown("""
<style>
    /* Import terminal-style font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    /* Global monochrome styling */
    .stApp {
        background-color: black !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Override text colors to white */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        font-family: 'JetBrains Mono', monospace !important;
        color: white !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 2rem;
        color: white;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 2px;
    }
    
    .main-header::before {
        content: "$ ";
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: white;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace !important;
        border-bottom: 1px dashed white;
        padding-bottom: 0.5rem;
    }
    
    .sub-header::before {
        content: "> ";
    }
    
    /* Cards */
    .metric-card, .agent-card, .terminal-box {
        background-color: black;
        padding: 1rem;
        border: 1px solid white;
        margin: 0.5rem 0;
    }
    
    .agent-card::before {
        content: "┌──[AGENT]";
        position: absolute;
        top: -10px;
        left: 10px;
        background: black;
        padding: 0 5px;
        color: white;
        font-size: 0.7rem;
    }
    
    /* Status colors - all white for monochrome */
    .bullish, .bearish, .neutral {
        color: white !important;
        font-weight: bold;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: black !important;
        border-right: 1px solid white !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background-color: #333333 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        color: white !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: white !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: black;
    }
    
    ::-webkit-scrollbar-thumb {
        background: white;
    }
    
    /* ASCII art header */
    .ascii-header {
        font-family: 'JetBrains Mono', monospace !important;
        color: white;
        font-size: 0.6rem;
        line-height: 1.1;
        white-space: pre;
        margin-bottom: 1rem;
    }
    
    /* System message */
    .sys-msg {
        color: white;
        font-size: 0.9rem;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .sys-msg::before {
        content: "[SYS] ";
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
    """Create interactive price chart with terminal-style theme"""
    fig = go.Figure()
    
    # Candlestick chart with terminal colors
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#ffffff',
        decreasing_line_color='#555555'
    ))
    
    # Add moving averages if enough data
    if len(df) >= 20:
        ma20 = df['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma20,
            name='MA20',
            line=dict(color='#aaaaaa', width=1, dash='dot')
        ))
    
    if len(df) >= 50:
        ma50 = df['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma50,
            name='MA50',
            line=dict(color='#ffffff', width=1)
        ))
    
    fig.update_layout(
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white', family='JetBrains Mono'),
        xaxis=dict(showgrid=False, color='white'),
        yaxis=dict(showgrid=True, gridcolor='#333333', color='white'),
        margin=dict(l=10, r=10, t=10, b=10)
    )
    return fig


def create_agent_decision_chart(decisions):
    """Create visualization of agent decisions with terminal style"""
    actions = [d.action for d in decisions]
    action_counts = {
        'BUY': actions.count('BUY'),
        'SELL': actions.count('SELL'),
        'HOLD': actions.count('HOLD')
    }
    
    # Terminal-style colors
    colors = {'BUY': '#00ff41', 'SELL': '#ff3e3e', 'HOLD': '#ffb000'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(action_counts.keys()),
            y=list(action_counts.values()),
            marker_color=[colors[k] for k in action_counts.keys()],
            text=list(action_counts.values()),
            textposition='auto',
            textfont=dict(color='#0a0e14', family='JetBrains Mono, monospace', size=16)
        )
    ])
    
    fig.update_layout(
        title=dict(text=">> AGENT_DECISION_MATRIX", font=dict(color='#00ff41', family='JetBrains Mono, monospace')),
        yaxis_title="AGENT_COUNT",
        xaxis_title="ACTION",
        template="plotly_dark",
        height=300,
        paper_bgcolor='rgba(10, 14, 20, 0.9)',
        plot_bgcolor='rgba(10, 14, 20, 0.9)',
        font=dict(family='JetBrains Mono, monospace', color='#00d4ff'),
        xaxis=dict(gridcolor='rgba(0, 255, 65, 0.1)'),
        yaxis=dict(gridcolor='rgba(0, 255, 65, 0.1)')
    )
    
    return fig


def create_sentiment_gauge(sentiment_score):
    """Create sentiment gauge chart with terminal styling"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "MARKET_SENTIMENT", 'font': {'size': 16, 'color': '#00ff41', 'family': 'JetBrains Mono, monospace'}},
        delta={'reference': 50, 'increasing': {'color': "#00ff41"}, 'decreasing': {'color': "#ff3e3e"}},
        number={'font': {'color': '#00ff41', 'family': 'JetBrains Mono, monospace'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#00d4ff", 'tickfont': {'color': '#00d4ff'}},
            'bar': {'color': "#00ff41"},
            'bgcolor': "#0a0e14",
            'borderwidth': 2,
            'bordercolor': "#00d4ff",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(255, 62, 62, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(255, 176, 0, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(0, 255, 65, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#00d4ff", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(10, 14, 20, 0.9)',
        font=dict(family='JetBrains Mono, monospace', color='#00d4ff')
    )
    return fig


def create_nash_equilibrium_viz(equilibrium_data):
    """Create Nash equilibrium visualization with terminal styling"""
    categories = ['BUY_RATIO', 'SELL_RATIO', 'HOLD_RATIO']
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
            name='Current State',
            line=dict(color='#00ff41'),
            fillcolor='rgba(0, 255, 65, 0.2)'
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='#00d4ff', family='JetBrains Mono, monospace'),
                gridcolor='rgba(0, 212, 255, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(color='#00ff41', family='JetBrains Mono, monospace'),
                gridcolor='rgba(0, 212, 255, 0.2)'
            ),
            bgcolor='rgba(10, 14, 20, 0.9)'
        ),
        showlegend=False,
        title=dict(text=">> NASH_EQUILIBRIUM", font=dict(color='#00ff41', family='JetBrains Mono, monospace')),
        height=300,
        paper_bgcolor='rgba(10, 14, 20, 0.9)',
        font=dict(family='JetBrains Mono, monospace', color='#00d4ff')
    )
    
    return fig


def main():
    """Main dashboard application"""
    initialize_session_state()
    
    # ASCII Art Header
    st.markdown("""
    <div class="ascii-header">
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║   ██████╗ ████████╗███████╗ █████╗     ████████╗███████╗██████╗ ███╗   ███╗  ║
    ║  ██╔════╝ ╚══██╔══╝██╔════╝██╔══██╗    ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║  ║
    ║  ██║  ███╗   ██║   ███████╗███████║       ██║   █████╗  ██████╔╝██╔████╔██║  ║
    ║  ██║   ██║   ██║   ╚════██║██╔══██║       ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║  ║
    ║  ╚██████╔╝   ██║   ███████║██║  ██║       ██║   ███████╗██║  ██║██║ ╚═╝ ██║  ║
    ║   ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    </div>
    """, unsafe_allow_html=True)
    
    # Header with terminal prompt style
    st.markdown('<div class="main-header glitch-text">GAME_THEORY_STOCK_AGENT<span class="cursor"></span></div>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sys-msg">AI-Driven Stock Analysis :: Game-Theoretic Principles :: Multi-Agent System v2.0</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration - Terminal Style
    with st.sidebar:
        st.markdown("### ⚡ SYSTEM CONFIG")
        st.markdown("---")
        
        # Demo mode notice
        st.markdown("""
        <div class="terminal-box">
            <span class="sys-msg">DEMO_MODE: Active</span><br>
            <small style="color: #888;">Using simulated stock data</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📡 INPUT PARAMETERS")
        
        # Stock selection
        ticker = st.text_input(
            ">> TICKER_SYMBOL",
            value="AAPL",
            help="Enter stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Time period
        period = st.selectbox(
            ">> ANALYSIS_PERIOD",
            options=["1mo", "3mo", "6mo", "1y"],
            index=1
        )
        
        # Investment amount
        investment_amount = st.number_input(
            ">> INVESTMENT_AMT ($)",
            min_value=100.0,
            max_value=10000.0,
            value=1000.0,
            step=100.0
        )
        
        st.session_state.investment_strategy.investment_amount = investment_amount
        
        st.markdown("---")
        st.markdown("### 🤖 ACTIVE AGENTS")
        st.markdown(f"""
        <div class="terminal-box">
            <span style="color: #00ff41;">STATUS:</span> ONLINE<br>
            <span style="color: #00d4ff;">AGENTS:</span> {len(st.session_state.agent_system.agents)} deployed
        </div>
        """, unsafe_allow_html=True)
        
        for agent in st.session_state.agent_system.agents:
            strategy_color = "#00ff41" if agent.strategy.value == "aggressive" else "#ffb000" if agent.strategy.value == "conservative" else "#00d4ff"
            st.markdown(f"""
            <small style="color: #888;">├─ <span style="color: {strategy_color};">{agent.agent_id}</span></small><br>
            <small style="color: #666;">│  └─ {agent.strategy.value.upper()}</small>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        run_analysis = st.button("⚡ EXECUTE ANALYSIS", type="primary", use_container_width=True)
    
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
            st.markdown('<div class="sub-header">📡 STOCK_OVERVIEW</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = technical_indicators.get('Current_Price', 0)
                st.metric(
                    "CURRENT_PRICE",
                    f"${current_price:.2f}",
                    f"{technical_indicators.get('Daily_Change', 0):.2f}%"
                )
            
            with col2:
                st.metric(
                    "ENTITY",
                    stock_info.get('name', ticker)[:20],
                    stock_info.get('sector', 'N/A')
                )
            
            with col3:
                sentiment_class = sentiment_signal.lower()
                st.metric(
                    "SENTIMENT",
                    sentiment_signal,
                    f"{composite_sentiment*100:.1f}%"
                )
            
            with col4:
                st.metric(
                    "CONSENSUS",
                    consensus,
                    f"{len([d for d in agent_decisions if d.action == consensus])}/{len(agent_decisions)} agents"
                )
            
            # Charts row
            st.markdown('<div class="sub-header">📊 MARKET_ANALYSIS</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                price_chart = create_price_chart(stock_data, ticker)
                st.plotly_chart(price_chart, use_container_width=True)
            
            with col2:
                sentiment_gauge = create_sentiment_gauge(composite_sentiment)
                st.plotly_chart(sentiment_gauge, use_container_width=True)
                
                # Sentiment components
                st.markdown("**>> SENTIMENT_COMPONENTS:**")
                for component, value in sentiment_result['components'].items():
                    st.progress(value, text=f"{component.upper()}: {value*100:.1f}%")
            
            # Agent decisions
            st.markdown('<div class="sub-header">🤖 AGENT_MATRIX // GAME_THEORY</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                agent_chart = create_agent_decision_chart(agent_decisions)
                st.plotly_chart(agent_chart, use_container_width=True)
                
                # Detailed agent decisions
                st.markdown("**>> AGENT_ANALYSIS:**")
                for decision in agent_decisions:
                    action_color = "bullish" if decision.action == "BUY" else "bearish" if decision.action == "SELL" else "neutral"
                    st.markdown(f"""
                    <div class="agent-card">
                        <strong style="color: #00d4ff;">{decision.agent_id}</strong> <span style="color: #888;">// {decision.strategy.value.upper()}</span><br>
                        <span style="color: #888;">ACTION:</span> <span class="{action_color}">{decision.action}</span><br>
                        <span style="color: #888;">CONF:</span> <span style="color: #00d4ff;">{decision.confidence*100:.1f}%</span><br>
                        <span style="color: #888;">ALLOC:</span> <span style="color: #ffb000;">{decision.allocation:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                nash_chart = create_nash_equilibrium_viz(nash_equilibrium)
                st.plotly_chart(nash_chart, use_container_width=True)
                
                # Nash equilibrium metrics
                st.markdown("**>> NASH_METRICS:**")
                st.metric("STABILITY", f"{nash_equilibrium['stability_score']*100:.1f}%")
                
                equilibrium_status = "✅ EQUILIBRIUM_REACHED" if nash_equilibrium['is_equilibrium'] else "⚠️ EQUILIBRIUM_PENDING"
                st.markdown(f"""
                <div class="terminal-box">
                    <span style="color: {'#00ff41' if nash_equilibrium['is_equilibrium'] else '#ffb000'};">{equilibrium_status}</span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**>> ACTION_DIST:**")
                st.markdown(f"""
                <div class="terminal-box">
                    <span style="color: #00ff41;">├─ BUY:</span> {nash_equilibrium['buy_ratio']*100:.1f}%<br>
                    <span style="color: #ff3e3e;">├─ SELL:</span> {nash_equilibrium['sell_ratio']*100:.1f}%<br>
                    <span style="color: #ffb000;">└─ HOLD:</span> {nash_equilibrium['hold_ratio']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            
            # Investment recommendation
            st.markdown('<div class="sub-header">💰 INVESTMENT_RECOMMENDATION</div>', 
                       unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                action_color = "bullish" if allocation['action'] == "BUY" else "bearish" if allocation['action'] == "SELL" else "neutral"
                st.markdown(f"**>> RECOMMENDED_ACTION:** <span class='{action_color}'>{allocation['action']}</span>", 
                           unsafe_allow_html=True)
                st.metric("ALLOCATION_AMT", f"${allocation['amount']:.2f}")
            
            with col2:
                st.metric("CONFIDENCE", f"{allocation['confidence']*100:.1f}%")
                next_investment_days = st.session_state.investment_strategy.days_until_next_investment()
                st.metric("NEXT_INVEST", f"{next_investment_days} days")
            
            with col3:
                st.markdown("**>> AGENT_VOTES:**")
                st.markdown(f"""
                <div class="terminal-box">
                    <span class="bullish">├─ BUY:</span> {allocation['agent_consensus']['buy']}<br>
                    <span class="bearish">├─ SELL:</span> {allocation['agent_consensus']['sell']}<br>
                    <span class="neutral">└─ HOLD:</span> {allocation['agent_consensus']['hold']}
                </div>
                """, unsafe_allow_html=True)
            
            # News & insights
            st.markdown('<div class="sub-header">📰 NEWS_FEED</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("""
            <div class="terminal-box">
                <span class="sys-msg">SIMULATED_DATA: Headlines from demo engine</span>
            </div>
            """, unsafe_allow_html=True)
            for i, news in enumerate(news_texts, 1):
                sentiment_score = st.session_state.sentiment_analyzer.analyze_text(news)
                sentiment_color = "#00ff41" if sentiment_score > 0.6 else "#ff3e3e" if sentiment_score < 0.4 else "#ffb000"
                st.markdown(f"""
                <div style="padding: 0.5rem 0; border-bottom: 1px solid rgba(0, 212, 255, 0.2);">
                    <span style="color: {sentiment_color};">▸</span> <span style="color: #c0c0c0;">{news}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Technical indicators
            with st.expander("📊 TECHNICAL_INDICATORS"):
                tech_df = pd.DataFrame([technical_indicators]).T
                tech_df.columns = ['Value']
                st.dataframe(tech_df, use_container_width=True)
            
            # Game theory insights
            with st.expander("🎮 GAME_THEORY_INSIGHTS"):
                st.markdown("""
                **>> APPLIED_PRINCIPLES:**
                
                1. **NASH_EQUILIBRIUM**: The system iterates agent decisions to find a stable state where no agent 
                   can improve their payoff by unilaterally changing their strategy.
                
                2. **MULTI_AGENT_COORD**: Agents observe each other's actions and adjust their strategies,
                   similar to coordination games where collective action yields better outcomes.
                
                3. **PRISONERS_DILEMMA**: Contrarian agents can benefit from going against the consensus,
                   representing the tension between individual and collective rationality.
                
                4. **BEST_RESPONSE_DYNAMICS**: Each agent calculates their best response given other agents' actions,
                   converging toward equilibrium through iterative refinement.
                """)
                
                st.markdown(f"""
                **>> CURRENT_ANALYSIS:**
                ```
                STABILITY_SCORE: {nash_equilibrium['stability_score']*100:.1f}% (>70% = EQUILIBRIUM)
                AGENT_CONSENSUS: {consensus} [{len([d for d in agent_decisions if d.action == consensus])}/{len(agent_decisions)} agents]
                MARKET_SENTIMENT: {sentiment_signal} ({composite_sentiment*100:.1f}%)
                ```
                """)
    
    else:
        # Welcome screen - Terminal Style
        st.markdown("""
        <div class="terminal-box">
            <span style="color: #00ff41;">$ ./init_system.sh</span><br>
            <span style="color: #888;">[INFO] Loading GTSA Terminal v2.0...</span><br>
            <span style="color: #888;">[INFO] Initializing multi-agent system...</span><br>
            <span style="color: #00ff41;">[OK] System ready.</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sub-header">📡 SYSTEM_INFO</div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="terminal-box">
        <span style="color: #00d4ff;">This advanced AI-powered terminal combines cutting-edge research in:</span><br><br>
        <span style="color: #00ff41;">├─</span> <span style="color: #c0c0c0;">🤖 MULTI_AGENT_AI</span> <span style="color: #888;">// Multiple specialized AI agents with different strategies</span><br>
        <span style="color: #00ff41;">├─</span> <span style="color: #c0c0c0;">🎮 GAME_THEORY</span> <span style="color: #888;">// Nash equilibrium, best response dynamics, coordination games</span><br>
        <span style="color: #00ff41;">├─</span> <span style="color: #c0c0c0;">📊 SENTIMENT_ANALYSIS</span> <span style="color: #888;">// NLP-based market sentiment from multiple sources</span><br>
        <span style="color: #00ff41;">├─</span> <span style="color: #c0c0c0;">📈 TECHNICAL_ANALYSIS</span> <span style="color: #888;">// Price momentum, volume trends, indicators</span><br>
        <span style="color: #00ff41;">└─</span> <span style="color: #c0c0c0;">💰 BIWEEKLY_STRATEGY</span> <span style="color: #888;">// Optimized allocation for regular investing</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sub-header">⚡ USAGE</div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="terminal-box">
        <span style="color: #ffb000;">$ gtsa --help</span><br><br>
        <span style="color: #c0c0c0;">COMMANDS:</span><br>
        <span style="color: #00ff41;">  1.</span> <span style="color: #c0c0c0;">Enter TICKER_SYMBOL in sidebar</span> <span style="color: #888;">(e.g., AAPL, GOOGL, MSFT)</span><br>
        <span style="color: #00ff41;">  2.</span> <span style="color: #c0c0c0;">Configure ANALYSIS_PERIOD</span> <span style="color: #888;">(1mo, 3mo, 6mo, 1y)</span><br>
        <span style="color: #00ff41;">  3.</span> <span style="color: #c0c0c0;">Set INVESTMENT_AMOUNT</span> <span style="color: #888;">($100 - $10,000)</span><br>
        <span style="color: #00ff41;">  4.</span> <span style="color: #c0c0c0;">Execute ANALYSIS</span> <span style="color: #888;">(⚡ EXECUTE_ANALYSIS button)</span><br>
        <span style="color: #00ff41;">  5.</span> <span style="color: #c0c0c0;">Review OUTPUT</span> <span style="color: #888;">(charts, metrics, recommendations)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sub-header">🤖 DEPLOYED_AGENTS</div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="terminal-box">
        <span style="color: #c0c0c0;">The system employs 5 specialized agents:</span><br><br>
        <span style="color: #00ff41;">Agent_Alpha</span> <span style="color: #888;">// AGGRESSIVE - High-risk, high-reward approach</span><br>
        <span style="color: #ffb000;">Agent_Beta</span> <span style="color: #888;">// CONSERVATIVE - Risk-averse, stability-focused</span><br>
        <span style="color: #00d4ff;">Agent_Gamma</span> <span style="color: #888;">// BALANCED - Moderate risk tolerance</span><br>
        <span style="color: #ff3e3e;">Agent_Delta</span> <span style="color: #888;">// CONTRARIAN - Goes against market consensus</span><br>
        <span style="color: #00d4ff;">Agent_Epsilon</span> <span style="color: #888;">// BALANCED - Secondary balanced perspective</span><br><br>
        <span style="color: #888;">Each agent analyzes: sentiment, technicals, game payoffs, agent interactions</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sub-header">🎯 QUICK_START</div>
        """, unsafe_allow_html=True)
        
        # Example stocks
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="terminal-box">
                <span style="color: #00ff41;">TECH</span><br>
                <span style="color: #888;">AAPL MSFT<br>GOOGL META</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="terminal-box">
                <span style="color: #00d4ff;">FINANCE</span><br>
                <span style="color: #888;">JPM BAC<br>GS V</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="terminal-box">
                <span style="color: #ffb000;">CONSUMER</span><br>
                <span style="color: #888;">AMZN TSLA<br>NKE DIS</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="terminal-box">
                <span style="color: #ff3e3e;">HEALTH</span><br>
                <span style="color: #888;">JNJ UNH<br>PFE ABBV</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 2rem; text-align: center; color: #888; font-size: 0.8rem;">
            <span style="color: #00ff41;">◄</span> Enter a ticker in the sidebar and click EXECUTE_ANALYSIS <span style="color: #00ff41;">►</span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
