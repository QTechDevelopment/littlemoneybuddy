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
from io import BytesIO
from collections import Counter

from game_theory_agent import (
    MultiAgentSystem, AgentStrategy, GameTheoryAgent
)
from sentiment_analyzer import SentimentAnalyzer, generate_mock_news
from stock_data import StockDataFetcher, BiweeklyInvestmentStrategy


def parse_excel_tickers(uploaded_file) -> list:
    """
    Parse Excel file to extract stock tickers.
    Looks for columns named 'ticker', 'symbol', 'stock', or the first column.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        List of unique ticker symbols
    """
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        if df.empty:
            return []
        
        # Look for common column names for tickers
        ticker_columns = ['ticker', 'symbol', 'stock', 'tickers', 'symbols', 'stocks']
        
        # Find the column with tickers (case-insensitive)
        ticker_col = None
        for col in df.columns:
            if str(col).lower().strip() in ticker_columns:
                ticker_col = col
                break
        
        # If no standard column found, use the first column
        if ticker_col is None:
            ticker_col = df.columns[0]
        
        # Extract tickers, clean and deduplicate
        tickers = df[ticker_col].dropna().astype(str).str.upper().str.strip().unique().tolist()
        
        # Filter out empty strings - allow alphanumeric tickers and common formats like BRK.B
        tickers = [t for t in tickers if t and len(t) <= 10 and t.replace('.', '').replace('-', '').isalnum()]
        
        return tickers[:50]  # Limit to 50 tickers for performance
    except Exception as e:
        st.error(f"Error parsing Excel file: {e}")
        return []


def analyze_portfolio(tickers: list, period: str, investment_amount: float, 
                     stock_fetcher, agent_system, sentiment_analyzer, 
                     investment_strategy) -> list:
    """
    Analyze multiple stocks and return portfolio breakdown.
    
    Args:
        tickers: List of stock ticker symbols
        period: Analysis period
        investment_amount: Total investment amount
        stock_fetcher: StockDataFetcher instance
        agent_system: MultiAgentSystem instance
        sentiment_analyzer: SentimentAnalyzer instance
        investment_strategy: BiweeklyInvestmentStrategy instance
        
    Returns:
        List of analysis results for each stock
    """
    if not tickers:
        return []
    
    # Pre-fetch data for all tickers to optimize performance
    stock_fetcher.fetch_batch_data(tickers, period)

    results = []
    per_stock_amount = investment_amount / len(tickers)
    
    for ticker in tickers:
        try:
            stock_data = stock_fetcher.get_stock_data(ticker, period)
            
            if stock_data is None or stock_data.empty:
                results.append({
                    'ticker': ticker,
                    'status': 'error',
                    'error': 'Could not fetch data'
                })
                continue
            
            stock_info = stock_fetcher.get_stock_info(ticker)
            technical_indicators = stock_fetcher.calculate_technical_indicators(stock_data)
            
            # Generate sentiment analysis
            prices = stock_data['Close'].tolist()
            volumes = stock_data['Volume'].tolist()
            news_texts = generate_mock_news(ticker, np.random.random())
            
            sentiment_result = sentiment_analyzer.calculate_composite_sentiment(
                news_texts=news_texts,
                prices=prices,
                volumes=volumes
            )
            
            composite_sentiment = sentiment_result['composite']
            sentiment_signal = sentiment_analyzer.get_sentiment_signal(composite_sentiment)
            
            # Run agent simulation
            market_data = {
                'price': prices[-1] if prices else 0,
                'volume': volumes[-1] if volumes else 0,
                'technical': technical_indicators
            }
            
            agent_decisions = agent_system.run_simulation(market_data, composite_sentiment)
            consensus = agent_system.get_consensus_decision(agent_decisions)
            
            # Calculate allocation using a temporary strategy to avoid side effects
            temp_strategy = BiweeklyInvestmentStrategy(investment_amount=per_stock_amount)
            allocation = temp_strategy.calculate_biweekly_allocation(
                agent_decisions, composite_sentiment
            )
            
            results.append({
                'ticker': ticker,
                'status': 'success',
                'name': stock_info.get('name', ticker),
                'sector': stock_info.get('sector', 'Unknown'),
                'current_price': technical_indicators.get('Current_Price', 0),
                'daily_change': technical_indicators.get('Daily_Change', 0),
                'sentiment': composite_sentiment,
                'sentiment_signal': sentiment_signal,
                'consensus': consensus,
                'action': allocation['action'],
                'confidence': allocation['confidence'],
                'recommended_amount': allocation['amount'],
                'buy_votes': allocation['agent_consensus']['buy'],
                'sell_votes': allocation['agent_consensus']['sell'],
                'hold_votes': allocation['agent_consensus']['hold']
            })
        except Exception as e:
            results.append({
                'ticker': ticker,
                'status': 'error',
                'error': str(e)
            })
    
    return results


def create_portfolio_breakdown_chart(results: list):
    """Create portfolio breakdown visualization."""
    # Filter successful results
    valid_results = [r for r in results if r['status'] == 'success']
    
    if not valid_results:
        return None
    
    # Action distribution using Counter for efficiency
    action_counter = Counter(r['action'] for r in valid_results)
    action_counts = {'BUY': action_counter.get('BUY', 0), 
                     'SELL': action_counter.get('SELL', 0), 
                     'HOLD': action_counter.get('HOLD', 0)}
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(action_counts.keys()),
            values=list(action_counts.values()),
            marker_colors=['#00ff41', '#ff3e3e', '#ffb000'],
            textinfo='label+percent',
            textfont=dict(color='#0a0e14', family='JetBrains Mono, monospace'),
            hole=0.4
        )
    ])
    
    fig.update_layout(
        title=dict(text=">> PORTFOLIO_ACTION_BREAKDOWN", 
                   font=dict(color='#00ff41', family='JetBrains Mono, monospace')),
        paper_bgcolor='rgba(10, 14, 20, 0.9)',
        font=dict(family='JetBrains Mono, monospace', color='#00d4ff'),
        height=300,
        showlegend=True,
        legend=dict(font=dict(color='#00d4ff'))
    )
    
    return fig


def create_sector_breakdown_chart(results: list):
    """Create sector distribution visualization."""
    valid_results = [r for r in results if r['status'] == 'success']
    
    if not valid_results:
        return None
    
    # Use Counter for efficient sector counting
    sectors = Counter(r.get('sector', 'Unknown') for r in valid_results)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(sectors.keys()),
            y=list(sectors.values()),
            marker_color='#00d4ff',
            text=list(sectors.values()),
            textposition='auto',
            textfont=dict(color='#0a0e14', family='JetBrains Mono, monospace')
        )
    ])
    
    fig.update_layout(
        title=dict(text=">> SECTOR_DISTRIBUTION", 
                   font=dict(color='#00ff41', family='JetBrains Mono, monospace')),
        yaxis_title="STOCK_COUNT",
        xaxis_title="SECTOR",
        template="plotly_dark",
        height=300,
        paper_bgcolor='rgba(10, 14, 20, 0.9)',
        plot_bgcolor='rgba(10, 14, 20, 0.9)',
        font=dict(family='JetBrains Mono, monospace', color='#00d4ff'),
        xaxis=dict(gridcolor='rgba(0, 255, 65, 0.1)', tickangle=-45),
        yaxis=dict(gridcolor='rgba(0, 255, 65, 0.1)')
    )
    
    return fig


# Page configuration - Terminal Style
st.set_page_config(
    page_title="GTSA Terminal // Game Theory Stock Agent",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Terminal-Style CSS
st.markdown("""
<style>
    /* Import terminal-style fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Fira+Code:wght@400;500;700&display=swap');
    
    /* Global terminal styling */
    .stApp {
        background-color: #0a0e14 !important;
        background-image: 
            repeating-linear-gradient(
                0deg,
                rgba(0, 255, 65, 0.03) 0px,
                rgba(0, 255, 65, 0.03) 1px,
                transparent 1px,
                transparent 2px
            );
    }
    
    /* Override Streamlit's default text colors */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }
    
    /* Main header - terminal prompt style */
    .main-header {
        font-size: 2rem;
        color: #00ff41;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace !important;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.7), 0 0 20px rgba(0, 255, 65, 0.5);
        letter-spacing: 2px;
    }
    
    .main-header::before {
        content: "$ ";
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.7);
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.2rem;
        color: #00d4ff;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-family: 'JetBrains Mono', monospace !important;
        text-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
        border-bottom: 1px solid rgba(0, 212, 255, 0.3);
        padding-bottom: 0.5rem;
    }
    
    .sub-header::before {
        content: "> ";
        color: #00ff41;
    }
    
    /* Terminal metric card */
    .metric-card {
        background-color: rgba(10, 14, 20, 0.9);
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #00ff41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2), inset 0 0 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Agent card - terminal window style */
    .agent-card {
        background-color: rgba(10, 14, 20, 0.95);
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid #00d4ff;
        margin: 0.5rem 0;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.15);
        position: relative;
    }
    
    .agent-card::before {
        content: "┌──[AGENT]";
        position: absolute;
        top: -10px;
        left: 10px;
        background: #0a0e14;
        padding: 0 5px;
        color: #00d4ff;
        font-size: 0.7rem;
    }
    
    /* Status colors */
    .bullish {
        color: #00ff41 !important;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(0, 255, 65, 0.5);
    }
    .bearish {
        color: #ff3e3e !important;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(255, 62, 62, 0.5);
    }
    .neutral {
        color: #ffb000 !important;
        font-weight: bold;
        text-shadow: 0 0 8px rgba(255, 176, 0, 0.5);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid #00ff41 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #c0c0c0 !important;
    }
    
    /* Input fields - terminal style */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #0d1117 !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.4) !important;
    }
    
    /* Buttons - terminal style */
    .stButton > button {
        background-color: transparent !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: rgba(0, 255, 65, 0.1) !important;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.4);
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.8);
    }
    
    .stButton > button[kind="primary"] {
        background-color: rgba(0, 255, 65, 0.15) !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background-color: rgba(10, 14, 20, 0.9) !important;
        border: 1px solid #00d4ff !important;
        color: #c0c0c0 !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #00ff41 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-shadow: 0 0 8px rgba(0, 255, 65, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: #00ff41 !important;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(10, 14, 20, 0.9) !important;
        border: 1px solid #00d4ff !important;
        color: #00d4ff !important;
    }
    
    /* Data frames */
    .stDataFrame {
        border: 1px solid #00ff41 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e14;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff41;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00d4ff;
    }
    
    /* Blinking cursor animation */
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .cursor {
        display: inline-block;
        width: 10px;
        height: 1.2em;
        background-color: #00ff41;
        margin-left: 2px;
        animation: blink 1s infinite;
    }
    
    /* Terminal box */
    .terminal-box {
        background-color: rgba(10, 14, 20, 0.95);
        border: 1px solid #00ff41;
        border-radius: 4px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.1), inset 0 0 30px rgba(0, 0, 0, 0.5);
    }
    
    /* System message style */
    .sys-msg {
        color: #00d4ff;
        font-size: 0.9rem;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .sys-msg::before {
        content: "[SYS] ";
        color: #ffb000;
    }
    
    /* ASCII art header styling */
    .ascii-header {
        font-family: 'JetBrains Mono', monospace !important;
        color: #00ff41;
        font-size: 0.6rem;
        line-height: 1.1;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
        white-space: pre;
        margin-bottom: 1rem;
    }
    
    /* Glitch effect for headers */
    @keyframes glitch {
        0% { text-shadow: 0 0 10px rgba(0, 255, 65, 0.7), 0 0 20px rgba(0, 255, 65, 0.5); }
        25% { text-shadow: -2px 0 #ff3e3e, 2px 0 #00d4ff; }
        50% { text-shadow: 0 0 10px rgba(0, 255, 65, 0.7), 0 0 20px rgba(0, 255, 65, 0.5); }
        75% { text-shadow: 2px 0 #ff3e3e, -2px 0 #00d4ff; }
        100% { text-shadow: 0 0 10px rgba(0, 255, 65, 0.7), 0 0 20px rgba(0, 255, 65, 0.5); }
    }
    
    .glitch-text {
        animation: glitch 3s infinite;
    }
    
    /* Accessibility: Respect user's motion preferences */
    @media (prefers-reduced-motion: reduce) {
        .glitch-text {
            animation: none;
            text-shadow: 0 0 10px rgba(0, 255, 65, 0.7), 0 0 20px rgba(0, 255, 65, 0.5);
        }
        .cursor {
            animation: none;
            opacity: 1;
        }
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
        increasing_line_color='#00ff41',
        decreasing_line_color='#ff3e3e'
    ))
    
    # Add moving averages if enough data
    if len(df) >= 20:
        ma20 = df['Close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma20,
            name='MA20',
            line=dict(color='#ffb000', width=1)
        ))
    
    if len(df) >= 50:
        ma50 = df['Close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma50,
            name='MA50',
            line=dict(color='#00d4ff', width=1)
        ))
    
    fig.update_layout(
        title=dict(text=f">> {ticker} PRICE_DATA", font=dict(color='#00ff41', family='JetBrains Mono, monospace')),
        yaxis_title="PRICE (USD)",
        xaxis_title="TIMESTAMP",
        template="plotly_dark",
        height=400,
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(10, 14, 20, 0.9)',
        plot_bgcolor='rgba(10, 14, 20, 0.9)',
        font=dict(family='JetBrains Mono, monospace', color='#00d4ff'),
        xaxis=dict(gridcolor='rgba(0, 255, 65, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(0, 255, 65, 0.1)', showgrid=True)
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
        
        # Input mode selection
        input_mode = st.radio(
            ">> INPUT_MODE",
            options=["Single Stock", "Excel Upload"],
            help="Choose to analyze a single stock or upload an Excel file with multiple stocks"
        )
        
        # Excel file upload (only show if Excel mode selected)
        uploaded_file = None
        if input_mode == "Excel Upload":
            st.markdown("""
            <div class="terminal-box">
                <span style="color: #ffb000;">FORMAT:</span> Excel (.xlsx)<br>
                <small style="color: #888;">Column: ticker, symbol, or stock</small>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                ">> UPLOAD_EXCEL",
                type=["xlsx"],
                help="Upload Excel file with stock tickers"
            )
        
        # Stock selection (only show if single mode)
        ticker = ""
        if input_mode == "Single Stock":
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
        # Handle Excel upload mode
        if input_mode == "Excel Upload":
            if uploaded_file is None:
                st.error("❌ Please upload an Excel file to analyze.")
            else:
                tickers = parse_excel_tickers(uploaded_file)
                
                if not tickers:
                    st.error("❌ No valid tickers found in the Excel file. Ensure it has a column named 'ticker', 'symbol', or 'stock'.")
                else:
                    with st.spinner(f"Analyzing {len(tickers)} stocks from Excel..."):
                        # Analyze portfolio
                        results = analyze_portfolio(
                            tickers=tickers,
                            period=period,
                            investment_amount=investment_amount,
                            stock_fetcher=st.session_state.stock_fetcher,
                            agent_system=st.session_state.agent_system,
                            sentiment_analyzer=st.session_state.sentiment_analyzer,
                            investment_strategy=st.session_state.investment_strategy
                        )
                        
                        # Display portfolio breakdown
                        st.markdown('<div class="sub-header">📊 PORTFOLIO_BREAKDOWN</div>', unsafe_allow_html=True)
                        
                        # Summary metrics
                        valid_results = [r for r in results if r['status'] == 'success']
                        error_results = [r for r in results if r['status'] == 'error']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("TOTAL_STOCKS", len(tickers))
                        
                        with col2:
                            st.metric("ANALYZED", len(valid_results))
                        
                        with col3:
                            buy_count = len([r for r in valid_results if r['action'] == 'BUY'])
                            st.metric("BUY_SIGNALS", buy_count)
                        
                        with col4:
                            avg_sentiment = np.mean([r['sentiment'] for r in valid_results]) if valid_results else 0
                            st.metric("AVG_SENTIMENT", f"{avg_sentiment*100:.1f}%")
                        
                        # Charts
                        st.markdown('<div class="sub-header">📈 ANALYSIS_CHARTS</div>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            action_chart = create_portfolio_breakdown_chart(results)
                            if action_chart:
                                st.plotly_chart(action_chart, use_container_width=True)
                        
                        with col2:
                            sector_chart = create_sector_breakdown_chart(results)
                            if sector_chart:
                                st.plotly_chart(sector_chart, use_container_width=True)
                        
                        # Detailed stock breakdown table
                        st.markdown('<div class="sub-header">📋 STOCK_DETAILS</div>', unsafe_allow_html=True)
                        
                        if valid_results:
                            # Create summary dataframe
                            df_data = []
                            for r in valid_results:
                                action_emoji = "🟢" if r['action'] == 'BUY' else "🔴" if r['action'] == 'SELL' else "🟡"
                                df_data.append({
                                    'Ticker': r['ticker'],
                                    'Name': r['name'][:25] if len(r['name']) > 25 else r['name'],
                                    'Sector': r['sector'],
                                    'Price': f"${r['current_price']:.2f}",
                                    'Change': f"{r['daily_change']:.2f}%",
                                    'Sentiment': r['sentiment_signal'],
                                    'Action': f"{action_emoji} {r['action']}",
                                    'Confidence': f"{r['confidence']*100:.1f}%",
                                    'Rec. Amount': f"${r['recommended_amount']:.2f}"
                                })
                            
                            df_summary = pd.DataFrame(df_data)
                            st.dataframe(df_summary, use_container_width=True, hide_index=True)
                            
                            # Investment allocation summary
                            st.markdown('<div class="sub-header">💰 ALLOCATION_SUMMARY</div>', unsafe_allow_html=True)
                            
                            total_buy = sum(r['recommended_amount'] for r in valid_results if r['action'] == 'BUY')
                            total_sell = sum(r['recommended_amount'] for r in valid_results if r['action'] == 'SELL')
                            total_hold = sum(r['recommended_amount'] for r in valid_results if r['action'] == 'HOLD')
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="terminal-box">
                                    <span class="bullish">BUY_ALLOCATION</span><br>
                                    <span style="color: #00ff41; font-size: 1.5rem;">${total_buy:.2f}</span><br>
                                    <small style="color: #888;">{len([r for r in valid_results if r['action'] == 'BUY'])} stocks</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="terminal-box">
                                    <span class="bearish">SELL_SIGNALS</span><br>
                                    <span style="color: #ff3e3e; font-size: 1.5rem;">${total_sell:.2f}</span><br>
                                    <small style="color: #888;">{len([r for r in valid_results if r['action'] == 'SELL'])} stocks</small>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="terminal-box">
                                    <span class="neutral">HOLD_ALLOCATION</span><br>
                                    <span style="color: #ffb000; font-size: 1.5rem;">${total_hold:.2f}</span><br>
                                    <small style="color: #888;">{len([r for r in valid_results if r['action'] == 'HOLD'])} stocks</small>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show errors if any
                        if error_results:
                            with st.expander(f"⚠️ FAILED_ANALYSIS ({len(error_results)} stocks)"):
                                for r in error_results:
                                    st.markdown(f"""
                                    <div style="padding: 0.3rem 0; border-bottom: 1px solid rgba(255, 62, 62, 0.2);">
                                        <span style="color: #ff3e3e;">✗</span> <span style="color: #c0c0c0;">{r['ticker']}</span>
                                        <small style="color: #888;"> - {r.get('error', 'Unknown error')}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
        
        # Handle single stock mode
        else:
            if not ticker:
                st.error("❌ Please enter a ticker symbol.")
            else:
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
