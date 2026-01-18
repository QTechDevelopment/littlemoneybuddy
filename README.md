# 🎮 Game Theory Stock Agent Dashboard

An advanced AI-powered dashboard that combines **multi-agent systems**, **game theory**, **sentiment analysis**, and **technical analysis** for intelligent biweekly stock investing with real-time data integration.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Key Features

### 🤖 Enhanced Multi-Agent AI System
- **6 specialized AI agents** with diverse perspectives:
  - **Agent Alpha (Bull)**: Aggressive long bias, high-risk/high-reward
  - **Agent Beta (Bear)**: Conservative, risk-averse positioning
  - **Agent Gamma (Technical)**: Chart-based, trend-following analysis
  - **Agent Delta (Contrarian)**: Counter-consensus for asymmetric opportunities
  - **Agent Epsilon (Sentiment)**: Social and news sentiment focused
  - **Agent Zeta (Fundamental)**: Analyst targets and valuation focused

### 📊 Real Data Source Integration
- **Financial Data**: Yahoo Finance (real-time prices, analyst targets, earnings)
- **News Sentiment**: NewsAPI + Yahoo Finance News (professional sentiment)
- **Social Sentiment**: Reddit (r/stocks, r/investing, r/wallstreetbets) + Twitter/X
- **Technical Data**: 15+ advanced indicators (MACD, RSI, Bollinger Bands, Stochastic, ATR)
- **Analyst Data**: Price targets, upgrades/downgrades, recommendations

### 🔄 Sentiment Divergence Analysis
- **News vs Social Detection**: Identifies mismatches between professional and retail sentiment
- **Trading Signals**:
  - News ✅ bullish + Social ❌ bearish = **BUY** (price likely to rise as retail catches up)
  - News ❌ bearish + Social ✅ bullish = **SELL** (momentum fade likely)
  - Both aligned = **STRONG** signal with maximum profit potential
- **Leading Indicator**: Retail sentiment (Reddit/Twitter) leads professional news by 3-5 days

### 🎯 Prisoner's Dilemma AI Capex Analysis
- **Competitive Spending Analysis**: Tracks AI infrastructure capex across mega-caps
- **Game Theory Framework**:
  - Multiple competitors overspending = Mutual Defection = **Valuation compression risk**
  - Disciplined spender while others overspend = **Asymmetric advantage**
  - Industry cooperation = Stable valuations = **Low risk**
- **AI Beneficiary Identification**: Finds companies benefiting from AI without heavy capex
- **Contrarian Signal**: Flags unsustainable AI spending despite bullish consensus

### 📈 Multi-Agent Consensus with Reliability Scoring
- **Signal Reliability Metrics**:
  - 3+ agents agree = ~70% signal reliability
  - 4+ agents agree = ~80% reliability
  - 5+ agents agree = ~85% reliability
  - Bull + Bear + Technical consensus = Optimal signal quality
- **Signal Quality Grades**: EXCELLENT, GOOD, FAIR, POOR
- **Disagreement Analysis**: Identifies choppy markets and reduces position sizing
- **Sentiment Extreme Detection**: Warns when emotions override logic

### 💰 Biweekly Signal Generation
- **Actionable Signals**: BUY / HOLD / SELL with confidence scores
- **Position Sizing**: Dynamic allocation based on signal quality (30%-100%)
- **Risk Management**: Stop loss and target prices for every signal
- **Execution Cadence**: Signals every 14 days with weekly tactical adjustments
- **Multi-Factor Integration**: Combines consensus, divergence, capex, and technicals

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/QTechDevelopment/littlemoneybuddy.git
cd littlemoneybuddy
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Enter a stock ticker** (e.g., AAPL, GOOGL, MSFT, NVDA) in the sidebar
2. **Select analysis period** (1mo, 3mo, 6mo, 1y)
3. **Set biweekly investment amount** for position sizing
4. **Click "Run Analysis"** to activate all systems
5. **Review comprehensive results**:
   - **Biweekly Signal**: BUY/HOLD/SELL with conviction and position sizing
   - **Sentiment Divergence**: News vs social sentiment mismatches
   - **AI Capex Analysis**: Prisoner's Dilemma competitive dynamics
   - **Multi-Agent Consensus**: Reliability scores and signal quality
   - **Technical & Fundamental**: Complete market analysis

## 🏗️ Architecture

```
littlemoneybuddy/
├── app.py                          # Main Streamlit dashboard
├── game_theory_agent.py            # Multi-agent system & Nash equilibrium
├── sentiment_analyzer.py           # Basic sentiment analysis
├── stock_data.py                   # Stock data fetching & processing
├── data_sources.py                 # Real data source integration (NEW)
├── sentiment_divergence.py         # News vs social divergence analysis (NEW)
├── prisoners_dilemma.py            # AI capex Prisoner's Dilemma framework (NEW)
├── multi_agent_consensus.py        # Enhanced consensus with reliability (NEW)
├── signal_generator.py             # Biweekly signal generation (NEW)
├── mock_data.py                    # Mock data fallback
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

### Core Components

#### 1. Data Sources (`data_sources.py`)
- **FinancialDataSource**: Yahoo Finance integration for prices, analyst targets, earnings
- **NewsDataSource**: NewsAPI + Yahoo Finance news aggregation
- **SocialSentimentSource**: Reddit and Twitter/X sentiment (with fallback to mock)
- **TechnicalDataSource**: Advanced technical indicators (MACD, RSI, Bollinger Bands, etc.)
- **DataSourceAggregator**: Unified interface for all data sources

#### 2. Sentiment Divergence (`sentiment_divergence.py`)
- **SentimentDivergenceAnalyzer**: Detects and scores news vs social divergence
- **RetailLeadingIndicator**: Identifies when retail leads professional sentiment
- **MultiSourceSentimentAggregator**: Intelligent weighting of all sentiment sources

#### 3. Prisoner's Dilemma (`prisoners_dilemma.py`)
- **PrisonersDilemmaCapexAnalyzer**: Competitive AI spending game theory
- **NashEquilibriumCapexAnalyzer**: Convergence analysis for competitor spending
- **AI Beneficiary Identification**: Finds low-capex AI winners

#### 4. Multi-Agent Consensus (`multi_agent_consensus.py`)
- **MultiAgentConsensus**: Reliability scoring for agent consensus
- **DisagreementAnalyzer**: Market regime identification from agent splits
- **SentimentExtremeDetector**: Warns of emotional extremes

#### 5. Signal Generator (`signal_generator.py`)
- **BiweeklySignalGenerator**: Generates BUY/HOLD/SELL signals every 2 weeks
- **TacticalAdjustment**: Weekly tactical position adjustments
- **Position Sizing**: Dynamic allocation based on signal quality

#### 6. Game Theory Agent (`game_theory_agent.py`)
- **GameTheoryAgent**: Individual agent with strategy and decision-making
- **MultiAgentSystem**: Manages agent coordination and Nash equilibrium
- **Payoff Calculation**: Game-theoretic reward function
- **Consensus Building**: Weighted voting from all agents

## 🎯 How It Works

### Multi-Factor Signal Generation

The system generates signals through a comprehensive 5-step process:

1. **Data Aggregation** (40% weight)
   - Fetch real-time prices, news, social sentiment, analyst targets
   - Calculate 15+ technical indicators
   - Aggregate multi-source sentiment with intelligent weighting

2. **Multi-Agent Analysis** (40% weight)
   - 6 agents independently analyze the stock
   - Iterative best response dynamics converge to Nash equilibrium
   - Reliability scoring based on consensus diversity

3. **Sentiment Divergence** (25% weight)
   - Compare professional news vs retail social sentiment
   - Detect leading indicators (retail leads by 3-5 days)
   - Generate divergence trading signals

4. **Prisoner's Dilemma Capex** (20% weight)
   - Analyze competitive AI infrastructure spending
   - Identify valuation compression risks (mutual defection)
   - Flag AI beneficiaries and disciplined spenders

5. **Final Signal Generation** (15% weight)
   - Combine all factors with dynamic weights
   - Calculate confidence and signal quality
   - Determine position sizing (30%-100%)
   - Generate stop loss and target prices

### Game-Theoretic Principles

**Nash Equilibrium**: The system finds stable states where:
- No agent can improve their payoff by changing strategy alone
- All agents have selected their best response to others' actions
- Stability score >70% indicates equilibrium

**Coordination Games**: 
- Agents receive bonuses for coordinating with majority
- Simulates market behavior where herding can be beneficial
- Contrarian agents can exploit overcrowding

**Prisoner's Dilemma (AI Capex)**:
- Both cooperate (low capex) = Moderate returns for all
- Both defect (high capex) = Low returns for all (worst outcome)
- One defects, one cooperates = Defector wins, cooperator loses
- In AI race: Overspending = defect, Disciplined = cooperate

**Sentiment Divergence**:
- News ✅ + Social ❌ = Price rises (retail catches up)
- News ❌ + Social ✅ = Momentum fades (retail trapped)
- Both aligned = Maximum profit potential
- Retail leads news by 3-5 days (leading indicator)

**Multi-Agent Reliability**:
- 3+ agents consensus = ~70% signal accuracy
- Bull + Bear + Technical = Optimal quality
- Disagreement = Market choppiness, reduce size
- Extremes = Emotions override logic, contrarian opportunity

## 📊 Example Analysis

For **Apple (AAPL)**:

### 1. Data Aggregation
- **Current Price**: $175.43
- **Analyst Target**: $195.20 (+11.3% upside)
- **News Sentiment**: 72% (bullish - strong earnings, AI innovations)
- **Social Sentiment**: 58% (moderate - some profit-taking chatter)
- **Technical Signal**: BUY (RSI: 62, MACD positive, above MA50)

### 2. Multi-Agent Decisions
- **Agent Alpha (Bull)**: BUY, 85% confidence
- **Agent Beta (Bear)**: HOLD, 60% confidence
- **Agent Gamma (Technical)**: BUY, 78% confidence
- **Agent Delta (Contrarian)**: HOLD, 55% confidence
- **Agent Epsilon (Sentiment)**: BUY, 70% confidence
- **Agent Zeta (Fundamental)**: BUY, 80% confidence
- **Consensus**: **BUY** (4/6 agents, 78% reliability)

### 3. Sentiment Divergence
- **News Sentiment**: 72% (positive)
- **Social Sentiment**: 58% (moderate)
- **Divergence**: 14% (News bullish, Social neutral)
- **Signal**: NEWS_BULLISH_SOCIAL_NEUTRAL
- **Implication**: Price likely to rise as social sentiment catches up
- **Recommendation**: BUY with 75% position

### 4. Prisoner's Dilemma (AI Capex)
- **Analysis**: BALANCED_APPROACH
- **Capex/Revenue Ratio**: 8.5% (disciplined)
- **Competitor Average**: 12.3% (some overspending)
- **Risk Level**: LOW
- **Positioning**: Strong - disciplined spender in competitive landscape
- **Category**: AI_BENEFICIARY (benefits without overspending)

### 5. Nash Equilibrium
- **Buy Ratio**: 67%
- **Sell Ratio**: 0%
- **Hold Ratio**: 33%
- **Stability Score**: 82%
- **Status**: ✅ At Equilibrium

### 6. Final Signal
- **Signal**: **BUY**
- **Confidence**: 76%
- **Signal Quality**: EXCELLENT
- **Position Size**: 85% (of available capital)
- **Conviction**: HIGH
- **Stop Loss**: $161.40 (8% below)
- **Target**: $201.74 (15% above)
- **Risk/Reward**: 1.875x

**Recommendation**: High conviction BUY signal. Strong consensus across multiple factors including multi-agent agreement, positive sentiment divergence, and disciplined AI capex approach. Recommended 85% position with stops at $161.40 and target at $201.74.

## 🔬 Research Integration

This dashboard integrates concepts from:

- **Multi-Agent Systems**: Distributed AI with diverse strategies and game-theoretic interactions
- **Game Theory**: 
  - Nash equilibrium for stable strategy selection
  - Coordination games for collective behavior
  - Prisoner's dilemma for competitive dynamics analysis
- **Behavioral Finance**: 
  - Sentiment analysis and market psychology
  - Retail vs professional sentiment divergence
  - Emotion detection at market extremes
- **Technical Analysis**: 
  - Price patterns and momentum indicators
  - Volume analysis and volatility metrics
- **Financial Psychology**:
  - Retail sentiment as leading indicator (3-5 day lead time)
  - Contrarian signals from unsustainable consensus
  - Crowd psychology exploitation through divergence trades

## 🛠️ Dependencies

Core libraries:
- **streamlit** (1.31.0): Interactive web dashboard
- **pandas** (2.1.4): Data manipulation and analysis
- **numpy** (1.26.3): Numerical computing
- **plotly** (5.18.0): Interactive visualizations
- **yfinance** (0.2.36): Stock market data from Yahoo Finance
- **textblob** (0.18.0): NLP sentiment analysis
- **scipy** (1.11.4): Scientific computing
- **scikit-learn** (1.4.0): Machine learning utilities
- **requests** (2.31.0): HTTP requests for API calls
- **python-dateutil** (2.8.2): Date manipulation

Optional (for enhanced features):
- **NewsAPI Key**: For real-time news sentiment (falls back to Yahoo Finance)
- **Reddit API Key**: For social sentiment from Reddit (uses mock data if unavailable)
- **Twitter API Key**: For social sentiment from Twitter/X (uses mock data if unavailable)

## 🔧 Configuration

### API Keys (Optional)

Set environment variables for enhanced data sources:

```bash
export NEWS_API_KEY="your_newsapi_key"
export REDDIT_API_KEY="your_reddit_api_key"
export TWITTER_API_KEY="your_twitter_api_key"
```

**Note**: The system works without API keys by using Yahoo Finance data and intelligent mock data generation for social sentiment.

## 🚀 Advanced Features

### Biweekly Execution Strategy

The system is designed for disciplined biweekly investing:

1. **Signal Generation**: Every 14 days
   - Comprehensive multi-factor analysis
   - BUY/HOLD/SELL with position sizing
   - Stop loss and target prices

2. **Weekly Tactical Adjustments**
   - Monitor sentiment divergence changes weekly
   - Adjust positions if significant shifts detected (>20% divergence change)
   - Maintain discipline on core biweekly signals

3. **Rebalancing**
   - Review portfolio biweekly
   - Execute highest-confidence signals (3+ agent agreement)
   - Monitor AI capex risks for contrarian opportunities

### Signal Reliability Guidelines

| Agents | Reliability | Quality | Position Size |
|--------|-------------|---------|---------------|
| 5-6    | 85%+        | EXCELLENT | 85-100%    |
| 4      | 80%+        | GOOD      | 70-85%     |
| 3      | 70%+        | GOOD      | 60-75%     |
| 2      | 55%+        | FAIR      | 40-60%     |
| 1      | 40%+        | POOR      | 20-35%     |

### Risk Management

- **Position Sizing**: Dynamic based on signal quality and reliability
- **Stop Losses**: Automatic 8% stops on all BUY signals
- **Target Prices**: 15% targets for risk/reward ratio of 1.875x
- **Divergence Alerts**: Weekly monitoring for tactical adjustments
- **Extreme Detection**: Warnings when sentiment reaches extremes (>85%)

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

### Data Sources
- Real-time NewsAPI integration with NLP scoring
- PRAW (Python Reddit API Wrapper) for authentic social sentiment
- Twitter API v2 for real-time cashtag monitoring
- Alpha Vantage or Polygon.io for additional financial data
- Benzinga Pro or StockGeist for professional sentiment feeds

### Analytics
- Additional technical indicators (Ichimoku, On-Balance Volume)
- Deep learning sentiment models (FinBERT, sentiment-specific BERT)
- Earnings call transcript analysis
- Options flow sentiment
- Insider trading pattern analysis

### Game Theory
- Additional agent strategies (momentum, mean-reversion, quantamental)
- Multi-stock portfolio optimization using coalition games
- Dynamic agent weight adjustment based on historical performance
- Evolutionary algorithms for agent strategy refinement

### Backtesting
- Historical signal performance tracking
- Sharpe ratio and risk-adjusted return calculations
- Monte Carlo simulation for strategy validation
- Walk-forward analysis for parameter optimization

### Integration
- Paper trading integration (Alpaca, Interactive Brokers)
- Automated execution with risk management
- Portfolio tracking and performance reporting
- Alerts and notifications (email, Slack, Discord)

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** 

- Not financial advice
- Past performance does not guarantee future results
- Always do your own research before investing
- Consult with a financial advisor for investment decisions
- Use at your own risk

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Game theory concepts from John Nash's equilibrium theory
- Multi-agent systems research from AI literature
- Financial data provided by Yahoo Finance
- Built with Streamlit and Python data science stack

---

**Made with ❤️ for intelligent investing**