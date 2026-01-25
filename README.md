# 🎮 Game Theory Stock Agent Dashboard

An interactive AI-powered dashboard that combines **multi-agent systems**, **game theory**, **sentiment analysis**, and **technical analysis** for intelligent biweekly stock investing.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

### 🤖 Multi-Agent AI System
- **5 specialized AI agents** with diverse strategies:
  - **Agent Alpha** (Aggressive): High-risk, high-reward approach
  - **Agent Beta** (Conservative): Risk-averse, stability-focused
  - **Agent Gamma** (Balanced): Moderate risk tolerance
  - **Agent Delta** (Contrarian): Goes against market consensus
  - **Agent Epsilon** (Balanced): Secondary balanced perspective

### 🎮 Game-Theoretic Analysis
- **Nash Equilibrium Calculation**: Find stable states where no agent benefits from changing strategy
- **Best Response Dynamics**: Iterative refinement of agent decisions
- **Coordination Games**: Agents observe and respond to each other's actions
- **Prisoner's Dilemma**: Contrarian strategies vs. consensus

### 😊 Advanced Sentiment Analysis
- **Multi-source sentiment** from:
  - News headlines (NLP-based text analysis)
  - Price momentum indicators
  - Volume trend analysis
- **Composite sentiment scores** with confidence metrics
- **Real-time market sentiment** visualization

### 📈 Technical Analysis
- **Moving Averages** (MA20, MA50)
- **RSI (Relative Strength Index)**
- **Volatility Analysis**
- **Interactive candlestick charts**

### 💰 Biweekly Investment Strategy
- **Automated allocation** recommendations
- **Consensus-based decisions** from multiple agents
- **Risk-adjusted positioning**
- **Investment timeline tracking**

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

#### Local Development
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

#### Deployment for Testing

For detailed deployment instructions to various platforms, see **[DEPLOYMENT.md](DEPLOYMENT.md)**.

**Quick deploy to Streamlit Community Cloud (Recommended):**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and deploy in 3 clicks
4. Get a public URL to share: `https://[app-name].streamlit.app`

**Other deployment options:**
- 🐳 Docker (see `Dockerfile`)
- 🚀 Heroku (see `Procfile`)
- 🎨 Render (see `render.yaml`)
- 🚂 Railway
- And more in [DEPLOYMENT.md](DEPLOYMENT.md)

## 📖 Usage

1. **Enter a stock ticker** (e.g., AAPL, GOOGL, MSFT) in the sidebar
2. **Select analysis period** (1mo, 3mo, 6mo, 1y)
3. **Set investment amount** for biweekly investing
4. **Click "Run Analysis"** to activate the AI agents
5. **Review the results**:
   - Stock price charts with technical indicators
   - Market sentiment gauge and components
   - Individual agent decisions and confidence levels
   - Nash equilibrium analysis
   - Investment recommendations

## 🏗️ Architecture

```
littlemoneybuddy/
├── app.py                    # Main Streamlit dashboard
├── game_theory_agent.py      # Multi-agent system & game theory
├── sentiment_analyzer.py     # Sentiment analysis module
├── stock_data.py            # Stock data fetching & processing
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Core Components

#### 1. Game Theory Agent (`game_theory_agent.py`)
- **`GameTheoryAgent`**: Individual agent with strategy and decision-making
- **`MultiAgentSystem`**: Manages agent coordination and Nash equilibrium
- **Payoff Calculation**: Game-theoretic reward function
- **Consensus Building**: Weighted voting from all agents

#### 2. Sentiment Analyzer (`sentiment_analyzer.py`)
- **Text Analysis**: NLP-based sentiment from news
- **Price Momentum**: Sentiment from price trends
- **Volume Analysis**: Trading volume patterns
- **Composite Scoring**: Multi-source sentiment aggregation

#### 3. Stock Data (`stock_data.py`)
- **Data Fetching**: Yahoo Finance integration
- **Technical Indicators**: RSI, MA, volatility
- **Biweekly Strategy**: Investment allocation logic

#### 4. Dashboard (`app.py`)
- **Interactive UI**: Streamlit-based web interface
- **Visualizations**: Plotly charts and gauges
- **Real-time Analysis**: On-demand agent simulation

## 🎯 How It Works

### Multi-Agent Decision Process

1. **Initial Assessment**: Each agent independently evaluates the stock
2. **Observation**: Agents observe others' preliminary decisions
3. **Best Response**: Agents refine decisions based on game theory
4. **Iteration**: Process repeats until Nash equilibrium is approached
5. **Consensus**: Final weighted recommendation is generated

### Game-Theoretic Principles

**Nash Equilibrium**: The system finds a stable state where:
- No agent can improve their payoff by changing strategy alone
- All agents have selected their best response to others' actions
- Stability score indicates how close to equilibrium (>70% = equilibrium)

**Coordination Games**: 
- Agents receive bonuses for coordinating with majority
- Simulates market behavior where herding can be beneficial
- Contrarian agents can exploit overcrowding

**Payoff Function**:
```python
payoff = base_sentiment * 10 
       + coordination_bonus 
       + contrarian_bonus 
       * risk_tolerance
```

## 📊 Example Analysis

For **Apple (AAPL)**:

1. **Sentiment Analysis**:
   - Text sentiment: 75% (positive news)
   - Price momentum: 68% (upward trend)
   - Volume: 82% (high interest)
   - **Composite: 73% Bullish**

2. **Agent Decisions**:
   - 4 agents recommend BUY
   - 1 agent recommends HOLD
   - Consensus: **BUY with 80% confidence**

3. **Nash Equilibrium**:
   - Buy ratio: 80%
   - Stability score: 85%
   - **Status: At Equilibrium ✅**

4. **Recommendation**:
   - Action: **BUY**
   - Amount: **$730 (73% of $1000)**
   - Next investment: **7 days**

## 🔬 Research Integration

This dashboard integrates concepts from:

- **Multi-Agent Systems**: Distributed AI with diverse strategies
- **Game Theory**: Nash equilibrium, coordination games, prisoner's dilemma
- **Behavioral Finance**: Sentiment analysis and market psychology
- **Technical Analysis**: Price patterns and momentum indicators
- **Reinforcement Learning**: Iterative best response dynamics

## 🛠️ Dependencies

- **streamlit**: Interactive web dashboard
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **yfinance**: Stock market data
- **textblob**: NLP sentiment analysis
- **scipy**: Scientific computing
- **scikit-learn**: Machine learning utilities

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Real news API integration (NewsAPI, Alpha Vantage)
- Additional agent strategies (momentum, mean-reversion)
- Deep learning sentiment models
- Portfolio optimization across multiple stocks
- Backtesting framework
- Paper trading integration

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