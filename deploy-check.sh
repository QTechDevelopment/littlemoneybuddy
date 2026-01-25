#!/bin/bash
# Quick deployment test script

echo "🚀 Little Money Buddy - Deployment Verification"
echo "================================================"
echo ""

# Check Python version
echo "📌 Checking Python version..."
python --version
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
python -m pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Verify imports
echo "🔍 Verifying app imports..."
python -c "import app; print('✓ App imports successfully')"
python -c "from stock_data import StockDataFetcher; print('✓ StockDataFetcher available')"
python -c "from game_theory_agent import MultiAgentSystem; print('✓ MultiAgentSystem available')"
python -c "from sentiment_analyzer import SentimentAnalyzer; print('✓ SentimentAnalyzer available')"
echo ""

# Check configuration
echo "⚙️  Checking configuration..."
if [ -f ".streamlit/config.toml" ]; then
    echo "✓ Streamlit config found"
else
    echo "⚠ Warning: .streamlit/config.toml not found"
fi
echo ""

# Check deployment files
echo "📋 Checking deployment files..."
[ -f "Dockerfile" ] && echo "✓ Dockerfile" || echo "⚠ Dockerfile missing"
[ -f "Procfile" ] && echo "✓ Procfile (Heroku)" || echo "⚠ Procfile missing"
[ -f "render.yaml" ] && echo "✓ render.yaml (Render)" || echo "⚠ render.yaml missing"
[ -f "runtime.txt" ] && echo "✓ runtime.txt" || echo "⚠ runtime.txt missing"
[ -f "DEPLOYMENT.md" ] && echo "✓ DEPLOYMENT.md" || echo "⚠ DEPLOYMENT.md missing"
echo ""

echo "✅ Verification complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Review DEPLOYMENT.md for deployment options"
echo "2. For quick testing, use Streamlit Community Cloud"
echo "3. Run locally: streamlit run app.py"
echo ""
