# 🚀 Deployment Guide for Little Money Buddy

## Overview

Little Money Buddy is a **Streamlit application** that requires a persistent server to run. This guide provides deployment options for various platforms.

---

## ⚡ Quick Deploy Options

### 1. Streamlit Community Cloud (Recommended) ⭐

**Best for**: Free, easy Streamlit hosting with zero configuration

**Steps**:
1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your forked repository
6. Set main file path: `app.py`
7. Click "Deploy"

**Pros**:
- ✅ Free forever
- ✅ Optimized for Streamlit
- ✅ Automatic SSL
- ✅ Easy updates via git push

**Cons**:
- ⚠️ Public apps only (private apps require Streamlit for Teams)
- ⚠️ Resource limits on free tier

---

### 2. Vercel (Current Setup)

**Status**: ⚠️ Limited Support

Vercel is configured for this repository, but **Streamlit apps require persistent servers** which conflicts with Vercel's serverless architecture.

**What works**:
- ✅ Static landing page explaining the app
- ✅ Links to GitHub repository
- ✅ Deployment instructions

**What doesn't work**:
- ❌ Interactive Streamlit dashboard
- ❌ Real-time data updates
- ❌ WebSocket connections (required by Streamlit)

**Deploy to Vercel**:
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

Or connect your GitHub repository to Vercel through their web interface.

---

### 3. Railway 🚂

**Best for**: Modern deployment with easy Python support

**Steps**:
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select this repository
5. Railway will auto-detect Python and use `requirements.txt`
6. Add start command: `streamlit run app.py --server.port $PORT`
7. Deploy

**Pros**:
- ✅ Free tier with $5 credit/month
- ✅ Automatic HTTPS
- ✅ Environment variables support
- ✅ Simple deployment

**Cons**:
- ⚠️ May require payment after free tier

---

### 4. Heroku

**Best for**: Established platform with Python support

**Setup**:

1. Create `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

heroku login
heroku create your-app-name
git push heroku main
heroku open
```

**Pros**:
- ✅ Free tier available
- ✅ Well-documented
- ✅ Add-ons ecosystem

**Cons**:
- ⚠️ Free tier sleeps after 30 minutes of inactivity
- ⚠️ Credit card required for free tier

---

### 5. Google Cloud Run

**Best for**: Scalable, containerized deployment

**Setup**:

1. Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true
```

2. Deploy:
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/littlemoneybuddy
gcloud run deploy littlemoneybuddy \
  --image gcr.io/YOUR_PROJECT_ID/littlemoneybuddy \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Pros**:
- ✅ Auto-scaling
- ✅ Pay only for what you use
- ✅ Free tier available
- ✅ Enterprise-grade

**Cons**:
- ⚠️ More complex setup
- ⚠️ Requires Google Cloud account

---

### 6. AWS (EC2 or Elastic Beanstalk)

**Best for**: Full control and AWS ecosystem integration

**EC2 Setup**:
```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip
git clone https://github.com/QTechDevelopment/littlemoneybuddy.git
cd littlemoneybuddy
pip3 install -r requirements.txt

# Run with screen or systemd
screen -S streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
# Detach with Ctrl+A then D
```

**Pros**:
- ✅ Full control
- ✅ Can use free tier
- ✅ Scalable

**Cons**:
- ⚠️ Manual configuration required
- ⚠️ Need to manage server security
- ⚠️ More DevOps knowledge needed

---

## 🔧 Local Development

Run locally for testing:

```bash
# Clone repository
git clone https://github.com/QTechDevelopment/littlemoneybuddy.git
cd littlemoneybuddy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📊 Platform Comparison

| Platform | Free Tier | Streamlit Support | Complexity | Best For |
|----------|-----------|-------------------|------------|----------|
| **Streamlit Cloud** | ✅ Yes | ⭐⭐⭐ Excellent | 🟢 Easy | Streamlit apps |
| **Railway** | ✅ $5/month | ⭐⭐⭐ Excellent | 🟢 Easy | Quick deploys |
| **Heroku** | ✅ Limited | ⭐⭐ Good | 🟡 Medium | General Python apps |
| **Google Cloud Run** | ✅ Yes | ⭐⭐ Good | 🟡 Medium | Scalable apps |
| **Vercel** | ✅ Yes | ⚠️ Limited | 🟢 Easy | Static/serverless |
| **AWS EC2** | ✅ 12 months | ⭐⭐⭐ Excellent | 🔴 Hard | Full control |

---

## 🌟 Recommended Deployment Path

**For Most Users**:
1. **Streamlit Community Cloud** - Easiest, free, optimized for Streamlit
2. **Railway** - Modern alternative with great Python support
3. **Google Cloud Run** - If you need more control/scaling

**For Vercel Users**:
- Current setup provides information page
- For full app functionality, use Streamlit Community Cloud or Railway
- Vercel is better suited for Next.js, static sites, and serverless functions

---

## 🆘 Troubleshooting

### "Address already in use"
```bash
# Kill existing Streamlit process
pkill -f streamlit
# Or find and kill specific process
lsof -i :8501
kill -9 <PID>
```

### "Module not found"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### "Connection refused"
- Check if firewall allows port 8501
- Ensure server is running with correct address: `--server.address=0.0.0.0`

---

## 📝 Environment Variables

If deploying to platforms that support env vars, you may want to set:

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

---

## 📞 Support

For deployment issues:
- Check [Streamlit deployment docs](https://docs.streamlit.io/streamlit-community-cloud)
- Open issue on GitHub
- Refer to platform-specific documentation

---

**Last Updated**: January 2026
