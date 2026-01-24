# 🚀 Deployment Guide for Little Money Buddy

This guide covers multiple deployment options for testing and production deployment of the Game Theory Stock Agent Dashboard.

## 📋 Prerequisites

Before deploying, ensure:
- All dependencies are listed in `requirements.txt`
- The app runs locally with `streamlit run app.py`
- Git repository is up to date

## 🌐 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended for Testing)

**Easiest and fastest option - FREE tier available!**

#### Steps:

1. **Prepare your repository**
   - Ensure your code is pushed to GitHub
   - Verify `requirements.txt` is complete
   - Commit any pending changes

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select:
     - Repository: `QTechDevelopment/littlemoneybuddy`
     - Branch: `copilot/sub-pr-17` (or your current branch)
     - Main file path: `app.py`
   - Click "Deploy"

3. **App URL**
   - Your app will be available at: `https://[app-name].streamlit.app`
   - Typically: `https://littlemoneybuddy-[random-string].streamlit.app`

4. **Configuration**
   - Streamlit Cloud automatically uses `.streamlit/config.toml`
   - No additional configuration needed

#### Advantages:
✅ Free tier available
✅ Automatic SSL/HTTPS
✅ Easy updates via git push
✅ Built-in monitoring
✅ No server management

---

### Option 2: Docker Deployment

**For more control and local testing**

#### Create Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and run:

```bash
# Build the image
docker build -t littlemoneybuddy:latest .

# Run the container
docker run -p 8501:8501 littlemoneybuddy:latest

# Access at http://localhost:8501
```

#### Deploy to Docker Hub (optional):

```bash
# Tag the image
docker tag littlemoneybuddy:latest yourusername/littlemoneybuddy:latest

# Push to Docker Hub
docker push yourusername/littlemoneybuddy:latest
```

---

### Option 3: Heroku Deployment

**For production-ready hosting**

#### Steps:

1. **Create Heroku-specific files**

Create `setup.sh`:
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

Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. **Deploy to Heroku**

```bash
# Login to Heroku
heroku login

# Create app
heroku create littlemoneybuddy

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku copilot/sub-pr-17:main

# Open app
heroku open
```

#### Advantages:
✅ Professional hosting
✅ Custom domains
✅ Auto-scaling
✅ Add-ons available

---

### Option 4: Railway

**Modern platform with generous free tier**

#### Steps:

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `QTechDevelopment/littlemoneybuddy`
5. Railway auto-detects Python and Streamlit
6. Click "Deploy"

#### Configuration:
- Railway uses `requirements.txt` automatically
- Set start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

---

### Option 5: Render

**Another excellent free option**

#### Steps:

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: littlemoneybuddy
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

2. Go to [render.com](https://render.com)
3. Sign in with GitHub
4. Click "New +" → "Web Service"
5. Connect your repository
6. Render auto-detects configuration
7. Click "Create Web Service"

---

## 🔧 Environment Variables

For production deployments, consider adding:

- `STREAMLIT_SERVER_HEADLESS=true`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
- `STREAMLIT_SERVER_PORT=8501` (or platform-specific)

Most platforms allow setting these through their dashboard.

---

## 🧪 Testing Your Deployment

After deployment, verify:

1. ✅ App loads without errors
2. ✅ Stock data fetching works (try AAPL)
3. ✅ Batch optimization is working
4. ✅ Charts and visualizations render correctly
5. ✅ Agent analysis completes successfully
6. ✅ Excel file upload works (if applicable)

---

## 📊 Monitoring

Once deployed, monitor:
- **Response times**: Should be fast with batch optimization
- **Error rates**: Check platform logs
- **API rate limits**: Yahoo Finance has limits
- **Memory usage**: Multiple stocks can use significant RAM

---

## 🎯 Quick Start (Recommended)

**For immediate testing, use Streamlit Community Cloud:**

1. Push your code: `git push origin copilot/sub-pr-17`
2. Go to: https://share.streamlit.io
3. Deploy in 3 clicks
4. Share the URL for testing

**Total time: ~2 minutes** ⚡

---

## 🆘 Troubleshooting

### App won't start
- Check `requirements.txt` is complete
- Verify Python version compatibility (3.8+)
- Check platform logs for errors

### Stock data not loading
- Yahoo Finance may have rate limits
- Check internet connectivity from host
- Fallback to mock data should work

### Slow performance
- Verify batch optimization is enabled
- Check that caching is working
- Consider upgrading hosting tier

---

## 📞 Support

For deployment issues:
1. Check platform-specific documentation
2. Review platform logs
3. Test locally first: `streamlit run app.py`
4. Verify all dependencies are installed

---

**Ready to deploy?** Start with Streamlit Community Cloud for the quickest path to testing! 🚀
