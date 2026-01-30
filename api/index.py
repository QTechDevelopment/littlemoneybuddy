"""
Vercel serverless function wrapper for Streamlit app
"""
from starlette.applications import Starlette
from starlette.responses import StreamingResponse, HTMLResponse
from starlette.routing import Route
import subprocess
import sys
import os

# Set Streamlit configuration
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

async def homepage(request):
    """Serve a simple homepage with instructions"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Little Money Buddy - Game Theory Stock Agent</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                padding: 40px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { margin-top: 0; }
            .notice {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background: white;
                color: #667eea;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                margin-top: 20px;
            }
            .btn:hover {
                background: #f0f0f0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎮 Little Money Buddy</h1>
            <h2>Game Theory Stock Agent Dashboard</h2>
            
            <div class="notice">
                <h3>⚠️ Deployment Notice</h3>
                <p>This Streamlit application is deployed on Vercel, but Streamlit requires a persistent server to run properly.</p>
                
                <p><strong>Recommended deployment options:</strong></p>
                <ul>
                    <li><strong>Streamlit Community Cloud</strong> - Free hosting optimized for Streamlit apps</li>
                    <li><strong>Heroku</strong> - Supports long-running Python applications</li>
                    <li><strong>Railway</strong> - Modern platform with Python support</li>
                    <li><strong>Google Cloud Run</strong> - Container-based deployment</li>
                </ul>
                
                <p><strong>To run locally:</strong></p>
                <pre style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 5px; overflow-x: auto;">
git clone https://github.com/QTechDevelopment/littlemoneybuddy.git
cd littlemoneybuddy
pip install -r requirements.txt
streamlit run app.py
                </pre>
            </div>
            
            <a href="https://github.com/QTechDevelopment/littlemoneybuddy" class="btn">View on GitHub</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)

routes = [
    Route('/', homepage),
]

app = Starlette(routes=routes)
