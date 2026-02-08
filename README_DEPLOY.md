
DEPLOYMENT GUIDE – RENDER.COM

1. Push to GitHub
-----------------
git init
git add .
git commit -m "AI Fund Manager"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

2. Deploy Dashboard
-------------------
- Go to https://dashboard.render.com
- New + → Web Service
- Connect your GitHub repo
- Environment: Docker
- Plan: Free
- Deploy

3. Deploy Background Worker
---------------------------
- New + → Background Worker
- Use same repository
- Start command is already configured in render.yaml:
  python worker/auto_worker.py
- Plan: Free

4. Usage
--------
- The worker runs autonomously and updates the database every hour.
- Open the Render dashboard URL to view portfolio performance.

You now have a fully autonomous AI fund manager running in the cloud.
