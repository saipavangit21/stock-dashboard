# Deploy to Vercel — Step by Step

## Prerequisites
- Node.js installed (for Vercel CLI)
- A Vercel account at vercel.com
- Python 3.9+ installed

---

## Option A: Deploy via Vercel CLI (Recommended)

### 1. Install the Vercel CLI
```bash
npm install -g vercel
```

### 2. Navigate into the project folder
```bash
cd stock-dashboard
```

### 3. Login to Vercel
```bash
vercel login
```

### 4. Deploy!
```bash
vercel
```
Follow the prompts:
- Set up and deploy: **Y**
- Which scope: select your account
- Link to existing project: **N**
- Project name: `stock-dashboard` (or anything you like)
- In which directory is your code: **./** (just press Enter)

Your dashboard will be live at a URL like:
`https://stock-dashboard-xyz.vercel.app`

### 5. Deploy to production
```bash
vercel --prod
```

---

## Option B: Deploy via GitHub

1. Push this `stock-dashboard` folder to a GitHub repository
2. Go to vercel.com → New Project
3. Import your GitHub repo
4. Vercel auto-detects the Python config — just click **Deploy**

---

## Project Structure

```
stock-dashboard/
├── api/
│   └── index.py       ← Flask app (model + API + HTML dashboard)
├── vercel.json        ← Routes all traffic to the Flask app
├── requirements.txt   ← Python dependencies
└── HOW_TO_DEPLOY.md   ← This file
```

---

## Test Locally Before Deploying

```bash
pip install -r requirements.txt
python api/index.py
```
Then open: http://localhost:5000

---

## Notes

- **First load is slow** (~20–40s) because the model trains fresh on first request.
  Subsequent requests are fast (model is cached in memory).
- Vercel Hobby plan has a **10s function timeout**. If training times out, upgrade
  to Vercel Pro (60s timeout) or reduce `START_DATE` in `api/index.py` to a more
  recent date like `"2023-01-01"`.
- The dashboard auto-refreshes every **10 minutes**.
- The `/api/predict` endpoint also works standalone — e.g.:
  `https://your-app.vercel.app/api/predict?ticker=AAPL`