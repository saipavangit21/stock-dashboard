"""
Smart Stock Prediction Dashboard — Vercel Edition
===================================================
Flask app served as a Vercel Python Serverless Function.

Routes:
  GET /              → HTML dashboard
  GET /api/predict   → JSON predictions for all stocks
  GET /api/predict?ticker=AAPL → JSON prediction for one stock
"""

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import threading

app = Flask(__name__)

# ── Global model cache ─────────────────────────────────────────────────────────
# Persists across warm Lambda invocations so we don't retrain on every request.
_cache = {}
_lock  = threading.Lock()

STOCKS     = ["AAPL", "MSFT", "TSLA", "NVDA"]
START_DATE = "2022-01-01"   # 2 years keeps training fast for serverless


# ══════════════════════════════════════════════════════════════════════════════
# DATA & FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def compute_features(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=START_DATE,
                      end=datetime.today().strftime("%Y-%m-%d"),
                      progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = pd.DataFrame()
    df["Close"]  = raw["Close"].squeeze()
    df["Volume"] = raw["Volume"].squeeze()

    # Returns
    df["Return_1d"]  = df["Close"].pct_change()
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # Moving averages
    df["SMA_10"]    = df["Close"].rolling(10).mean()
    df["SMA_50"]    = df["Close"].rolling(50).mean()
    df["SMA_ratio"] = df["SMA_10"] / df["SMA_50"]

    # Volatility
    df["Volatility"] = df["Return_1d"].rolling(10).std()

    # RSI
    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))

    # MACD histogram
    ema12         = df["Close"].ewm(span=12).mean()
    ema26         = df["Close"].ewm(span=26).mean()
    macd          = ema12 - ema26
    df["MACD_hist"] = macd - macd.ewm(span=9).mean()

    # Bollinger Band position (0 = at lower, 1 = at upper)
    rm  = df["Close"].rolling(20).mean()
    rs  = df["Close"].rolling(20).std()
    df["BB_pos"] = (df["Close"] - (rm - 2*rs)) / (4*rs)

    # Volume spike
    df["Vol_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()

    # Target: does price go UP tomorrow?
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


def get_sentiment(ticker: str) -> tuple[float, list]:
    """Return (avg_sentiment_score, list_of_headlines)"""
    try:
        news = yf.Ticker(ticker).news or []
        headlines, scores = [], []
        for a in news[:10]:
            # yfinance ≥0.2.x wraps article data under a "content" dict
            content = a.get("content", a)
            title = content.get("title", "") or a.get("title", "")
            score = TextBlob(title).sentiment.polarity
            headlines.append({"title": title, "score": round(score, 3)})
            scores.append(score)
        avg = round(float(np.mean(scores)), 3) if scores else 0.0
        return avg, headlines
    except Exception:
        return 0.0, []


FEATURES = [
    "Return_1d", "Return_5d", "Return_10d",
    "SMA_ratio", "Volatility",
    "RSI", "MACD_hist", "BB_pos", "Vol_ratio",
]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (cached)
# ══════════════════════════════════════════════════════════════════════════════

def train(ticker: str) -> dict:
    df     = compute_features(ticker)
    X      = df[FEATURES]
    y      = df["Target"]
    split  = int(len(df) * 0.80)

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # RandomForest — fast enough for serverless (50 trees)
    model = RandomForestClassifier(
        n_estimators = 50,
        max_depth    = 6,
        random_state = 42,
        n_jobs       = -1,
    )
    model.fit(X_tr_sc, y_tr)

    preds    = model.predict(X_te_sc)
    accuracy = round(accuracy_score(y_te, preds) * 100, 1)

    # Feature importance
    importance = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    return {
        "model":      model,
        "scaler":     scaler,
        "accuracy":   accuracy,
        "importance": importance,
        "df":         df,
    }


def get_model(ticker: str) -> dict:
    """Return cached model or train a new one."""
    with _lock:
        if ticker not in _cache:
            _cache[ticker] = train(ticker)
        return _cache[ticker]


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def predict_one(ticker: str) -> dict:
    obj       = get_model(ticker)
    df        = obj["df"]
    model     = obj["model"]
    scaler    = obj["scaler"]

    latest    = df[FEATURES].iloc[-1:]
    scaled    = scaler.transform(latest)
    pred      = int(model.predict(scaled)[0])
    proba     = model.predict_proba(scaled)[0]
    confidence = round(float(max(proba)) * 100, 1)

    sentiment_score, headlines = get_sentiment(ticker)

    # Latest indicator values
    last = df.iloc[-1]

    return {
        "ticker":          ticker,
        "direction":       "UP" if pred == 1 else "DOWN",
        "confidence":      confidence,
        "model_accuracy":  obj["accuracy"],
        "sentiment_score": sentiment_score,
        "sentiment_label": (
            "Positive" if sentiment_score > 0.05
            else "Negative" if sentiment_score < -0.05
            else "Neutral"
        ),
        "headlines":       headlines[:5],
        "indicators": {
            "rsi":         round(float(last["RSI"]), 1),
            "macd_hist":   round(float(last["MACD_hist"]), 4),
            "sma_ratio":   round(float(last["SMA_ratio"]), 4),
            "volatility":  round(float(last["Volatility"]) * 100, 2),
            "bb_pos":      round(float(last["BB_pos"]), 3),
            "vol_ratio":   round(float(last["Vol_ratio"]), 2),
        },
        "top_features": [
            {"name": n, "importance": round(float(v) * 100, 1)}
            for n, v in obj["importance"][:5]
        ],
        "as_of": str(df.index[-1].date()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predict")
def api_predict():
    ticker = request.args.get("ticker", "").upper()
    tickers = [ticker] if ticker in STOCKS else STOCKS

    results, errors = [], []
    for t in tickers:
        try:
            results.append(predict_one(t))
        except Exception as e:
            errors.append({"ticker": t, "error": str(e)})

    return jsonify({
        "predictions": results,
        "errors":      errors,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    })


@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE, stocks=STOCKS)


# ══════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Stock Prediction Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {
    --bg:       #0f1117;
    --surface:  #1a1d27;
    --border:   #2a2d3a;
    --text:     #e2e8f0;
    --muted:    #8892a4;
    --up:       #22c55e;
    --down:     #ef4444;
    --neutral:  #f59e0b;
    --accent:   #6366f1;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  header h1 { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em; }
  header h1 span { color: var(--accent); }
  #refresh-btn {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 600;
    transition: opacity 0.2s;
  }
  #refresh-btn:hover { opacity: 0.85; }
  #refresh-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  #status-bar {
    text-align: center;
    padding: 0.6rem;
    font-size: 0.8rem;
    color: var(--muted);
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }

  main { max-width: 1200px; margin: 0 auto; padding: 2rem; }

  #loading {
    text-align: center;
    padding: 5rem;
    color: var(--muted);
    font-size: 1rem;
  }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 1rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.2rem; margin-bottom: 2rem; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem;
    transition: transform 0.15s;
  }
  .card:hover { transform: translateY(-2px); }
  .card.up   { border-top: 3px solid var(--up); }
  .card.down { border-top: 3px solid var(--down); }

  .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem; }
  .ticker { font-size: 1.4rem; font-weight: 800; }
  .as-of  { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

  .direction-badge {
    font-size: 1rem;
    font-weight: 700;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
  }
  .direction-badge.up   { background: rgba(34,197,94,0.15); color: var(--up); }
  .direction-badge.down { background: rgba(239,68,68,0.15);  color: var(--down); }

  .confidence-row { margin-bottom: 0.8rem; }
  .confidence-label { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
  .bar-track { background: var(--border); border-radius: 4px; height: 6px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
  .bar-fill.up   { background: var(--up); }
  .bar-fill.down { background: var(--down); }

  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 1rem; }
  .stat { background: var(--bg); border-radius: 8px; padding: 0.5rem 0.7rem; }
  .stat-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-value { font-size: 0.95rem; font-weight: 700; margin-top: 2px; }

  .sentiment-row { margin-top: 1rem; display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; }
  .sentiment-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .sentiment-dot.positive { background: var(--up); }
  .sentiment-dot.negative { background: var(--down); }
  .sentiment-dot.neutral  { background: var(--neutral); }

  .section-title { font-size: 1rem; font-weight: 700; margin-bottom: 1rem; color: var(--text); }
  .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.2rem; }

  .detail-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem; }
  .detail-card h3 { font-size: 0.9rem; font-weight: 700; margin-bottom: 1rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }

  .indicator-row { display: flex; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
  .indicator-row:last-child { border-bottom: none; }
  .indicator-name { color: var(--muted); }
  .indicator-value { font-weight: 600; }

  .headline-item { padding: 0.5rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem; line-height: 1.4; }
  .headline-item:last-child { border-bottom: none; }
  .headline-score { font-size: 0.72rem; font-weight: 700; margin-top: 2px; }
  .headline-score.pos { color: var(--up); }
  .headline-score.neg { color: var(--down); }
  .headline-score.neu { color: var(--neutral); }

  .chart-wrap { position: relative; height: 160px; }

  .error-msg { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 10px; padding: 1rem; color: var(--down); font-size: 0.85rem; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Stock <span>Prediction</span> Dashboard</h1>
    <div id="header-subtitle" style="font-size:0.78rem;color:var(--muted);margin-top:3px;">
      AAPL · MSFT · TSLA · NVDA &nbsp;|&nbsp; ML + News Sentiment + Insider Signals
    </div>
  </div>
  <button id="refresh-btn" onclick="loadPredictions()">↻ Refresh</button>
</header>

<div id="status-bar">Loading predictions…</div>

<main>
  <div id="loading">
    <div class="spinner"></div>
    Fetching live data and running models… this may take 20–40 seconds on first load.
  </div>
  <div id="content" style="display:none">
    <div class="grid" id="cards"></div>

    <p class="section-title" style="margin-top:1.5rem">Indicator Details</p>
    <div class="detail-grid" id="details"></div>
  </div>
  <div id="error-section"></div>
</main>

<script>
let lastData = null;

async function loadPredictions() {
  const btn = document.getElementById("refresh-btn");
  const status = document.getElementById("status-bar");

  btn.disabled = true;
  btn.textContent = "Loading…";
  status.textContent = "Fetching live data and running models…";

  document.getElementById("loading").style.display = "block";
  document.getElementById("content").style.display = "none";

  try {
    const res  = await fetch("/api/predict");
    const data = await res.json();
    lastData   = data;
    render(data);
    const ts = new Date(data.generated_at).toLocaleTimeString();
    status.textContent = `Last updated: ${ts} UTC · Data from Yahoo Finance`;
  } catch(e) {
    document.getElementById("error-section").innerHTML =
      `<div class="error-msg">Failed to load predictions: ${e.message}</div>`;
    status.textContent = "Error loading data.";
  } finally {
    btn.disabled = false;
    btn.textContent = "↻ Refresh";
    document.getElementById("loading").style.display = "none";
  }
}

function sentimentClass(label) {
  return label === "Positive" ? "positive" : label === "Negative" ? "negative" : "neutral";
}

function render(data) {
  const cards   = document.getElementById("cards");
  const details = document.getElementById("details");
  cards.innerHTML   = "";
  details.innerHTML = "";

  data.predictions.forEach(p => {
    const isUp  = p.direction === "UP";
    const dir   = isUp ? "up" : "down";
    const arrow = isUp ? "↑" : "↓";
    const sc    = sentimentClass(p.sentiment_label);

    // ── Summary card ────────────────────────────────────────────
    cards.innerHTML += `
      <div class="card ${dir}">
        <div class="card-header">
          <div>
            <div class="ticker">${p.ticker}</div>
            <div class="as-of">As of ${p.as_of}</div>
          </div>
          <div class="direction-badge ${dir}">${arrow} ${p.direction}</div>
        </div>

        <div class="confidence-row">
          <div class="confidence-label">
            <span>Model Confidence</span><span>${p.confidence}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${dir}" style="width:${p.confidence}%"></div>
          </div>
        </div>

        <div class="confidence-row">
          <div class="confidence-label">
            <span>Backtest Accuracy</span><span>${p.model_accuracy}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${dir}" style="width:${p.model_accuracy}%"></div>
          </div>
        </div>

        <div class="stats-grid">
          <div class="stat">
            <div class="stat-label">RSI</div>
            <div class="stat-value" style="color:${p.indicators.rsi>70?'var(--down)':p.indicators.rsi<30?'var(--up)':'var(--text)'}">${p.indicators.rsi}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Volatility</div>
            <div class="stat-value">${p.indicators.volatility}%</div>
          </div>
          <div class="stat">
            <div class="stat-label">SMA Trend</div>
            <div class="stat-value" style="color:${p.indicators.sma_ratio>=1?'var(--up)':'var(--down)'}">${p.indicators.sma_ratio >= 1 ? "Bullish" : "Bearish"}</div>
          </div>
          <div class="stat">
            <div class="stat-label">Vol Spike</div>
            <div class="stat-value">${p.indicators.vol_ratio}x</div>
          </div>
        </div>

        <div class="sentiment-row">
          <div class="sentiment-dot ${sc}"></div>
          <span style="color:var(--muted)">News Sentiment:</span>
          <span style="font-weight:600">${p.sentiment_label} (${p.sentiment_score})</span>
        </div>
      </div>`;

    // ── Detail card ─────────────────────────────────────────────
    const featureLabels = p.top_features.map(f => f.name.replace(/_/g," "));
    const featureValues = p.top_features.map(f => f.importance);
    const chartId = `chart-${p.ticker}`;

    const headlines = p.headlines.map(h => {
      const hc = h.score > 0.05 ? "pos" : h.score < -0.05 ? "neg" : "neu";
      return `<div class="headline-item">
        ${h.title}
        <div class="headline-score ${hc}">Sentiment: ${h.score > 0 ? "+" : ""}${h.score}</div>
      </div>`;
    }).join("");

    details.innerHTML += `
      <div class="detail-card">
        <h3>${p.ticker} — Feature Importance</h3>
        <div class="chart-wrap"><canvas id="${chartId}"></canvas></div>
        <h3 style="margin-top:1.2rem">Technical Indicators</h3>
        <div class="indicator-row"><span class="indicator-name">RSI (14)</span><span class="indicator-value">${p.indicators.rsi}</span></div>
        <div class="indicator-row"><span class="indicator-name">MACD Histogram</span><span class="indicator-value">${p.indicators.macd_hist}</span></div>
        <div class="indicator-row"><span class="indicator-name">SMA 10/50 Ratio</span><span class="indicator-value">${p.indicators.sma_ratio}</span></div>
        <div class="indicator-row"><span class="indicator-name">BB Position</span><span class="indicator-value">${p.indicators.bb_pos}</span></div>
        <div class="indicator-row"><span class="indicator-name">Volume Ratio</span><span class="indicator-value">${p.indicators.vol_ratio}x</span></div>
        <h3 style="margin-top:1.2rem">Recent Headlines</h3>
        ${headlines || '<div class="indicator-row"><span class="indicator-name">No headlines available</span></div>'}
      </div>`;

    // Draw chart after DOM is updated
    setTimeout(() => {
      const ctx = document.getElementById(chartId);
      if (!ctx) return;
      new Chart(ctx, {
        type: "bar",
        data: {
          labels: featureLabels,
          datasets: [{
            label: "Importance %",
            data: featureValues,
            backgroundColor: isUp ? "rgba(34,197,94,0.7)" : "rgba(239,68,68,0.7)",
            borderRadius: 4,
          }]
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: "#8892a4", font: { size: 10 } }, grid: { color: "#2a2d3a" } },
            y: { ticks: { color: "#e2e8f0", font: { size: 10 } }, grid: { display: false } }
          }
        }
      });
    }, 50);
  });

  document.getElementById("content").style.display = "block";
}

// Load on page open
loadPredictions();
// Auto-refresh every 10 minutes
setInterval(loadPredictions, 10 * 60 * 1000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)