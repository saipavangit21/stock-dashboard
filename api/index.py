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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime
import threading

app = Flask(__name__)

# ── Global model cache ─────────────────────────────────────────────────────────
_cache      = {}
_cache_date = {}
_lock       = threading.Lock()

STOCKS     = [
    # My Portfolio
    "CSWC", "GME", "CCEC", "VOO", "NVAX", "NIO", "RXRX", "BDMD",
    # Watchlist
    "AAPL", "MSFT", "TSLA", "NVDA", "^NSEI",
]
START_DATE = "2022-01-01"   # 2 years keeps training fast for serverless
SR_START   = "2018-01-01"   # 6+ years of history for support/resistance

# ── Market scan universes ───────────────────────────────────────────────────────
US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "XOM", "CVX", "PFE", "JNJ", "UNH",
    "SPY", "QQQ", "ARKK", "BDMD",
]
SURGE_UNIVERSE = [
    # Mega-cap tech (always liquid, options-heavy)
    "AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA", "NFLX",
    # High-beta AI / semiconductors
    "AMD", "AVGO", "QCOM", "MU", "INTC", "ARM", "SMCI", "PLTR",
    "IONQ", "RGTI", "QUBT", "SOUN", "BBAI", "AI", "GTLB",
    # Crypto & fintech
    "COIN", "MSTR", "MARA", "RIOT", "CLSK", "BTBT", "HOOD", "SOFI",
    "XYZ", "PYPL", "AFRM", "UPST",
    # EV / energy
    "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "BLNK", "CHPT",
    "ENPH", "FSLR", "SEDG",
    # Biotech / healthcare
    "HIMS", "RXRX", "BEAM", "CRSP", "EDIT", "NTLA", "PACB",
    "MRNA", "BNTX", "NVAX", "SGEN", "RCKT",
    # Space / defense / deep tech
    "RKLB", "LUNR", "ACHR", "JOBY", "LILM", "RCAT", "ASTS",
    # Consumer / retail high-beta
    "GME", "AMC", "BBBY", "SNAP", "PINS", "UBER", "LYFT", "ABNB",
    "DASH", "ETSY", "CHWY", "W",
    # Software / cloud
    "SNOW", "DDOG", "NET", "CRWD", "ZS", "OKTA", "MDB", "GTLB",
    "U", "RBLX", "PATH", "ASAN", "CFLT",
    # Healthcare / weight-loss / trending
    "LLY", "NVO", "HIMS", "NTRA", "OSCR",
    # Industrials / materials high-momentum
    "CLF", "X", "FCX", "AA", "MP",
    # Banks / financials (volatile around earnings)
    "BAC", "C", "WFC", "GS", "MS", "JPM",
]
TREND_UNIVERSE = [
    # Specialty pharma / biotech uptrends
    "HIMS", "VKTX", "TGTX", "ACAD", "PRAX", "IMVT", "RDUS", "NVCR",
    "EXAS", "NTRA", "GHDX", "VCYT", "CERT", "RARE", "FOLD", "ARGX",
    "DAWN", "CERE", "MRUS", "INVA", "RVMD", "KYMR", "PCVX", "ARQT",
    # Medical devices
    "IRTC", "AXNX", "NURO", "INSP", "SWAV", "TNDM", "DXCM", "PODD",
    # Nuclear / clean energy
    "NNE", "OKLO", "SMR", "BWXT", "LEU", "UUUU", "CCJ", "DNN",
    # Defense / drone tech
    "KTOS", "AVAV", "HWM", "TDG", "DRS", "RKLB", "ASTS", "RCAT",
    # AI / cloud small-mid cap
    "MNDY", "TTD", "HUBS", "DOCN", "BILL", "ZI", "AEHR", "AMBA",
    "CELH", "SMCI", "ANET", "FTNT", "PANW",
    # Weight-loss / GLP-1
    "LLY", "NVO", "HIMS", "ZFOX", "NTRA", "OSCR",
    # Industrials momentum
    "HWM", "TDG", "WWD", "AXON", "TYL", "CACI", "LDOS", "BAH",
    # Small-cap consumer / retail momentum
    "ELF", "CELH", "USFD", "PFGC", "CAVA", "SHAK", "BROS",
    # Financials momentum
    "HOOD", "SOFI", "AFRM", "UPST", "CSWC", "ARCC", "HTGC",
    # Materials / commodities momentum
    "MP", "USLM", "STLD", "NUE", "RS", "FCX", "RGLD", "WPM",
]

INDIA_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "WIPRO.NS", "HCLTECH.NS", "AXISBANK.NS", "BAJFINANCE.NS", "SBIN.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "DRREDDY.NS",
    "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "ADANIENT.NS", "LT.NS",
    "TITAN.NS",
]


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
    df["High"]   = raw["High"].squeeze()
    df["Low"]    = raw["Low"].squeeze()
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


_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
_POS_WORDS = {"surge","soar","beat","record","growth","profit","rally","upgrade",
              "bullish","gain","jump","rise","strong","high","boom","buy","positive"}
_NEG_WORDS = {"fall","drop","miss","loss","cut","downgrade","bearish","decline",
              "risk","warn","weak","crash","sell","negative","low","concern","fear"}

def get_sentiment(ticker: str) -> tuple[float, list]:
    """
    Return (avg_sentiment_score, list_of_headlines).
    Uses FinBERT via HF Inference API when HF_API_TOKEN env var is set;
    falls back to keyword scoring otherwise.
    """
    import os, requests as _req
    try:
        news = yf.Ticker(ticker).news or []
        titles = []
        for a in news[:8]:
            title = (a.get("title")
                     or a.get("content", {}).get("title")
                     or a.get("headline") or "")
            if title:
                titles.append(title)
        if not titles:
            return 0.0, []

        hf_token = os.environ.get("HF_API_TOKEN", "")
        if hf_token:
            try:
                r = _req.post(
                    _FINBERT_URL,
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": titles},
                    timeout=12,
                )
                results = r.json()
                # Handle model-still-loading response
                if isinstance(results, dict) and "error" in results:
                    raise ValueError(results["error"])
                scores, headlines = [], []
                for i, preds in enumerate(results):
                    if not isinstance(preds, list):
                        continue
                    best = max(preds, key=lambda x: x["score"])
                    s = (best["score"]  if best["label"] == "positive" else
                         -best["score"] if best["label"] == "negative" else 0.0)
                    scores.append(s)
                    headlines.append({"title": titles[i], "score": round(s, 3),
                                      "label": best["label"]})
                avg = round(float(np.mean(scores)), 3) if scores else 0.0
                return avg, headlines
            except Exception:
                pass  # fall through to keyword fallback

        # Keyword fallback (no token or FinBERT unavailable)
        scores, headlines = [], []
        for t in titles:
            words = set(t.lower().split())
            pos = len(words & _POS_WORDS)
            neg = len(words & _NEG_WORDS)
            s = round(max(-1.0, min(1.0, (pos - neg) / max(len(words), 1) * 4)), 3)
            scores.append(s)
            headlines.append({"title": t, "score": s})
        avg = round(float(np.mean(scores)), 3) if scores else 0.0
        return avg, headlines
    except Exception:
        return 0.0, []


def compute_support_resistance(highs, lows, current_price: float) -> dict:
    """
    Core S/R computation from pre-downloaded High/Low series.
    Extracted so scan_ticker can reuse its already-downloaded data.
    """
    try:
        window = 10   # bars to look each side when detecting a swing

        swing_highs, swing_lows = [], []

        for i in range(window, len(highs) - window):
            h = highs.iloc[i]
            l = lows.iloc[i]
            if h == highs.iloc[i - window: i + window + 1].max():
                swing_highs.append(float(h))
            if l == lows.iloc[i - window: i + window + 1].min():
                swing_lows.append(float(l))

        def cluster(levels: list, tolerance: float = 0.02) -> list:
            """Group price levels within `tolerance` % of each other."""
            if not levels:
                return []
            levels = sorted(levels)
            clusters, group = [], [levels[0]]
            for price in levels[1:]:
                if (price - group[0]) / group[0] <= tolerance:
                    group.append(price)
                else:
                    clusters.append({
                        "price":   round(float(np.mean(group)), 2),
                        "touches": len(group),
                    })
                    group = [price]
            clusters.append({"price": round(float(np.mean(group)), 2), "touches": len(group)})
            return clusters

        resistances = cluster(swing_highs)
        supports    = cluster(swing_lows)

        # Keep only levels on the correct side of current price
        # Require at least 0.5% distance so we don't show levels AT the current price
        resistances = [r for r in resistances if r["price"] / current_price - 1 >= 0.005]
        supports    = [s for s in supports    if 1 - s["price"] / current_price >= 0.005]

        # Filter extreme outliers beyond 40% — those are from a different price era
        resistances = [r for r in resistances if r["price"] / current_price - 1 <= 0.40]
        supports    = [s for s in supports    if 1 - s["price"] / current_price <= 0.40]

        resistances = sorted(resistances, key=lambda x: x["price"])[:3]
        supports    = sorted(supports,    key=lambda x: x["price"], reverse=True)[:3]

        # Add % distance from current price
        for r in resistances:
            r["pct_away"] = round((r["price"] - current_price) / current_price * 100, 2)
        for s in supports:
            s["pct_away"] = round((current_price - s["price"]) / current_price * 100, 2)

        return {"support": supports, "resistance": resistances}

    except Exception as e:
        return {"support": [], "resistance": []}


def get_support_resistance(ticker: str, current_price: float) -> dict:
    """Download full price history and compute S/R levels."""
    try:
        raw = yf.Ticker(ticker).history(period="max")
        if raw.empty:
            return {"support": [], "resistance": []}
        return compute_support_resistance(
            raw["High"], raw["Low"], current_price
        )
    except Exception:
        return {"support": [], "resistance": []}


def get_market_regime(index_ticker: str) -> dict:
    """
    Determine broad market regime from an index (SPY for US, ^NSEI for India).
    Uses SMA trend, RSI, and recent momentum to classify:
      bullish  — trend up, momentum positive
      bearish  — trend down, momentum negative
      ranging  — mixed signals
      rally    — oversold bounce (sharp up move from low base)
      selloff  — overbought collapse (sharp down from high base)
    """
    try:
        raw   = yf.Ticker(index_ticker).history(period="1y")
        close = raw["Close"].dropna()

        if len(close) < 52:
            return {"regime": "unknown", "label": "Unknown", "desc": "Not enough data", "color": "neutral"}

        sma20   = float(close.rolling(20).mean().dropna().iloc[-1])
        sma50   = float(close.rolling(50).mean().dropna().iloc[-1])
        price   = float(close.iloc[-1])
        ret5    = float(close.pct_change(5).dropna().iloc[-1] * 100)
        ret1    = float(close.pct_change(1).dropna().iloc[-1] * 100)

        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi   = float((100 - (100 / (1 + gain / loss))).dropna().iloc[-1])

        above_sma20 = price > sma20
        above_sma50 = price > sma50

        # Classify regime
        if rsi < 35 and ret5 < -5 and ret1 > 1.5:
            regime = "rally"
            label  = "Relief Rally"
            desc   = "Market bouncing from oversold — technical signals unreliable, momentum overrides"
            color  = "neutral"
        elif rsi > 65 and ret5 > 5:
            regime = "overbought"
            label  = "Overbought / Extended"
            desc   = "Market extended after a run — sell signals stronger, buy signals risky"
            color  = "down"
        elif above_sma20 and above_sma50 and rsi > 50:
            regime = "bullish"
            label  = "Bullish Trend"
            desc   = "Market in uptrend — buy signals more reliable, sell signals against the tide"
            color  = "up"
        elif not above_sma20 and not above_sma50 and rsi < 50:
            regime = "bearish"
            label  = "Bearish Trend"
            desc   = "Market in downtrend — sell signals more reliable, buy signals are counter-trend"
            color  = "down"
        elif rsi < 40 and ret5 < -3:
            regime = "selloff"
            label  = "Active Selloff"
            desc   = "Market selling off — avoid buying dips until stabilisation, no bottom confirmed"
            color  = "down"
        else:
            regime = "ranging"
            label  = "Ranging / Neutral"
            desc   = "No clear trend — signals have lower reliability in either direction"
            color  = "neutral"

        return {
            "regime":  regime,
            "label":   label,
            "desc":    desc,
            "color":   color,
            "rsi":     round(rsi, 1),
            "ret5":    round(ret5, 2),
            "ret1":    round(ret1, 2),
            "price":   round(price, 2),
            "vs_sma20": round((price / sma20 - 1) * 100, 2),
            "vs_sma50": round((price / sma50 - 1) * 100, 2),
        }
    except Exception:
        return {"regime": "unknown", "label": "Unknown", "desc": "Could not determine market regime", "color": "neutral"}


def get_pcr(ticker: str) -> float:
    """
    Put/Call Ratio from nearest options expiry.
    PCR > 1.2 → fear/oversold (contrarian BUY signal)
    PCR < 0.7 → greed/overbought (contrarian SELL signal)
    Returns None if options not available (e.g. Indian stocks).
    """
    try:
        tk   = yf.Ticker(ticker)
        exp  = tk.options
        if not exp:
            return None
        chain = tk.option_chain(exp[0])
        put_oi  = chain.puts["openInterest"].sum()
        call_oi = chain.calls["openInterest"].sum()
        if call_oi == 0:
            return None
        return round(float(put_oi / call_oi), 3)
    except Exception:
        return None


def scan_ticker(ticker: str) -> dict | None:
    """
    Lightweight technical score for market scanning (no ML training).
    Returns a dict with buy_score, sell_score, and key indicators.
    """
    try:
        raw = yf.Ticker(ticker).history(period="3y")
        if raw.empty or len(raw) < 60:
            return None

        close  = raw["Close"]
        volume = raw["Volume"]

        # Sanity check: latest close must be within 3x / 0.33x of its own 60-day median
        # Catches yfinance returning unadjusted/wrong price series
        median60 = float(close.iloc[-60:].median())
        latest   = float(close.iloc[-1])
        if median60 > 0 and not (median60 * 0.33 < latest < median60 * 3.0):
            return None   # data is clearly corrupted, skip this ticker

        # ── Indicators ──────────────────────────────────────────────
        ret1  = float(close.pct_change().iloc[-1])
        ret5  = float(close.pct_change(5).iloc[-1])

        sma10 = close.rolling(10).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        sma_ratio = float(sma10 / sma50)

        delta    = close.diff()
        gain     = delta.where(delta > 0, 0).rolling(14).mean()
        loss     = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_series = 100 - (100 / (1 + gain / loss))
        rsi      = float(rsi_series.iloc[-1])

        ema12     = close.ewm(span=12).mean()
        ema26     = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        sig_line  = macd_line.ewm(span=9).mean()
        macd_hist_s = (macd_line - sig_line).dropna()
        macd_hist   = float(macd_hist_s.iloc[-1])
        prev_macd   = float(macd_hist_s.iloc[-2]) if len(macd_hist_s) >= 2 else macd_hist

        rm  = close.rolling(20).mean()
        rs  = close.rolling(20).std()
        bb_series = (close - (rm - 2*rs)) / (4*rs)
        bb_pos    = float(bb_series.iloc[-1])

        vol_ratio = float((volume / volume.rolling(10).mean()).iloc[-1])
        price     = round(float(close.iloc[-1]), 2)
        sma200    = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(sma50)

        pcr = get_pcr(ticker)

        # ── Traditional buy/sell scoring ────────────────────────────
        buy = sell = 0.0

        # RSI absolute thresholds
        if   rsi < 25:  buy  += 3.0
        elif rsi < 30:  buy  += 2.0
        elif rsi < 40:  buy  += 1.0
        elif rsi > 75:  sell += 3.0
        elif rsi > 70:  sell += 2.0
        elif rsi > 60:  sell += 0.5

        # MACD histogram direction
        if macd_hist > 0:  buy  += 1.5
        else:              sell += 1.5
        # Fresh crossover bonus
        if macd_hist > 0 and prev_macd <= 0:  buy  += 1.5
        if macd_hist < 0 and prev_macd >= 0:  sell += 1.5

        # Price vs SMA50 and SMA200 (trend structure)
        if   price > sma50 and sma50 > sma200:  buy  += 2.0
        elif price > sma50:                      buy  += 1.0
        elif price < sma50 and sma50 < sma200:  sell += 2.0
        elif price < sma50:                      sell += 1.0

        # Bollinger Band position
        if   bb_pos < 0.1:  buy  += 1.5
        elif bb_pos > 0.9:  sell += 1.5

        # Volume confirmation
        if vol_ratio > 2.0:
            if macd_hist > 0: buy  += 1.0
            else:             sell += 1.0
        elif vol_ratio > 1.5 and macd_hist > 0:
            buy += 0.5

        # Sharp 5-day dip = potential reversal
        if ret5 < -0.08:   buy  += 1.5
        elif ret5 < -0.05: buy  += 1.0
        # Extended 5-day run = stretched
        if ret5 > 0.10:    sell += 1.0

        # PCR options sentiment
        if pcr is not None:
            if   pcr > 1.3:  buy  += 1.0
            elif pcr < 0.7:  sell += 1.0

        # Reuse already-downloaded data — avoids a second yfinance call per ticker
        sr = compute_support_resistance(
            raw["High"].squeeze(), raw["Low"].squeeze(), price
        )

        return {
            "ticker":     ticker,
            "price":      price,
            "buy_score":  round(buy, 2),
            "sell_score": round(sell, 2),
            "rsi":        round(rsi, 1),
            "bb_pos":     round(bb_pos, 3),
            "macd_hist":  round(macd_hist, 4),
            "vol_ratio":  round(vol_ratio, 2),
            "ret5":       round(ret5 * 100, 2),
            "pcr":        pcr,
            "support":    sr["support"],
            "resistance": sr["resistance"],
        }
    except Exception:
        return None


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
    """Return cached model, retraining if cache is from a previous day."""
    today = datetime.today().date()
    with _lock:
        if ticker not in _cache or _cache_date.get(ticker) != today:
            _cache[ticker]      = train(ticker)
            _cache_date[ticker] = today
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

    # Fetch live price — don't use cached training data's last close
    # (cache can be hours/days old on warm serverless instances)
    try:
        live = yf.Ticker(ticker).history(period="1d")
        current_price = round(float(live["Close"].iloc[-1]), 2) if not live.empty else round(float(last["Close"]), 2)
    except Exception:
        current_price = round(float(last["Close"]), 2)

    # Historical support & resistance (from 6+ years of swing data)
    sr = get_support_resistance(ticker, current_price)

    return {
        "ticker":          ticker,
        "direction":       "UP" if pred == 1 else "DOWN",
        "confidence":      confidence,
        "model_accuracy":  obj["accuracy"],
        "current_price":   current_price,
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
        "support":    sr["support"],
        "resistance": sr["resistance"],
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

    return jsonify(sanitize({
        "predictions": results,
        "errors":      errors,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


def sanitize(obj):
    """Recursively replace NaN/Inf floats with None so JSON stays valid."""
    if isinstance(obj, float):
        return None if (obj != obj or obj == float("inf") or obj == float("-inf")) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


@app.route("/api/scan")
def api_scan():
    """Scan US and India universes and return top buy/sell picks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def scan_group(universe):
        results = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(scan_ticker, t): t for t in universe}
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)
        return results

    us_results     = scan_group(US_UNIVERSE)
    india_results  = scan_group(INDIA_UNIVERSE)

    def pick(results, regime):
        if not results:
            return None, None, [], []

        r = regime.get("regime", "ranging")
        # Lower buy threshold in bullish markets (momentum signals fire at lower scores)
        # Lower sell threshold in bearish markets
        min_buy  = 3.0 if r in ("bullish", "rally")          else 4.0
        min_sell = 3.0 if r in ("bearish", "selloff", "overbought") else 4.0

        by_buy  = sorted(results, key=lambda x: x["buy_score"],  reverse=True)
        by_sell = sorted(results, key=lambda x: x["sell_score"], reverse=True)

        best_buy  = by_buy[0]  if by_buy[0]["buy_score"]   >= min_buy  else None
        best_sell = by_sell[0] if by_sell[0]["sell_score"] >= min_sell else None

        # Avoid showing the same ticker as both buy and sell
        if best_buy and best_sell and best_buy["ticker"] == best_sell["ticker"]:
            if best_buy["buy_score"] >= best_sell["sell_score"]:
                best_sell = next((r for r in by_sell[1:] if r["sell_score"] >= min_sell), None)
            else:
                best_buy  = next((r for r in by_buy[1:]  if r["buy_score"]  >= min_buy),  None)

        # Always return top 3 watch candidates with S/R regardless of threshold
        watch_buy  = [r for r in by_buy[:3]  if not best_buy  or r["ticker"] != best_buy["ticker"]][:3]
        watch_sell = [r for r in by_sell[:3] if not best_sell or r["ticker"] != best_sell["ticker"]][:3]

        return best_buy, best_sell, watch_buy, watch_sell

    us_regime    = get_market_regime("SPY")
    india_regime = get_market_regime("^NSEI")

    us_buy,    us_sell,    us_watch_buy,    us_watch_sell    = pick(us_results,    us_regime)
    india_buy, india_sell, india_watch_buy, india_watch_sell = pick(india_results, india_regime)

    return jsonify(sanitize({
        "us":    {"buy": us_buy,    "sell": us_sell,    "watch_buy": us_watch_buy,    "watch_sell": us_watch_sell,    "scanned": len(us_results),    "regime": us_regime},
        "india": {"buy": india_buy, "sell": india_sell, "watch_buy": india_watch_buy, "watch_sell": india_watch_sell, "scanned": len(india_results), "regime": india_regime},
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


@app.route("/api/surge")
def api_surge():
    """
    Pre-surge detector: finds stocks BEFORE they move 20-30%, not after.
    Signals: Bollinger Band squeeze (volatility compression), ATR coiling,
    neutral RSI (room to run), volume just starting to build, call OI buildup.
    Stocks that already ran are penalised heavily.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def scan_surge(ticker):
        try:
            raw = yf.download(ticker, period="3mo", progress=False)
            if raw.empty or len(raw) < 30:
                return None

            close = raw["Close"].squeeze()
            high  = raw["High"].squeeze()
            low   = raw["Low"].squeeze()
            vol   = raw["Volume"].squeeze()

            price = round(float(close.iloc[-1]), 2)
            if price < 1.0:
                return None

            # RSI
            delta = close.diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi   = round(float((100 - (100 / (1 + gain / loss))).iloc[-1]), 1)

            # Bollinger Band squeeze: 10d std vs 30d std
            # < 0.7 means bands are tightening → energy building for a breakout
            bb_std_now = float(close.rolling(10).std().iloc[-1])
            bb_std_ref = float(close.rolling(30).std().iloc[-1])
            bb_squeeze = round(bb_std_now / bb_std_ref, 3) if bb_std_ref > 0 else 1.0

            # ATR compression: recent 5d ATR vs 20d ATR
            # < 0.7 means price is coiling in a tight range
            atr_recent = float((high - low).rolling(5).mean().iloc[-1])
            atr_normal = float((high - low).rolling(20).mean().iloc[-1])
            atr_ratio  = round(atr_recent / atr_normal, 3) if atr_normal > 0 else 1.0

            # Volume: today vs 20d baseline (exclude today from avg to avoid self-reference)
            vol_today = float(vol.iloc[-1])
            vol_20avg = float(vol.rolling(20).mean().iloc[-2])
            vol_ratio = round(vol_today / vol_20avg, 2) if vol_20avg > 0 else 1.0

            # Returns
            ret1 = round(float((close.iloc[-1] / close.iloc[-2]  - 1) * 100), 2)
            ret5 = round(float((close.iloc[-1] / close.iloc[-6]  - 1) * 100), 2) if len(close) > 6 else 0.0

            # ── Directional confirmation: price vs moving averages ───────────
            sma20  = float(close.rolling(20).mean().iloc[-1])
            sma50  = float(close.rolling(50).mean().iloc[-1])
            above_sma20 = price > sma20
            above_sma50 = price > sma50

            # ── MACD for momentum direction ──────────────────────────────────
            ema12    = close.ewm(span=12).mean()
            ema26    = close.ewm(span=26).mean()
            macd_sig = (ema12 - ema26).ewm(span=9).mean()
            macd_hist = float((ema12 - ema26 - macd_sig).iloc[-1])
            macd_prev = float((ema12 - ema26 - macd_sig).iloc[-2])

            # ── Higher lows check: last 3 lows trending up (accumulation) ────
            recent_lows = [float(low.iloc[-i]) for i in range(1, 6)]
            higher_lows = recent_lows[0] > recent_lows[2] > recent_lows[4]

            score   = 0
            reasons = []

            # ── DIRECTIONAL GATE: must show bullish bias to qualify ──────────
            bullish_signals = sum([
                above_sma20,
                above_sma50,
                macd_hist > 0,
                higher_lows,
            ])
            if bullish_signals < 2:
                return None  # squeeze with bearish bias — skip it

            # ── Volatility compression (primary signal) ─────────────────────
            if bb_squeeze < 0.55:
                score += 4; reasons.append("BB squeeze")
            elif bb_squeeze < 0.70:
                score += 3; reasons.append("BB squeeze")
            elif bb_squeeze < 0.80:
                score += 2; reasons.append("BB tightening")
            elif bb_squeeze < 0.90:
                score += 1

            # ── ATR coiling ──────────────────────────────────────────────────
            if atr_ratio < 0.55:
                score += 3; reasons.append("coiling")
            elif atr_ratio < 0.70:
                score += 2; reasons.append("coiling")
            elif atr_ratio < 0.80:
                score += 1

            # ── Volume starting to build (not yet exploding) ─────────────────
            if 1.4 <= vol_ratio < 2.5:
                score += 2; reasons.append(f"vol {vol_ratio}x")
            elif vol_ratio >= 2.5:
                score += 1; reasons.append(f"vol {vol_ratio}x")
            elif vol_ratio >= 1.2:
                score += 1

            # ── RSI neutral sweet spot: room to run upward ───────────────────
            if 42 <= rsi <= 62:
                score += 2; reasons.append(f"RSI {rsi}")
            elif 35 <= rsi < 42 or 62 < rsi <= 68:
                score += 1; reasons.append(f"RSI {rsi}")

            # ── Bullish direction bonuses ─────────────────────────────────────
            if above_sma20 and above_sma50:
                score += 2; reasons.append("above MAs")
            elif above_sma20:
                score += 1; reasons.append("above SMA20")
            if macd_hist > 0 and macd_prev <= 0:
                score += 2; reasons.append("MACD cross↑")
            elif macd_hist > 0:
                score += 1; reasons.append("MACD bull")
            if higher_lows:
                score += 1; reasons.append("higher lows")

            # ── Hard penalties: already moved / overbought ───────────────────
            if rsi > 78:
                score -= 5
            elif rsi > 72:
                score -= 3
            elif rsi > 68:
                score -= 1

            if ret5 > 20:
                score -= 6
            elif ret5 > 12:
                score -= 4
            elif ret5 > 7:
                score -= 2
            elif ret5 > 4:
                score -= 1

            # ── Call OI buildup (smart money positioning ahead of move) ──────
            call_put_ratio = None
            tk_obj = yf.Ticker(ticker)
            try:
                exps = tk_obj.options
                if exps:
                    chain   = tk_obj.option_chain(exps[0])
                    call_oi = int(chain.calls["openInterest"].sum())
                    put_oi  = int(chain.puts["openInterest"].sum())
                    if put_oi > 0:
                        call_put_ratio = round(call_oi / put_oi, 2)
                        if call_put_ratio > 2.5:
                            score += 2; reasons.append(f"C/P {call_put_ratio}x")
                        elif call_put_ratio > 1.5:
                            score += 1; reasons.append(f"C/P {call_put_ratio}x")
            except Exception:
                pass

            # ── Earnings catalyst check ──────────────────────────────────────
            days_to_earnings = None
            try:
                cal = tk_obj.calendar
                if cal is not None:
                    ed = cal.get("Earnings Date") or cal.get("earnings_date")
                    if ed is None and hasattr(cal, "iloc"):
                        # calendar can be a DataFrame
                        ed = cal.loc["Earnings Date"].dropna().tolist() if "Earnings Date" in cal.index else None
                    if ed:
                        if not isinstance(ed, list):
                            ed = [ed]
                        from datetime import date as _date
                        today = _date.today()
                        upcoming = [e for e in ed if hasattr(e, "date") and e.date() >= today]
                        if upcoming:
                            days_to_earnings = (upcoming[0].date() - today).days
            except Exception:
                pass

            # ── ATR-based stop-loss ──────────────────────────────────────────
            atr14      = float((high - low).rolling(14).mean().iloc[-1])
            stop_loss  = round(price - 2.0 * atr14, 2)
            stop_pct   = round((price - stop_loss) / price * 100, 1)

            if score < 4:
                return None

            return {
                "ticker":            ticker,
                "price":             price,
                "score":             score,
                "rsi":               rsi,
                "bb_squeeze":        bb_squeeze,
                "atr_ratio":         atr_ratio,
                "vol_ratio":         vol_ratio,
                "vol_today":         int(vol_today),
                "vol_20avg":         int(vol_20avg),
                "macd_bull":         macd_hist > 0,
                "above_sma20":       above_sma20,
                "above_sma50":       above_sma50,
                "ret5":              ret5,
                "ret1":              ret1,
                "call_put_ratio":    call_put_ratio,
                "stop_loss":         stop_loss,
                "stop_pct":          stop_pct,
                "days_to_earnings":  days_to_earnings,
                "reasons":           ", ".join(reasons) if reasons else "setup forming",
            }
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(scan_surge, t): t for t in set(SURGE_UNIVERSE)}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)

    return jsonify(sanitize({
        "surges":       results[:7],
        "scanned":      len(set(SURGE_UNIVERSE)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


MY_PORTFOLIO = {
    #          entry    invested($)
    "CSWC":    (23.56,  200),
    "VUAA.DE": (114.05, 115),   # Vanguard S&P 500 UCITS ETF
    "GME":     (25.19,  200),
    "CCEC":    (22.00,  200),
    "XNDU":    (14.80,  139),   # Xanadu Quantum
    "NVAX":    (8.22,   100),
    "NIO":     (6.24,   100),
    "RXRX":    (3.54,   100),
    "BDMD":    (3.09,   100),
    "PTP.DE":  (3.53,    27),   # Pentixapharm (Frankfurt)
    "DFTK.DE": (6.03,    43),   # DFTK (Frankfurt)
}

@app.route("/api/portfolio")
def api_portfolio():
    """Live P/L with real $ and € amounts based on investment size."""
    # USD → EUR rate
    try:
        usdeur = float(yf.Ticker("USDEUR=X").fast_info.last_price)
    except Exception:
        usdeur = 0.90

    results = []
    total_invested = total_current = 0.0

    for ticker, (entry, invested) in MY_PORTFOLIO.items():
        try:
            info    = yf.Ticker(ticker).fast_info
            current = round(float(info.last_price), 3)
            pre     = None
            try:
                p = info.pre_market_price
                if p and abs(float(p) - current) > 0.001:
                    pre = round(float(p), 3)
            except Exception:
                pass

            shares   = invested / entry
            cur_val  = shares * current
            pnl_usd  = round(cur_val - invested, 2)
            pnl_pct  = round(pnl_usd / invested * 100, 2)
            pnl_eur  = round(pnl_usd * usdeur, 2)

            results.append({
                "ticker":    ticker,
                "entry":     entry,
                "current":   current,
                "invested":  invested,
                "shares":    round(shares, 3),
                "cur_val":   round(cur_val, 2),
                "pnl_usd":   pnl_usd,
                "pnl_eur":   pnl_eur,
                "pnl_pct":   pnl_pct,
                "pre":       pre,
            })
            total_invested += invested
            total_current  += cur_val
        except Exception as e:
            results.append({"ticker": ticker, "entry": entry, "invested": invested,
                            "pnl_usd": None, "pnl_eur": None, "pnl_pct": None, "pre": None})

    total_pnl_usd = round(total_current - total_invested, 2)
    total_pnl_eur = round(total_pnl_usd * usdeur, 2)
    total_pct     = round(total_pnl_usd / total_invested * 100, 2) if total_invested else 0

    return jsonify(sanitize({
        "positions":       results,
        "total_invested":  round(total_invested, 2),
        "total_pnl_usd":   total_pnl_usd,
        "total_pnl_eur":   total_pnl_eur,
        "total_pct":       total_pct,
        "usdeur":          round(usdeur, 4),
        "generated_at":    datetime.utcnow().isoformat() + "Z",
    }))


@app.route("/api/movers")
def api_movers():
    """
    Today's top gainers from Yahoo Finance's real-time screener.
    Catches stocks already moving 10%+ with unusual volume — news-driven
    micro/small-cap movers that can't be predicted in advance.
    """
    import requests as _req
    try:
        r = _req.get(
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
            params={"formatted": "false", "lang": "en-US", "region": "US",
                    "scrIds": "day_gainers", "count": 100},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        quotes = r.json()["finance"]["result"][0]["quotes"]

        movers = []
        for q in quotes:
            chg     = q.get("regularMarketChangePercent", 0)
            volume  = q.get("regularMarketVolume", 0)
            avg_vol = q.get("averageDailyVolume3Month", 1) or 1
            vol_ratio = round(volume / avg_vol, 1)
            mktcap  = q.get("marketCap", 0)

            if chg < 10 or vol_ratio < 2:
                continue

            def fmt_cap(c):
                if not c: return "—"
                if c >= 1e9: return f"${c/1e9:.1f}B"
                if c >= 1e6: return f"${c/1e6:.0f}M"
                return f"${c/1e3:.0f}K"

            movers.append({
                "ticker":     q.get("symbol", ""),
                "name":       q.get("shortName", ""),
                "price":      round(q.get("regularMarketPrice", 0), 2),
                "change_pct": round(chg, 1),
                "volume":     volume,
                "vol_ratio":  vol_ratio,
                "mktcap":     fmt_cap(mktcap),
                "mktcap_raw": mktcap,
            })

        movers.sort(key=lambda x: x["change_pct"], reverse=True)
        return jsonify(sanitize({
            "movers":       movers[:25],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/trending")
def api_trending():
    """
    Trending Breakout scan: finds stocks already in a multi-week uptrend
    that are forming a volatility squeeze at the highs — the MANE pattern.
    Price above rising SMA20/SMA50, near 52-week highs, bands compressing.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def scan_trending(ticker):
        try:
            raw = yf.download(ticker, period="6mo", progress=False)
            if raw.empty or len(raw) < 60:
                return None

            close = raw["Close"].squeeze()
            high  = raw["High"].squeeze()
            low   = raw["Low"].squeeze()
            vol   = raw["Volume"].squeeze()

            price  = round(float(close.iloc[-1]), 2)
            if price < 2:
                return None

            sma20 = float(close.rolling(20).mean().iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma20_old = float(close.rolling(20).mean().iloc[-21])
            sma50_old = float(close.rolling(50).mean().iloc[-21])

            # Uptrend gate: price > SMA20 > SMA50, both rising
            if not (price > sma20 > sma50):
                return None
            if not (sma20 > sma20_old and sma50 > sma50_old):
                return None

            # RSI
            delta = close.diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi   = round(float((100 - 100 / (1 + gain / loss)).iloc[-1]), 1)
            if rsi > 80:  # exhaustion
                return None

            # BB squeeze
            bb_now = float(close.rolling(10).std().iloc[-1])
            bb_ref = float(close.rolling(30).std().iloc[-1])
            bb_squeeze = round(bb_now / bb_ref, 3) if bb_ref > 0 else 1.0

            # ATR compression
            atr_r = float((high - low).rolling(5).mean().iloc[-1])
            atr_n = float((high - low).rolling(20).mean().iloc[-1])
            atr_ratio = round(atr_r / atr_n, 3) if atr_n > 0 else 1.0

            # Distance from 52-week high
            high_52w = float(close.max())
            pct_from_high = round((price / high_52w - 1) * 100, 1)

            # Accumulation: up-day volume vs down-day volume
            diffs = close.diff().iloc[-20:]
            up_vol   = float(vol.iloc[-20:][diffs > 0].mean() or 0)
            down_vol = float(vol.iloc[-20:][diffs <= 0].mean() or 1)
            accumulation = up_vol > down_vol * 1.1

            # Returns
            ret20 = round(float((close.iloc[-1] / close.iloc[-21] - 1) * 100), 1)
            ret5  = round(float((close.iloc[-1] / close.iloc[-6]  - 1) * 100), 1)

            vol_today = float(vol.iloc[-1])
            vol_20avg = float(vol.rolling(20).mean().iloc[-2])
            vol_ratio = round(vol_today / vol_20avg, 2) if vol_20avg > 0 else 1.0

            score = 0; reasons = []

            # Trend strength
            score += 3; reasons.append("uptrend")
            if sma20 > sma20_old * 1.02: score += 1; reasons.append("SMA rising fast")

            # Near highs (key: squeeze at highs, not at lows)
            if pct_from_high >= -5:
                score += 4; reasons.append("at 52w high")
            elif pct_from_high >= -12:
                score += 2; reasons.append("near highs")
            elif pct_from_high >= -20:
                score += 1
            else:
                return None  # too far from highs — not the MANE pattern

            # Squeeze
            if bb_squeeze < 0.60:   score += 3; reasons.append("BB squeeze")
            elif bb_squeeze < 0.75: score += 2; reasons.append("BB squeeze")
            elif bb_squeeze < 0.85: score += 1

            # Coiling
            if atr_ratio < 0.65:   score += 2; reasons.append("coiling")
            elif atr_ratio < 0.80: score += 1

            # RSI in strong-trend zone
            if 55 <= rsi <= 72:    score += 2; reasons.append(f"RSI {rsi}")
            elif 45 <= rsi < 55:   score += 1; reasons.append(f"RSI {rsi}")

            # Accumulation (smart money quietly buying)
            if accumulation:       score += 2; reasons.append("accumulation")

            # Trend momentum (not stalling)
            if 5 < ret20 < 35:     score += 2; reasons.append(f"+{ret20}% /20d")
            elif 2 < ret20 <= 5:   score += 1
            elif ret20 <= 0:       score -= 3  # trend stalling

            if score < 9:
                return None

            return {
                "ticker":        ticker,
                "price":         price,
                "score":         score,
                "rsi":           rsi,
                "bb_squeeze":    bb_squeeze,
                "atr_ratio":     atr_ratio,
                "pct_from_high": pct_from_high,
                "ret20":         ret20,
                "ret5":          ret5,
                "vol_ratio":     vol_ratio,
                "accumulation":  accumulation,
                "reasons":       ", ".join(reasons),
            }
        except Exception:
            return None

    universe = list(set(TREND_UNIVERSE + SURGE_UNIVERSE))
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(scan_trending, t): t for t in universe}
        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(sanitize({
        "trending":     results[:8],
        "scanned":      len(universe),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


@app.route("/api/chart")
def api_chart():
    ticker = request.args.get("ticker", "AAPL").upper()
    period = request.args.get("period", "2y")
    try:
        # Use 1h interval for short periods, daily for longer
        interval = "1h" if period in ("1d", "5d") else "1d"
        raw = yf.Ticker(ticker).history(period=period, interval=interval)
        if raw.empty:
            return jsonify({"error": "No data"}), 404
        try:
            info = yf.Ticker(ticker).fast_info
            name = getattr(info, "name", ticker)
        except Exception:
            name = ticker
        india = ticker.endswith((".NS", ".BO")) or ticker in ("^NSEI", "^NSEBANK", "^BSESN")
        mkt_tz = "Asia/Kolkata" if india else "America/New_York"

        result = []
        for dt, row in raw.iterrows():
            if interval == "1h":
                local = dt.tz_convert(mkt_tz) if dt.tzinfo else dt
                label = local.strftime("%H:%M")
            else:
                label = str(dt.date())
            result.append({
                "date":   label,
                "open":   round(float(row["Open"]),  2),
                "high":   round(float(row["High"]),  2),
                "low":    round(float(row["Low"]),   2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        return jsonify(sanitize({"ticker": ticker, "name": name, "data": result}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze")
def api_analyze():
    """
    Pure technical-indicator analysis — no external LLM required.
    Scores RSI z-score, MACD, BB z-score, SMA trend, volume, momentum,
    news sentiment, then derives verdict / confidence / entry / target / stop.
    """
    ticker = request.args.get("ticker", "AAPL").upper()
    india  = ticker.endswith((".NS", ".BO")) or ticker in ("^NSEI", "^NSEBANK", "^BSESN")
    sym    = "₹" if india else "$"
    try:
        raw    = yf.Ticker(ticker).history(period="3y")
        close  = raw["Close"].dropna()
        volume = raw["Volume"].dropna()
        if len(close) < 60:
            return jsonify({"error": "Not enough data"}), 404

        price = round(float(close.iloc[-1]), 2)
        ret1  = round(float(close.pct_change(1).iloc[-1]  * 100), 2)
        ret5  = round(float(close.pct_change(5).iloc[-1]  * 100), 2)
        ret20 = round(float(close.pct_change(20).iloc[-1] * 100), 2)

        # RSI with z-score (3yr history normalises per-stock baseline)
        delta    = close.diff()
        gain     = delta.where(delta > 0, 0).rolling(14).mean()
        loss     = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_s    = 100 - (100 / (1 + gain / loss))
        rsi      = round(float(rsi_s.dropna().iloc[-1]), 1)
        rsi_mean = float(rsi_s.mean())
        rsi_std  = float(rsi_s.std())
        rsi_z    = round((rsi - rsi_mean) / rsi_std if rsi_std > 0 else 0.0, 2)

        # MACD histogram
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd  = ema12 - ema26
        mh    = round(float((macd - macd.ewm(span=9).mean()).dropna().iloc[-1]), 4)
        # MACD trend: count recent bars above zero
        macd_hist_series = (macd - macd.ewm(span=9).mean()).dropna()
        macd_bull_bars   = int((macd_hist_series.iloc[-5:] > 0).sum())

        # Bollinger Band z-score
        rm       = close.rolling(20).mean()
        rs       = close.rolling(20).std()
        bb_s     = (close - (rm - 2*rs)) / (4*rs)
        bb_pos   = round(float(bb_s.dropna().iloc[-1]), 3)
        bb_mean  = float(bb_s.mean())
        bb_std   = float(bb_s.std())
        bb_z     = round((bb_pos - bb_mean) / bb_std if bb_std > 0 else 0.0, 2)

        # SMA trend
        sma50    = round(float(close.rolling(50).mean().dropna().iloc[-1]), 2)
        sma200   = round(float(close.rolling(200).mean().dropna().iloc[-1]), 2) if len(close) >= 200 else sma50
        sma10_s  = close.rolling(10).mean() / close.rolling(50).mean()
        sma_z    = round(float((sma10_s.iloc[-1] - sma10_s.mean()) / sma10_s.std()), 2) if sma10_s.std() > 0 else 0.0
        sma50_pct  = round((price / sma50  - 1) * 100, 1)
        sma200_pct = round((price / sma200 - 1) * 100, 1)

        # Volume ratio
        vol_r = round(float((volume / volume.rolling(10).mean()).dropna().iloc[-1]), 2)

        # News sentiment
        sentiment_score, _ = get_sentiment(ticker)

        # S/R levels
        sr = compute_support_resistance(raw["High"].squeeze(), raw["Low"].squeeze(), price)

        # ── Traditional scoring ──────────────────────────────────────
        buy = sell = 0.0

        # RSI absolute thresholds
        if   rsi < 25:  buy  += 3.0
        elif rsi < 30:  buy  += 2.0
        elif rsi < 40:  buy  += 1.0
        elif rsi > 75:  sell += 3.0
        elif rsi > 70:  sell += 2.0
        elif rsi > 60:  sell += 0.5

        # MACD histogram direction + fresh crossover
        macd_hist_s = (macd - macd.ewm(span=9).mean()).dropna()
        prev_mh     = float(macd_hist_s.iloc[-2]) if len(macd_hist_s) >= 2 else mh
        if mh > 0:  buy  += 1.5
        else:       sell += 1.5
        if mh > 0 and prev_mh <= 0:  buy  += 1.5  # fresh bullish crossover
        if mh < 0 and prev_mh >= 0:  sell += 1.5  # fresh bearish crossover

        # Price vs SMA50 and SMA200
        if   price > sma50 and sma50 > sma200:  buy  += 2.0
        elif price > sma50:                      buy  += 1.0
        elif price < sma50 and sma50 < sma200:  sell += 2.0
        elif price < sma50:                      sell += 1.0

        # Bollinger Band position
        if   bb_pos < 0.1:  buy  += 1.5
        elif bb_pos > 0.9:  sell += 1.5

        # Volume confirmation
        if vol_r > 2.0:
            if mh > 0: buy  += 1.0
            else:      sell += 1.0
        elif vol_r > 1.5 and mh > 0:
            buy += 0.5

        # Sharp dip / extended run
        if ret5 < -8:    buy  += 1.5
        elif ret5 < -5:  buy  += 1.0
        if ret5 > 10:    sell += 1.0

        # Sentiment nudge
        if sentiment_score > 0.1:  buy  += 0.5
        if sentiment_score < -0.1: sell += 0.5

        buy  = round(buy,  2)
        sell = round(sell, 2)

        # ── Verdict ─────────────────────────────────────────────────
        gap = buy - sell
        if   gap >= 4: verdict, confidence = "BUY",  "High"
        elif gap >= 2: verdict, confidence = "BUY",  "Medium"
        elif gap >= 1: verdict, confidence = "BUY",  "Low"
        elif gap <= -4: verdict, confidence = "SELL", "High"
        elif gap <= -2: verdict, confidence = "SELL", "Medium"
        elif gap <= -1: verdict, confidence = "SELL", "Low"
        else:           verdict, confidence = "HOLD", "Low"

        # ── Entry / Target / Stop-Loss from S/R ─────────────────────
        nearest_sup = sr["support"][0]["price"]    if sr["support"]    else None
        nearest_res = sr["resistance"][0]["price"] if sr["resistance"] else None

        if verdict == "BUY":
            entry     = f"{sym}{round(price, 2)} (current) or on dip to {sym}{nearest_sup}" if nearest_sup else f"{sym}{price}"
            target    = f"{sym}{nearest_res}" if nearest_res else f"{sym}{round(price * 1.08, 2)} (+8%)"
            stop_loss = f"{sym}{round(nearest_sup * 0.985, 2)}" if nearest_sup else f"{sym}{round(price * 0.95, 2)} (-5%)"
        elif verdict == "SELL":
            entry     = f"{sym}{price} (exit now) or at bounce to {sym}{nearest_res}" if nearest_res else f"{sym}{price}"
            target    = f"{sym}{nearest_sup}" if nearest_sup else f"{sym}{round(price * 0.92, 2)} (-8%)"
            stop_loss = f"{sym}{round(nearest_res * 1.015, 2)}" if nearest_res else f"{sym}{round(price * 1.05, 2)} (+5%)"
        else:
            entry     = f"Wait — watch {sym}{nearest_sup} support / {sym}{nearest_res} resistance" if nearest_sup and nearest_res else f"{sym}{price}"
            target    = f"{sym}{nearest_res}" if nearest_res else "—"
            stop_loss = f"{sym}{nearest_sup}" if nearest_sup else "—"

        # ── Timeframe ────────────────────────────────────────────────
        if   macd_bull_bars >= 4 and vol_r > 1.5: timeframe = "short-term (days)"
        elif macd_bull_bars >= 2:                  timeframe = "medium-term (weeks)"
        else:                                       timeframe = "medium-term (weeks)"

        # ── Human-readable summary ───────────────────────────────────
        rsi_desc  = "oversold" if rsi_z < -1.5 else "overbought" if rsi_z > 1.5 else "neutral"
        macd_desc = "bullish" if mh > 0 else "bearish"
        trend     = "above" if sma50_pct > 0 else "below"
        reasons   = []
        if rsi_z < -1.5:  reasons.append(f"RSI is historically low (z={rsi_z}) — dip signal")
        if rsi_z >  1.5:  reasons.append(f"RSI is historically high (z={rsi_z}) — stretched")
        if mh > 0:        reasons.append("MACD histogram bullish")
        else:             reasons.append("MACD histogram bearish")
        if bb_z < -1.5:   reasons.append("price at lower Bollinger Band extreme")
        if bb_z >  1.5:   reasons.append("price at upper Bollinger Band extreme")
        reasons.append(f"price {trend} SMA50 by {abs(sma50_pct)}%")
        if vol_r > 1.5:   reasons.append(f"volume {vol_r}x above average confirms move")
        if sentiment_score > 0.1:  reasons.append("news sentiment positive")
        if sentiment_score < -0.1: reasons.append("news sentiment negative")
        summary = f"RSI {rsi} ({rsi_desc}, z={rsi_z}), MACD {macd_desc}. " + \
                  "; ".join(reasons[:3]) + f". Buy score {buy} vs Sell score {sell}."

        risk_factors = []
        if verdict == "BUY"  and sma200_pct < -10: risk_factors.append("price well below SMA200 — downtrend intact")
        if verdict == "SELL" and sma200_pct >  10: risk_factors.append("strong uptrend may continue")
        if abs(ret5) < 1:    risk_factors.append("low recent momentum")
        if vol_r < 0.8:      risk_factors.append("low volume reduces conviction")
        risk = "; ".join(risk_factors) if risk_factors else "Standard market and sector risk"

        return jsonify(sanitize({
            "ticker":   ticker,
            "currency": sym,
            "price":    price,
            "indicators": {
                "rsi": rsi, "rsi_z": rsi_z,
                "macd_hist": mh, "bb_pos": bb_pos, "bb_z": bb_z,
                "ret5": ret5, "ret20": ret20, "vol_ratio": vol_r,
                "sma50_pct": sma50_pct, "sma200_pct": sma200_pct,
                "buy_score": buy, "sell_score": sell,
            },
            "analysis": {
                "verdict":    verdict,
                "confidence": confidence,
                "summary":    summary,
                "entry":      entry,
                "target":     target,
                "stop_loss":  stop_loss,
                "risk":       risk,
                "timeframe":  timeframe,
            },
            "support":    sr["support"],
            "resistance": sr["resistance"],
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/options")
def api_options():
    """
    NIFTY / BankNifty options chain via the NSE proxy deployed on Railway/Render.
    Set NSE_PROXY_URL env var to the proxy base URL, e.g. https://your-proxy.railway.app
    """
    import os, requests as _req
    from datetime import datetime as _dt, date as _date

    proxy_base = os.environ.get("NSE_PROXY_URL", "").rstrip("/")
    if not proxy_base:
        return jsonify({"error": "NSE_PROXY_URL not set — deploy the proxy first"}), 503

    def fetch_records(symbol):
        r = _req.get(f"{proxy_base}/options/{symbol}", timeout=20)
        r.raise_for_status()
        return r.json()

    def compute_chain(calls, puts, expiry_str):
        total_ce = sum(v["oi"] for v in calls.values())
        total_pe = sum(v["oi"] for v in puts.values())
        pcr = round(total_pe / total_ce, 3) if total_ce > 0 else None

        all_s = sorted(set(list(calls) + list(puts)))
        mp_val, mp_strike = float("inf"), None
        for s in all_s:
            loss = (sum(max(s - k, 0) * v["oi"] for k, v in calls.items()) +
                    sum(max(k - s, 0) * v["oi"] for k, v in puts.items()))
            if loss < mp_val:
                mp_val, mp_strike = loss, s

        top_ce = sorted(calls.items(), key=lambda x: x[1]["oi"], reverse=True)[:3]
        top_pe = sorted(puts.items(),  key=lambda x: x[1]["oi"], reverse=True)[:3]

        if   pcr is None:  signal, pcr_text = "HOLD", "n/a"
        elif pcr >= 1.5:   signal, pcr_text = "BUY",  f"{pcr} — Strong support"
        elif pcr >= 1.1:   signal, pcr_text = "BUY",  f"{pcr} — Moderate support"
        elif pcr <= 0.6:   signal, pcr_text = "SELL", f"{pcr} — Strong resistance"
        elif pcr <= 0.9:   signal, pcr_text = "SELL", f"{pcr} — Moderate resistance"
        else:              signal, pcr_text = "HOLD", f"{pcr} — Neutral"

        return {
            "expiry": expiry_str, "signal": signal, "pcr": pcr, "pcr_text": pcr_text,
            "total_call_oi": total_ce, "total_put_oi": total_pe, "max_pain": mp_strike,
            "ce_resistance": [{"strike":k,"oi":v["oi"],"vol":v["vol"],"ltp":v["ltp"]} for k,v in top_ce],
            "pe_support":    [{"strike":k,"oi":v["oi"],"vol":v["vol"],"ltp":v["ltp"]} for k,v in top_pe],
        }

    def process_index(symbol, display_name):
        try:
            records = fetch_records(symbol)
            if not records or "error" in records:
                return {"error": records.get("error", "Proxy returned no data")}

            spot = float(records.get("underlyingValue", 0))
            expiry_dates = records.get("expiryDates", [])
            chain_data   = records.get("data", [])

            if not expiry_dates or not chain_data:
                return {"error": "Empty option chain from NSE", "spot": spot}

            today = _date.today()
            parsed = sorted(
                (_dt.strptime(e, "%d-%b-%Y").date(), e)
                for e in expiry_dates
                if _dt.strptime(e, "%d-%b-%Y").date() >= today
            )
            if not parsed:
                return {"error": "No future expiries found", "spot": spot}

            weekly_str  = parsed[0][1]
            monthly_str = next(
                (s for d, s in parsed if d.month == today.month),
                parsed[-1][1]
            )

            def build_expiry(exp_str):
                calls, puts = {}, {}
                for row in chain_data:
                    if row.get("expiryDate") != exp_str:
                        continue
                    strike = int(row.get("strikePrice", 0))
                    if spot and abs(strike - spot) / spot > 0.15:
                        continue
                    if "CE" in row:
                        ce = row["CE"]
                        calls[strike] = {
                            "oi":  int(ce.get("openInterest", 0) or 0),
                            "vol": int(ce.get("totalTradedVolume", 0) or 0),
                            "ltp": float(ce.get("lastPrice", 0) or 0),
                        }
                    if "PE" in row:
                        pe = row["PE"]
                        puts[strike] = {
                            "oi":  int(pe.get("openInterest", 0) or 0),
                            "vol": int(pe.get("totalTradedVolume", 0) or 0),
                            "ltp": float(pe.get("lastPrice", 0) or 0),
                        }
                return compute_chain(calls, puts, exp_str)

            out = {"spot": spot, "name": display_name}
            try:    out["weekly"]  = build_expiry(weekly_str)
            except Exception as e: out["weekly"] = {"error": str(e)}

            if monthly_str != weekly_str:
                try:    out["monthly"] = build_expiry(monthly_str)
                except Exception as e: out["monthly"] = {"error": str(e)}
            else:
                out["monthly"] = out["weekly"]

            return out
        except Exception as e:
            return {"error": str(e)}

    return jsonify(sanitize({
        "NIFTY":        process_index("NIFTY",     "Nifty 50"),
        "BANKNIFTY":    process_index("BANKNIFTY", "Bank Nifty"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


@app.route("/")
def dashboard():
    import os
    return render_template_string(HTML_TEMPLATE, stocks=STOCKS,
                                  nse_proxy=os.environ.get("NSE_PROXY_URL", "").rstrip("/"))


# ══════════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD — Stock Explorer
# ══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Stock Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:      #0f1117;
  --surface: #1a1d27;
  --surface2:#20232f;
  --border:  #2a2d3a;
  --text:    #e2e8f0;
  --muted:   #8892a4;
  --up:      #22c55e;
  --down:    #ef4444;
  --neutral: #f59e0b;
  --accent:  #6366f1;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden;}

/* ── Header ─────────────────────────────────────────────── */
header{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:.8rem 1.4rem;display:flex;align-items:center;gap:1rem;flex-shrink:0;
}
header h1{font-size:1.2rem;font-weight:800;white-space:nowrap;}
header h1 span{color:var(--accent);}
.hdr-btns{display:flex;gap:.6rem;margin-left:auto;}
.hdr-btn{
  background:var(--surface2);border:1px solid var(--border);color:var(--text);
  padding:.4rem .9rem;border-radius:8px;cursor:pointer;font-size:.82rem;font-weight:600;
  transition:background .15s;white-space:nowrap;
}
.hdr-btn:hover{background:var(--border);}

/* ── App body (sidebar + main) ──────────────────────────── */
.app-body{display:flex;flex:1;overflow:hidden;}

/* ── Sidebar ────────────────────────────────────────────── */
#sidebar{
  width:220px;flex-shrink:0;background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}
.tab-row{display:flex;border-bottom:1px solid var(--border);}
.tab{flex:1;padding:.6rem;text-align:center;font-size:.8rem;font-weight:700;cursor:pointer;color:var(--muted);}
.tab.active{color:var(--accent);border-bottom:2px solid var(--accent);}
#stock-search{
  margin:.6rem;padding:.45rem .7rem;background:var(--bg);border:1px solid var(--border);
  color:var(--text);border-radius:8px;font-size:.82rem;outline:none;
}
#stock-search:focus{border-color:var(--accent);}
#stock-list{flex:1;overflow-y:auto;padding-bottom:.5rem;}
.sector-label{
  font-size:.65rem;font-weight:700;color:var(--muted);text-transform:uppercase;
  letter-spacing:.07em;padding:.7rem .9rem .3rem;
}
.stock-item{
  padding:.42rem .9rem;font-size:.85rem;cursor:pointer;display:flex;
  justify-content:space-between;align-items:center;transition:background .1s;border-radius:6px;margin:1px 4px;
}
.stock-item:hover{background:var(--surface2);}
.stock-item.selected{background:rgba(99,102,241,.18);color:var(--accent);}
.stk-name{font-size:.7rem;color:var(--muted);max-width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

/* ── Main panel ─────────────────────────────────────────── */
#main{flex:1;overflow-y:auto;padding:1.2rem;}

#empty-state{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:70%;color:var(--muted);gap:.8rem;
}
#empty-state svg{opacity:.3;}
#empty-state p{font-size:.95rem;}

#stock-header{display:none;margin-bottom:1rem;}
.stk-title{font-size:1.5rem;font-weight:800;}
.stk-price{font-size:1.2rem;font-weight:700;margin-top:.2rem;}
.stk-change.up{color:var(--up);}
.stk-change.down{color:var(--down);}

.period-row{display:flex;gap:.4rem;margin-bottom:1rem;}
.period-btn{
  padding:.3rem .75rem;border-radius:6px;font-size:.78rem;font-weight:600;
  cursor:pointer;background:var(--surface);border:1px solid var(--border);color:var(--muted);
}
.period-btn.active{background:var(--accent);border-color:var(--accent);color:#fff;}

#chart-wrap{
  background:var(--surface);border:1px solid var(--border);border-radius:14px;
  padding:1rem;margin-bottom:1rem;position:relative;height:300px;
}
#chart-loading{
  position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:var(--muted);font-size:.85rem;background:var(--surface);border-radius:14px;
}

.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;}
@media(max-width:700px){.info-grid{grid-template-columns:1fr;}}

.card{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:1.2rem;}
.card-title{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:.9rem;}

.verdict-badge{
  display:inline-block;padding:.35rem .9rem;border-radius:20px;font-weight:800;font-size:1rem;margin-bottom:.6rem;
}
.verdict-badge.BUY{background:rgba(34,197,94,.15);color:var(--up);}
.verdict-badge.SELL{background:rgba(239,68,68,.15);color:var(--down);}
.verdict-badge.HOLD{background:rgba(245,158,11,.15);color:var(--neutral);}

.analysis-summary{font-size:.84rem;line-height:1.5;color:var(--text);margin-bottom:.8rem;}
.analysis-grid{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;}
.a-item{background:var(--bg);border-radius:8px;padding:.5rem .7rem;}
.a-label{font-size:.67rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}
.a-value{font-size:.88rem;font-weight:700;margin-top:2px;}

.ind-row{display:flex;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid var(--border);font-size:.83rem;}
.ind-row:last-child{border-bottom:none;}
.ind-name{color:var(--muted);}
.ind-val{font-weight:600;}

.sr-heading{font-size:.75rem;font-weight:700;text-transform:uppercase;color:var(--muted);letter-spacing:.07em;margin-bottom:.4rem;}
.sr-level{
  display:flex;justify-content:space-between;align-items:center;
  padding:.35rem .6rem;border-radius:8px;font-size:.82rem;margin-bottom:.3rem;
}
.sr-level.res{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.2);}
.sr-level.sup{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.2);}
.sr-price{font-weight:700;}
.sr-pct{font-size:.75rem;color:var(--muted);}
.sr-touches{font-size:.72rem;color:var(--muted);}

.regime-banner{
  border-radius:10px;padding:.6rem 1rem;font-size:.82rem;margin-bottom:1rem;
  border:1px solid var(--border);display:flex;align-items:center;gap:.6rem;flex-wrap:wrap;
}
.regime-banner.up{border-color:rgba(34,197,94,.3);background:rgba(34,197,94,.07);}
.regime-banner.down{border-color:rgba(239,68,68,.3);background:rgba(239,68,68,.07);}
.regime-banner.neutral{border-color:rgba(245,158,11,.3);background:rgba(245,158,11,.07);}

/* ── Scan overlay ─────────────────────────────────────────── */
#scan-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;
  overflow-y:auto;padding:2rem;
}
#scan-box{
  max-width:920px;margin:0 auto;background:var(--surface);border:1px solid var(--border);
  border-radius:16px;padding:1.5rem;
}
.scan-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;}
.scan-header h2{font-size:1.1rem;font-weight:800;}
#close-scan{background:none;border:none;color:var(--muted);cursor:pointer;font-size:1.5rem;line-height:1;}
#close-scan:hover{color:var(--text);}
.scan-loading{text-align:center;padding:2rem;color:var(--muted);}
.spin{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;margin:.5rem auto;}
@keyframes spin{to{transform:rotate(360deg);}}
#scan-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem;margin-top:1rem;}

.scan-card{background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1.1rem;}
.sc-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem;}
.sc-ticker{font-size:1.1rem;font-weight:800;}
.sc-badge{font-size:.8rem;font-weight:700;padding:.25rem .7rem;border-radius:20px;}
.sc-badge.BUY{background:rgba(34,197,94,.15);color:var(--up);}
.sc-badge.SELL{background:rgba(239,68,68,.15);color:var(--down);}
.sc-badge.WATCH{background:rgba(245,158,11,.15);color:var(--neutral);}
.sc-price{font-size:1rem;font-weight:700;margin-bottom:.5rem;}
.sc-row{font-size:.78rem;color:var(--muted);margin-bottom:.2rem;}
.sc-row span{color:var(--text);font-weight:600;}
.sc-sr{margin-top:.5rem;font-size:.75rem;}
.sc-sr-line{padding:.2rem .4rem;border-radius:5px;margin-bottom:.2rem;}
.sc-sr-line.res{background:rgba(239,68,68,.1);}
.sc-sr-line.sup{background:rgba(34,197,94,.1);}

/* ── Signal guide overlay ──────────────────────────────────── */
#guide-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:200;overflow-y:auto;padding:2rem;}
#guide-box{max-width:720px;margin:0 auto;background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;}
.guide-close-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;}
#close-guide{background:none;border:none;color:var(--muted);cursor:pointer;font-size:1.5rem;}
.guide-table{width:100%;border-collapse:collapse;font-size:.82rem;}
.guide-table th{text-align:left;padding:.5rem .7rem;color:var(--muted);border-bottom:1px solid var(--border);}
.guide-table td{padding:.45rem .7rem;border-bottom:1px solid var(--border);}
.guide-table tr:last-child td{border-bottom:none;}

.analyze-loading{text-align:center;padding:1.5rem;color:var(--muted);font-size:.85rem;}
</style>
</head>
<body>

<header>
  <h1>Stock <span>Explorer</span></h1>
  <div class="hdr-btns">
    <button class="hdr-btn" onclick="showScan('market')">Market Scan</button>
    <button class="hdr-btn" onclick="showScan('surge')">Surge Scan</button>
    <button class="hdr-btn" onclick="showScan('movers')">Today's Movers</button>
    <button class="hdr-btn" onclick="showScan('trending')">Trend Breakouts</button>
    <button class="hdr-btn" onclick="showOptions()">India Options</button>
    <button class="hdr-btn" onclick="document.getElementById('rr-overlay').style.display='flex'" style="background:var(--buy);color:#000;font-weight:800;">R:R Calc</button>
    <button class="hdr-btn" onclick="document.getElementById('guide-overlay').style.display='block'">Signal Guide</button>
  </div>
</header>

<div class="app-body">

  <!-- Sidebar -->
  <div id="sidebar">
    <div class="tab-row">
      <div class="tab active" id="tab-us"    onclick="switchTab('us')">US</div>
      <div class="tab"        id="tab-india" onclick="switchTab('india')">India</div>
    </div>
    <div style="padding:.5rem .9rem .3rem;">
      <div id="portfolio-overall" style="line-height:1.5;">—</div>
    </div>
    <input id="stock-search" placeholder="Search ticker or name…" oninput="filterStocks(this.value)"/>
    <div id="stock-list"></div>
  </div>

  <!-- Main panel -->
  <div id="main">

    <div id="empty-state">
      <svg width="64" height="64" fill="none" stroke="#6366f1" stroke-width="1.5" viewBox="0 0 24 24">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
      <p>Select a stock from the sidebar to begin</p>
    </div>

    <div id="stock-header">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem;margin-bottom:.6rem;">
        <div>
          <div class="stk-title" id="hdr-ticker"></div>
          <div id="hdr-name" style="color:var(--muted);font-size:.85rem;margin-top:2px;"></div>
        </div>
        <div style="text-align:right;">
          <div class="stk-price" id="hdr-price"></div>
          <div class="stk-change" id="hdr-change"></div>
        </div>
      </div>
      <div id="regime-row"></div>
    </div>

    <div class="period-row" id="period-row" style="display:none;">
      <button class="period-btn" data-p="1d"  onclick="setPeriod(this)">1D</button>
      <button class="period-btn" data-p="5d"  onclick="setPeriod(this)">1W</button>
      <button class="period-btn" data-p="1mo" onclick="setPeriod(this)">1M</button>
      <button class="period-btn" data-p="6mo" onclick="setPeriod(this)">6M</button>
      <button class="period-btn active" data-p="2y" onclick="setPeriod(this)">2Y</button>
      <button class="period-btn" data-p="5y"  onclick="setPeriod(this)">5Y</button>
    </div>

    <div id="chart-wrap" style="display:none;">
      <div id="chart-loading">Loading chart…</div>
      <canvas id="price-chart"></canvas>
    </div>

    <div class="info-grid" id="info-grid" style="display:none;">
      <div class="card" id="analysis-card">
        <div class="card-title">AI Analysis</div>
        <div id="analysis-content" class="analyze-loading">
          <div class="spin"></div>Loading analysis…
        </div>
      </div>
      <div class="card">
        <div class="card-title">Indicators</div>
        <div id="indicators-content"></div>
      </div>
    </div>

    <div class="card" id="sr-card" style="display:none;margin-bottom:1rem;">
      <div class="card-title">Support &amp; Resistance</div>
      <div id="sr-content"></div>
    </div>

  </div><!-- /main -->
</div><!-- /app-body -->

<!-- Scan overlay -->
<div id="scan-overlay">
  <div id="scan-box">
    <div class="scan-header">
      <h2 id="scan-title">Market Scan</h2>
      <button id="close-scan" onclick="closeScan()">&#x2715;</button>
    </div>
    <div id="regime-us-banner"></div>
    <div id="regime-india-banner"></div>
    <div id="scan-loading" class="scan-loading"><div class="spin"></div><div>Scanning…</div></div>
    <div id="scan-cards"></div>
  </div>
</div>

<!-- R:R Calculator overlay -->
<div id="rr-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:200;align-items:center;justify-content:center;">
  <div style="background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.8rem;width:360px;max-width:95vw;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;">
      <h2 style="font-size:1rem;font-weight:800;">Trade R:R Calculator</h2>
      <button onclick="document.getElementById('rr-overlay').style.display='none'" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:1.4rem;">&#x2715;</button>
    </div>
    <div style="display:grid;gap:.7rem;">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;">
        <label style="font-size:.78rem;color:var(--muted);">Entry Price
          <input id="rr-entry" type="number" step="0.01" placeholder="e.g. 250"
            oninput="calcRR()" style="width:100%;margin-top:.3rem;padding:.5rem;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.85rem;">
        </label>
        <label style="font-size:.78rem;color:var(--muted);">Stop Loss
          <input id="rr-sl" type="number" step="0.01" placeholder="e.g. 235"
            oninput="calcRR()" style="width:100%;margin-top:.3rem;padding:.5rem;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.85rem;">
        </label>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;">
        <label style="font-size:.78rem;color:var(--muted);">Target 1
          <input id="rr-t1" type="number" step="0.01" placeholder="e.g. 275"
            oninput="calcRR()" style="width:100%;margin-top:.3rem;padding:.5rem;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.85rem;">
        </label>
        <label style="font-size:.78rem;color:var(--muted);">Target 2
          <input id="rr-t2" type="number" step="0.01" placeholder="e.g. 295"
            oninput="calcRR()" style="width:100%;margin-top:.3rem;padding:.5rem;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.85rem;">
        </label>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;">
        <label style="font-size:.78rem;color:var(--muted);">Capital (₹/$)
          <input id="rr-capital" type="number" step="1000" placeholder="e.g. 100000"
            oninput="calcRR()" style="width:100%;margin-top:.3rem;padding:.5rem;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.85rem;">
        </label>
        <label style="font-size:.78rem;color:var(--muted);">Risk % per trade
          <input id="rr-risk-pct" type="number" step="0.5" value="1" min="0.1" max="5"
            oninput="calcRR()" style="width:100%;margin-top:.3rem;padding:.5rem;background:var(--bg);border:1px solid var(--border);color:var(--text);border-radius:8px;font-size:.85rem;">
        </label>
      </div>
    </div>
    <div id="rr-result" style="margin-top:1.2rem;"></div>
  </div>
</div>

<!-- India Options overlay -->
<div id="options-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:150;overflow-y:auto;padding:2rem;">
  <div style="max-width:960px;margin:0 auto;background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1.2rem;">
      <h2 style="font-size:1.1rem;font-weight:800;">India Options — NIFTY &amp; Bank Nifty</h2>
      <button onclick="document.getElementById('options-overlay').style.display='none'" style="background:none;border:none;color:var(--muted);cursor:pointer;font-size:1.5rem;">&#x2715;</button>
    </div>
    <div id="options-loading" style="text-align:center;padding:2rem;color:var(--muted);"><div class="spin"></div><div>Loading options data…</div></div>
    <div id="options-content" style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;"></div>
    <p style="margin-top:1rem;font-size:.72rem;color:var(--muted);">PCR = Put/Call Ratio · Max Pain = strike where option buyers lose most at expiry · OI = Open Interest</p>
  </div>
</div>

<!-- Signal Guide overlay -->
<div id="guide-overlay">
  <div id="guide-box">
    <div class="guide-close-row">
      <h2 style="font-size:1.1rem;font-weight:800;">Signal Guide</h2>
      <button id="close-guide" onclick="document.getElementById('guide-overlay').style.display='none'">&#x2715;</button>
    </div>
    <table class="guide-table">
      <thead><tr><th>Signal</th><th>What it means</th><th>Threshold</th></tr></thead>
      <tbody>
        <tr><td>RSI Z-score</td><td>Compares RSI to its own 3yr avg. Negative = oversold for <em>this</em> stock.</td><td>&lt;-2 strong buy; &gt;+2 strong sell</td></tr>
        <tr><td>MACD Histogram</td><td>Momentum direction. Positive = short-term trend bullish.</td><td>&gt;0 bull signal</td></tr>
        <tr><td>BB Position Z</td><td>Bollinger Band vs own history. &lt;-1.5 = extended dip.</td><td>&lt;-1.5 buy; &gt;+1.5 sell</td></tr>
        <tr><td>SMA Z-score</td><td>Short/long MA ratio vs baseline. Positive = trend above norm.</td><td>&gt;+0.5 momentum buy</td></tr>
        <tr><td>Volume Ratio</td><td>Current volume vs 10-day avg. &gt;2× confirms breakouts.</td><td>&gt;2× significant</td></tr>
        <tr><td>5d Return</td><td>5-day price change. Negative = dip; very positive = extended.</td><td>&lt;-5% dip; &gt;+8% caution</td></tr>
        <tr><td>PCR (US only)</td><td>Put/Call Ratio. &gt;1.3 = fear → contrarian buy signal.</td><td>&gt;1.3 buy; &lt;0.6 sell</td></tr>
        <tr><td>Buy / Sell Score</td><td>Composite of all signals. Threshold adjusts with regime.</td><td>≥4 standard (≥3 bullish)</td></tr>
        <tr><td>Market Regime</td><td>SPY/NSEI trend + RSI. Adjusts signal thresholds.</td><td>bullish/bearish/ranging/rally/selloff</td></tr>
      </tbody>
    </table>
  </div>
</div>

<script>
const UNIVERSE = {
  us: {
    "⭐ My Portfolio": [
      {t:"CSWC", n:"Capital Southwest",   pl:+3.1},
      {t:"VUAA.DE", n:"Vanguard S&P500 UCITS", pl:+5.0},
      {t:"GME",  n:"GameStop",            pl:-4.0},
      {t:"CCEC", n:"Cap Clean Energy",    pl:-7.4},
      {t:"XNDU", n:"Xanadu Quantum",      pl:-8.8},
      {t:"NVAX", n:"Novanax",             pl:-2.92},
      {t:"NIO",  n:"NIO",                 pl:-4.8},
      {t:"RXRX", n:"Recursion Pharma",    pl:-5.37},
      {t:"BDMD",    n:"Baird Medical",       pl:-40.0},
      {t:"PTP.DE",  n:"Pentixapharm",        pl:-41.0},
      {t:"DFTK.DE", n:"DFTK Tradegate",      pl:-71.0},
    ],
    Technology:  [{t:"AAPL",n:"Apple"},{t:"MSFT",n:"Microsoft"},{t:"NVDA",n:"NVIDIA"},{t:"AMD",n:"AMD"},{t:"META",n:"Meta"},{t:"GOOGL",n:"Alphabet"},{t:"AMZN",n:"Amazon"},{t:"TSLA",n:"Tesla"}],
    Finance:     [{t:"JPM",n:"JP Morgan"},{t:"BAC",n:"Bank of America"},{t:"GS",n:"Goldman Sachs"},{t:"MS",n:"Morgan Stanley"},{t:"V",n:"Visa"},{t:"MA",n:"Mastercard"}],
    Healthcare:  [{t:"UNH",n:"UnitedHealth"},{t:"PFE",n:"Pfizer"},{t:"JNJ",n:"J&J"}],
    Energy:      [{t:"XOM",n:"ExxonMobil"},{t:"CVX",n:"Chevron"}],
    ETFs:        [{t:"SPY",n:"S&P 500 ETF"},{t:"QQQ",n:"Nasdaq ETF"},{t:"ARKK",n:"ARK Innovation"}],
  },
  india: {
    IT:      [{t:"TCS.NS",n:"TCS"},{t:"INFY.NS",n:"Infosys"},{t:"WIPRO.NS",n:"Wipro"},{t:"HCLTECH.NS",n:"HCL Tech"}],
    Banking: [{t:"HDFCBANK.NS",n:"HDFC Bank"},{t:"ICICIBANK.NS",n:"ICICI Bank"},{t:"AXISBANK.NS",n:"Axis Bank"},{t:"SBIN.NS",n:"SBI"}],
    Finance: [{t:"BAJFINANCE.NS",n:"Bajaj Finance"}],
    Energy:  [{t:"ONGC.NS",n:"ONGC"},{t:"NTPC.NS",n:"NTPC"},{t:"POWERGRID.NS",n:"Power Grid"}],
    Auto:    [{t:"MARUTI.NS",n:"Maruti"},{t:"TATAMOTORS.NS",n:"Tata Motors"}],
    Pharma:  [{t:"SUNPHARMA.NS",n:"Sun Pharma"},{t:"DRREDDY.NS",n:"Dr Reddy's"}],
    Consumer:[{t:"TITAN.NS",n:"Titan"}],
    Infra:   [{t:"ADANIENT.NS",n:"Adani Ent."},{t:"LT.NS",n:"L&T"}],
  }
};

let currentTab    = "us";
let currentTicker = null;
let currentPeriod = "2y";
let chartInst     = null;
let curSym        = "$";
const NSE_PROXY   = "{{ nse_proxy }}";

function switchTab(tab) {
  currentTab = tab;
  document.getElementById("tab-us").classList.toggle("active", tab==="us");
  document.getElementById("tab-india").classList.toggle("active", tab==="india");
  document.getElementById("stock-search").value = "";
  renderSidebar(tab, "");
}

function renderSidebar(tab, filter) {
  const sectors = UNIVERSE[tab];
  const list    = document.getElementById("stock-list");
  let html = "";
  for (const [sector, stocks] of Object.entries(sectors)) {
    const visible = stocks.filter(s =>
      !filter ||
      s.t.toLowerCase().includes(filter.toLowerCase()) ||
      s.n.toLowerCase().includes(filter.toLowerCase())
    );
    if (!visible.length) continue;
    html += `<div class="sector-label">${sector}</div>`;
    for (const s of visible) {
      const sel = s.t === currentTicker ? " selected" : "";
      const plColor  = s.pl >= 0 ? "var(--buy)" : "var(--sell)";
      const plBg     = s.pl >= 0 ? "rgba(74,222,128,.12)" : "rgba(239,68,68,.12)";
      const preTag   = s.pre ? `<span style="font-size:.6rem;color:var(--neutral);margin-right:.2rem;">PRE</span>` : "";
      const plBadge  = s.pl != null
        ? `<div style="text-align:right;line-height:1.3;">
            ${preTag}
            <div style="font-size:.68rem;font-weight:800;color:${plColor};">${s.pl>=0?"+":""}${s.pl.toFixed(1)}%</div>
            ${s.pnlUsd!=null?`<div style="font-size:.62rem;color:${plColor};opacity:.85;">${s.pnlUsd>=0?"+":""}$${s.pnlUsd.toFixed(0)}</div>`:""}
           </div>`
        : "";
      const urgentSell = s.pl != null && s.pl <= -15;
      const warnSell   = s.pl != null && s.pl < -8 && s.pl > -15;
      const rowBg = urgentSell ? "background:rgba(239,68,68,.13);border-left:3px solid var(--sell);"
                  : warnSell   ? "background:rgba(251,146,60,.08);border-left:3px solid var(--neutral);"
                  : "";
      html += `<div class="stock-item${sel}" onclick="selectStock('${s.t}','${s.n.replace(/'/g,"\\\\'")}')" style="${rowBg}">
        <div style="flex:1;">
          <div style="font-weight:700;">${s.t.replace(".NS","").replace(".DE","")}</div>
          <div class="stk-name">${s.n}${urgentSell?` <span style="color:var(--sell);font-size:.6rem;font-weight:800;">SELL</span>`:""}</div>
        </div>
        ${plBadge}
      </div>`;
    }
  }
  list.innerHTML = html || `<div style="padding:.8rem;color:var(--muted);font-size:.82rem;">No results</div>`;
}

function filterStocks(val) { renderSidebar(currentTab, val); }

function selectStock(ticker, name) {
  currentTicker = ticker;
  curSym = (ticker.endsWith(".NS") || ticker.endsWith(".BO") ||
            ticker === "^NSEI" || ticker === "^NSEBANK") ? "₹" : "$";
  renderSidebar(currentTab, document.getElementById("stock-search").value);

  document.getElementById("empty-state").style.display  = "none";
  document.getElementById("stock-header").style.display = "block";
  document.getElementById("period-row").style.display   = "flex";
  document.getElementById("chart-wrap").style.display   = "block";
  document.getElementById("info-grid").style.display    = "grid";
  document.getElementById("sr-card").style.display      = "block";

  document.getElementById("hdr-ticker").textContent = ticker.replace(".NS","");
  document.getElementById("hdr-name").textContent   = name;
  document.getElementById("hdr-price").textContent  = "—";
  document.getElementById("hdr-change").className   = "stk-change";
  document.getElementById("hdr-change").textContent = "";
  document.getElementById("regime-row").innerHTML   = "";
  document.getElementById("analysis-content").innerHTML =
    `<div class="analyze-loading"><div class="spin"></div>Loading analysis…</div>`;
  document.getElementById("indicators-content").innerHTML = "";
  document.getElementById("sr-content").innerHTML   = "";

  document.getElementById("chart-loading").style.display = "flex";
  if (chartInst) { chartInst.destroy(); chartInst = null; }

  loadChart(ticker, currentPeriod);
  loadAnalysis(ticker);
}

function setPeriod(btn) {
  document.querySelectorAll(".period-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentPeriod = btn.dataset.p;
  if (currentTicker) loadChart(currentTicker, currentPeriod);
}

async function loadChart(ticker, period) {
  document.getElementById("chart-loading").style.display = "flex";
  if (chartInst) { chartInst.destroy(); chartInst = null; }
  try {
    const res  = await fetch(`/api/chart?ticker=${encodeURIComponent(ticker)}&period=${period}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const pts  = data.data;
    const last = pts[pts.length-1];
    const prev = pts[pts.length-2];
    const chg  = prev ? ((last.close-prev.close)/prev.close*100).toFixed(2) : null;

    document.getElementById("hdr-price").textContent = `${curSym}${last.close.toLocaleString()}`;
    if (chg !== null) {
      const el = document.getElementById("hdr-change");
      el.textContent = `${chg > 0 ? "+":""}${chg}% today`;
      el.className = "stk-change " + (parseFloat(chg) >= 0 ? "up" : "down");
    }

    document.getElementById("chart-loading").style.display = "none";

    const labels = pts.map(d => d.date);
    const prices = pts.map(d => d.close);
    const isUp   = prices[prices.length-1] >= prices[0];
    const ctx    = document.getElementById("price-chart").getContext("2d");
    const grad   = ctx.createLinearGradient(0,0,0,260);
    grad.addColorStop(0, isUp ? "rgba(34,197,94,.25)" : "rgba(239,68,68,.25)");
    grad.addColorStop(1, "rgba(0,0,0,0)");

    chartInst = new Chart(ctx, {
      type:"line",
      data:{
        labels,
        datasets:[{
          data:prices,
          borderColor: isUp ? "#22c55e" : "#ef4444",
          backgroundColor: grad,
          borderWidth:1.8,
          pointRadius:0,
          tension:.3,
          fill:true,
        }]
      },
      options:{
        responsive:true,
        maintainAspectRatio:false,
        interaction:{mode:"index",intersect:false},
        plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${curSym}${c.raw.toLocaleString()}`}}},
        scales:{
          x:{grid:{color:"rgba(255,255,255,.05)"},ticks:{color:"#8892a4",maxTicksLimit:8,font:{size:10}}},
          y:{grid:{color:"rgba(255,255,255,.05)"},ticks:{color:"#8892a4",font:{size:10},callback:v=>`${curSym}${v.toLocaleString()}`},position:"right"},
        }
      }
    });
  } catch(e) {
    const el = document.getElementById("chart-loading");
    el.textContent = "Chart unavailable: " + e.message;
    el.style.display = "flex";
  }
}

async function loadAnalysis(ticker) {
  try {
    const res  = await fetch(`/api/analyze?ticker=${encodeURIComponent(ticker)}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const a = data.analysis || {};
    const v = a.verdict || "HOLD";

    document.getElementById("analysis-content").innerHTML = `
      <span class="verdict-badge ${v}">${v}</span>
      <span style="font-size:.8rem;color:var(--muted);margin-left:.5rem;">${a.confidence||""} confidence</span>
      <p class="analysis-summary">${a.summary||""}</p>
      <div class="analysis-grid">
        <div class="a-item"><div class="a-label">Entry</div><div class="a-value">${a.entry||"—"}</div></div>
        <div class="a-item"><div class="a-label">Target</div><div class="a-value">${a.target||"—"}</div></div>
        <div class="a-item"><div class="a-label">Stop Loss</div><div class="a-value">${a.stop_loss||"—"}</div></div>
        <div class="a-item"><div class="a-label">Timeframe</div><div class="a-value">${a.timeframe||"—"}</div></div>
      </div>
      ${a.risk ? `<div style="margin-top:.7rem;font-size:.78rem;color:var(--muted);">&#9888; Risk: ${a.risk}</div>` : ""}
    `;

    const ind = data.indicators || {};
    const fmt = v => (v == null ? "n/a" : v);
    const colRSI    = ind.rsi    > 65 ? "var(--down)" : ind.rsi    < 35 ? "var(--up)" : "var(--text)";
    const colMACD   = (ind.macd_hist||0) > 0 ? "var(--up)" : "var(--down)";
    const colBB     = (ind.bb_pos||0)    < 0.2 ? "var(--up)" : (ind.bb_pos||0) > 0.8 ? "var(--down)" : "var(--text)";
    const colVol    = (ind.vol_ratio||0) > 1.5 ? "var(--neutral)" : "var(--text)";
    const colSMA50  = (ind.sma50_pct||0) > 0 ? "var(--up)" : "var(--down)";
    const colSMA200 = (ind.sma200_pct||0)> 0 ? "var(--up)" : "var(--down)";

    document.getElementById("indicators-content").innerHTML = `
      <div class="ind-row"><span class="ind-name">RSI(14)</span><span class="ind-val" style="color:${colRSI}">${fmt(ind.rsi)}</span></div>
      <div class="ind-row"><span class="ind-name">MACD Hist</span><span class="ind-val" style="color:${colMACD}">${fmt(ind.macd_hist)}</span></div>
      <div class="ind-row"><span class="ind-name">BB Position</span><span class="ind-val" style="color:${colBB}">${fmt(ind.bb_pos)}</span></div>
      <div class="ind-row"><span class="ind-name">Volume Ratio</span><span class="ind-val" style="color:${colVol}">${fmt(ind.vol_ratio)}x</span></div>
      <div class="ind-row"><span class="ind-name">vs SMA50</span><span class="ind-val" style="color:${colSMA50}">${fmt(ind.sma50_pct)}%</span></div>
      <div class="ind-row"><span class="ind-name">vs SMA200</span><span class="ind-val" style="color:${colSMA200}">${fmt(ind.sma200_pct)}%</span></div>
      <div class="ind-row"><span class="ind-name">5d Return</span><span class="ind-val">${fmt(ind.ret5)}%</span></div>
      <div class="ind-row"><span class="ind-name">20d Return</span><span class="ind-val">${fmt(ind.ret20)}%</span></div>
    `;

    let srHtml = "";
    if (data.resistance && data.resistance.length) {
      srHtml += `<div class="sr-heading">Resistance</div>`;
      data.resistance.forEach(r => {
        srHtml += `<div class="sr-level res">
          <span class="sr-price">${curSym}${r.price.toLocaleString()}</span>
          <span class="sr-pct">+${r.pct_away}% away</span>
          <span class="sr-touches">${r.touches}x</span>
        </div>`;
      });
    }
    if (data.support && data.support.length) {
      srHtml += `<div class="sr-heading" style="margin-top:.5rem;">Support</div>`;
      data.support.forEach(s => {
        srHtml += `<div class="sr-level sup">
          <span class="sr-price">${curSym}${s.price.toLocaleString()}</span>
          <span class="sr-pct">-${s.pct_away}% away</span>
          <span class="sr-touches">${s.touches}x</span>
        </div>`;
      });
    }
    document.getElementById("sr-content").innerHTML = srHtml ||
      `<span style="color:var(--muted);font-size:.83rem;">No significant levels found</span>`;

  } catch(e) {
    document.getElementById("analysis-content").innerHTML =
      `<span style="color:var(--down);font-size:.83rem;">Analysis failed: ${e.message}</span>`;
  }
}

// ── Scan ──────────────────────────────────────────────────────
function showScan(type) {
  const titles = {surge:"Surge Scan", market:"Market Scan", movers:"Today's Movers (+10% moves)", trending:"Trend Breakouts (MANE pattern)"};
  document.getElementById("scan-overlay").style.display = "block";
  document.getElementById("scan-title").textContent     = titles[type] || "Market Scan";
  document.getElementById("scan-loading").style.display = "block";
  document.getElementById("scan-cards").innerHTML       = "";
  document.getElementById("regime-us-banner").innerHTML    = "";
  document.getElementById("regime-india-banner").innerHTML = "";
  const urls = {surge:"/api/surge", market:"/api/scan", movers:"/api/movers", trending:"/api/trending"};
  fetch(urls[type] || "/api/scan")
    .then(r => r.json())
    .then(data => {
      document.getElementById("scan-loading").style.display = "none";
      if (type === "surge")    renderSurge(data);
      else if (type === "movers")   renderMovers(data);
      else if (type === "trending") renderTrending(data);
      else renderScan(data);
    })
    .catch(e => {
      document.getElementById("scan-loading").innerHTML =
        `<span style="color:var(--down)">Scan failed: ${e.message}</span>`;
    });
}

function closeScan() { document.getElementById("scan-overlay").style.display = "none"; }
document.getElementById("scan-overlay").addEventListener("click", function(e){ if(e.target===this) closeScan(); });

function regimeBanner(r, id) {
  if (!r) return;
  const cls = {up:"up",down:"down",neutral:"neutral"}[r.color] || "neutral";
  const fmt = v => (v == null ? "n/a" : v);
  document.getElementById(id).innerHTML = `
    <div class="regime-banner ${cls}" style="margin-bottom:.6rem;">
      <strong>${r.label||r.regime}</strong>
      &nbsp;RSI ${fmt(r.rsi)} &middot; 5d ${fmt(r.ret5)}% &middot; vs SMA50 ${fmt(r.vs_sma50)}%
      <span style="color:var(--muted);font-size:.78rem;">&mdash; ${r.desc||""}</span>
    </div>`;
}

function verdictLabel(buy, sell) {
  if (buy >= 4 && buy > sell)  return "BUY";
  if (sell >= 4 && sell > buy) return "SELL";
  if (buy >= 3 && buy > sell)  return "BUY";
  if (sell >= 3 && sell > buy) return "SELL";
  return "WATCH";
}

function scanCard(r, role) {
  const v = role || verdictLabel(r.buy_score, r.sell_score);
  let srHtml = "";
  if (r.resistance && r.resistance[0])
    srHtml += `<div class="sc-sr-line res">R: ${r.ticker.endsWith(".NS")||r.ticker.endsWith(".BO")?"₹":"$"}${r.resistance[0].price} (+${r.resistance[0].pct_away}%)</div>`;
  if (r.support && r.support[0])
    srHtml += `<div class="sc-sr-line sup">S: ${r.ticker.endsWith(".NS")||r.ticker.endsWith(".BO")?"₹":"$"}${r.support[0].price} (-${r.support[0].pct_away}%)</div>`;
  return `<div class="scan-card">
    <div class="sc-top">
      <span class="sc-ticker">${r.ticker.replace(".NS","")}</span>
      <span class="sc-badge ${v}">${v}</span>
    </div>
    <div class="sc-price">$${r.price.toLocaleString()}</div>
    <div class="sc-row">Buy <span>${r.buy_score}</span> &nbsp; Sell <span>${r.sell_score}</span></div>
    <div class="sc-row">RSI <span>${r.rsi}</span></div>
    <div class="sc-row">MACD <span>${r.macd_hist>0?"▲ bull":"▼ bear"}</span> &nbsp; Vol <span>${r.vol_ratio}x</span></div>
    <div class="sc-row">5d <span>${r.ret5}%</span>${r.pcr!=null?` &nbsp; PCR <span>${r.pcr}</span>`:""}</div>
    ${srHtml ? `<div class="sc-sr">${srHtml}</div>` : ""}
  </div>`;
}

function renderScan(data) {
  regimeBanner(data.us?.regime,    "regime-us-banner");
  regimeBanner(data.india?.regime, "regime-india-banner");
  let html = "";
  const add = (r, role, label) => {
    if (!r) return;
    html += `<div><div style="font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">${label}</div>${scanCard(r,role)}</div>`;
  };
  add(data.us?.buy,    "BUY",  "&#127482;&#127480; US Best Buy");
  add(data.us?.sell,   "SELL", "&#127482;&#127480; US Best Sell");
  add(data.india?.buy, "BUY",  "&#127470;&#127475; India Best Buy");
  add(data.india?.sell,"SELL", "&#127470;&#127475; India Best Sell");
  (data.us?.watch_buy    ||[]).forEach(r=>add(r,"WATCH","&#127482;&#127480; Watch Buy"));
  (data.us?.watch_sell   ||[]).forEach(r=>add(r,"WATCH","&#127482;&#127480; Watch Sell"));
  (data.india?.watch_buy ||[]).forEach(r=>add(r,"WATCH","&#127470;&#127475; Watch Buy"));
  (data.india?.watch_sell||[]).forEach(r=>add(r,"WATCH","&#127470;&#127475; Watch Sell"));
  document.getElementById("scan-cards").innerHTML = html || `<div style="color:var(--muted);">No strong signals found</div>`;
  const meta = document.createElement("div");
  meta.style.cssText = "font-size:.72rem;color:var(--muted);margin-top:.8rem;grid-column:1/-1;text-align:center;";
  meta.textContent   = `Scanned ${data.us?.scanned||0} US + ${data.india?.scanned||0} India stocks · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("scan-cards").appendChild(meta);
}

function renderTrending(data) {
  const list = data.trending || [];
  let html = "";
  list.forEach(r => {
    const sqzColor = r.bb_squeeze < 0.70 ? "var(--buy)" : r.bb_squeeze < 0.85 ? "var(--neutral)" : "var(--text)";
    const highColor = r.pct_from_high >= -5 ? "var(--buy)" : r.pct_from_high >= -12 ? "var(--neutral)" : "var(--muted)";
    html += `<div>
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">
        Score ${r.score} — Trend Breakout Setup
      </div>
      <div class="scan-card">
        <div class="sc-top">
          <span class="sc-ticker">${r.ticker}</span>
          <span class="sc-badge BUY">TREND</span>
        </div>
        <div class="sc-price">$${r.price}</div>
        <div class="sc-row">From 52w high <span style="color:${highColor}">${r.pct_from_high}%</span></div>
        <div class="sc-row">BB <span style="color:${sqzColor}">${r.bb_squeeze}</span> &nbsp; ATR <span>${r.atr_ratio}</span></div>
        <div class="sc-row">RSI <span>${r.rsi}</span> &nbsp; 20d <span style="color:var(--buy)">+${r.ret20}%</span></div>
        <div class="sc-row">Vol <span>${r.vol_ratio}x</span> &nbsp; Accum <span style="color:${r.accumulation?'var(--buy)':'var(--muted)'}">${r.accumulation?'✓':'—'}</span></div>
        <div style="font-size:.68rem;color:var(--buy);margin-top:.3rem;">${r.reasons}</div>
      </div>
    </div>`;
  });
  document.getElementById("scan-cards").innerHTML = html ||
    `<div style="color:var(--muted);">No trend breakout setups found — check during market hours</div>`;
  const meta = document.createElement("div");
  meta.style.cssText = "font-size:.72rem;color:var(--muted);margin-top:.8rem;grid-column:1/-1;text-align:center;";
  meta.textContent = `Scanned ${data.scanned||0} stocks · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("scan-cards").appendChild(meta);
}

function renderMovers(data) {
  const list = data.movers || [];
  if (data.error) {
    document.getElementById("scan-cards").innerHTML = `<div style="color:var(--sell)">${data.error}</div>`;
    return;
  }
  let html = "";
  list.forEach(r => {
    const capColor = r.mktcap_raw < 300e6 ? "var(--sell)" : r.mktcap_raw < 2e9 ? "var(--neutral)" : "var(--muted)";
    const capLabel = r.mktcap_raw < 300e6 ? "micro" : r.mktcap_raw < 2e9 ? "small" : r.mktcap_raw < 10e9 ? "mid" : "large";
    html += `<div>
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">
        +${r.change_pct}% today
      </div>
      <div class="scan-card">
        <div class="sc-top">
          <span class="sc-ticker">${r.ticker}</span>
          <span class="sc-badge BUY">+${r.change_pct}%</span>
        </div>
        <div style="font-size:.72rem;color:var(--muted);margin-bottom:.3rem;">${r.name}</div>
        <div class="sc-price">$${r.price}</div>
        <div class="sc-row">Vol <span>${fmtVol(r.volume)}</span> <span style="color:var(--buy)">(${r.vol_ratio}x avg)</span></div>
        <div class="sc-row">Cap <span style="color:${capColor}">${r.mktcap} (${capLabel})</span></div>
      </div>
    </div>`;
  });
  document.getElementById("scan-cards").innerHTML = html ||
    `<div style="color:var(--muted);">No 10%+ movers right now — check during market hours (9:30 AM–4 PM ET)</div>`;
  const meta = document.createElement("div");
  meta.style.cssText = "font-size:.72rem;color:var(--muted);margin-top:.8rem;grid-column:1/-1;text-align:center;";
  meta.textContent = `Live Yahoo Finance gainers · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("scan-cards").appendChild(meta);
}

function fmtVol(v) {
  if (v == null) return "n/a";
  if (v >= 1e9) return (v/1e9).toFixed(1)+"B";
  if (v >= 1e6) return (v/1e6).toFixed(1)+"M";
  if (v >= 1e3) return (v/1e3).toFixed(0)+"K";
  return v;
}

function renderSurge(data) {
  const list = data.surges || [];
  let html = "";
  list.forEach(r => {
    const sym      = r.ticker.endsWith(".NS") || r.ticker.endsWith(".BO") ? "₹" : "$";
    const sqzColor = r.bb_squeeze < 0.70 ? "var(--buy)" : r.bb_squeeze < 0.85 ? "var(--neutral)" : "var(--text)";
    const sqzLabel = r.bb_squeeze < 0.55 ? "TIGHT" : r.bb_squeeze < 0.70 ? "squeeze" : r.bb_squeeze < 0.85 ? "tightening" : "normal";
    const atrColor = r.atr_ratio  < 0.70 ? "var(--buy)" : r.atr_ratio  < 0.85 ? "var(--neutral)" : "var(--text)";
    const cprStr  = r.call_put_ratio != null ? ` &nbsp; C/P <span>${r.call_put_ratio}x</span>` : "";
    const earnStr = r.days_to_earnings != null
      ? (r.days_to_earnings <= 7
          ? `<div class="sc-row" style="color:var(--sell)">&#9888; Earnings in <span>${r.days_to_earnings}d</span></div>`
          : `<div class="sc-row">Earnings <span>${r.days_to_earnings}d away</span></div>`)
      : "";
    const stopStr = r.stop_loss != null
      ? `<div class="sc-row">Stop <span style="color:var(--sell)">${sym}${r.stop_loss}</span> <span style="color:var(--muted)">(-${r.stop_pct}%)</span></div>`
      : "";
    html += `<div>
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">
        Score ${r.score} — Pre-Surge Setup
      </div>
      <div class="scan-card">
        <div class="sc-top">
          <span class="sc-ticker">${r.ticker}</span>
          <span class="sc-badge BUY">SETUP</span>
        </div>
        <div class="sc-price">${sym}${r.price.toLocaleString()}</div>
        <div class="sc-row">BB <span style="color:${sqzColor}">${sqzLabel} (${r.bb_squeeze})</span> &nbsp; ATR <span style="color:${atrColor}">${r.atr_ratio}</span></div>
        <div class="sc-row">RSI <span>${r.rsi}</span> &nbsp; MACD <span style="color:${r.macd_bull?'var(--buy)':'var(--muted)'}">${r.macd_bull?'▲ bull':'▼ bear'}</span></div>
        <div class="sc-row">Vol <span>${fmtVol(r.vol_today)}</span> <span style="color:var(--muted);font-size:.7rem">(${r.vol_ratio}x avg)</span>${cprStr}</div>
        <div class="sc-row">5d <span>${r.ret5>0?"+":""}${r.ret5}%</span> &nbsp; 1d <span>${r.ret1>0?"+":""}${r.ret1}%</span></div>
        ${stopStr}
        ${earnStr}
        <div style="font-size:.68rem;color:var(--buy);margin-top:.3rem;">${r.reasons}</div>
      </div>
    </div>`;
  });
  document.getElementById("scan-cards").innerHTML = html || `<div style="color:var(--muted);">No pre-surge setups found right now — check back later</div>`;
  const meta = document.createElement("div");
  meta.style.cssText = "font-size:.72rem;color:var(--muted);margin-top:.8rem;grid-column:1/-1;text-align:center;";
  meta.textContent   = `Scanned ${data.scanned||0} stocks · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("scan-cards").appendChild(meta);
}

// ── India Options ─────────────────────────────────────────────
function processChain(rows, spot, expStr) {
  const calls = {}, puts = {};
  for (const row of rows) {
    if (row.expiryDate !== expStr) continue;
    const strike = parseInt(row.strikePrice);
    if (spot && Math.abs(strike - spot) / spot > 0.15) continue;
    if (row.CE) calls[strike] = { oi: row.CE.openInterest||0, vol: row.CE.totalTradedVolume||0, ltp: row.CE.lastPrice||0 };
    if (row.PE) puts[strike]  = { oi: row.PE.openInterest||0, vol: row.PE.totalTradedVolume||0, ltp: row.PE.lastPrice||0 };
  }
  const totalCE = Object.values(calls).reduce((s,v)=>s+v.oi,0);
  const totalPE = Object.values(puts).reduce((s,v)=>s+v.oi,0);
  const pcr = totalCE > 0 ? Math.round(totalPE/totalCE*1000)/1000 : null;

  const strikes = [...new Set([...Object.keys(calls),...Object.keys(puts)].map(Number))].sort((a,b)=>a-b);
  let mpVal = Infinity, mpStrike = null;
  for (const s of strikes) {
    const loss = Object.entries(calls).reduce((t,[k,v])=>t+Math.max(s-k,0)*v.oi,0)
               + Object.entries(puts).reduce((t,[k,v])=>t+Math.max(k-s,0)*v.oi,0);
    if (loss < mpVal) { mpVal = loss; mpStrike = s; }
  }

  let signal, pcrText;
  if      (pcr === null) { signal="HOLD"; pcrText="n/a"; }
  else if (pcr >= 1.5)   { signal="BUY";  pcrText=`${pcr} — Strong support`; }
  else if (pcr >= 1.1)   { signal="BUY";  pcrText=`${pcr} — Moderate support`; }
  else if (pcr <= 0.6)   { signal="SELL"; pcrText=`${pcr} — Strong resistance`; }
  else if (pcr <= 0.9)   { signal="SELL"; pcrText=`${pcr} — Moderate resistance`; }
  else                   { signal="HOLD"; pcrText=`${pcr} — Neutral`; }

  const topCE = Object.entries(calls).sort((a,b)=>b[1].oi-a[1].oi).slice(0,3).map(([k,v])=>({strike:+k,...v}));
  const topPE = Object.entries(puts).sort((a,b)=>b[1].oi-a[1].oi).slice(0,3).map(([k,v])=>({strike:+k,...v}));
  return { expiry:expStr, signal, pcr, pcr_text:pcrText, total_call_oi:totalCE, total_put_oi:totalPE,
           max_pain:mpStrike, ce_resistance:topCE, pe_support:topPE };
}

function renderExpiry(label, exp) {
  if (!exp) return "";
  if (exp.error) return `<div style="color:var(--muted);font-size:.8rem;margin-bottom:.8rem;">${label}: ${exp.error}</div>`;
  const sc = exp.signal==="BUY"?"var(--up)":exp.signal==="SELL"?"var(--down)":"var(--neutral)";
  const bg = exp.signal==="BUY"?"rgba(34,197,94,.12)":exp.signal==="SELL"?"rgba(239,68,68,.12)":"rgba(245,158,11,.12)";
  return `<div style="background:${bg};border-radius:10px;padding:.7rem .9rem;margin-bottom:.8rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
      <span style="font-size:.75rem;font-weight:700;text-transform:uppercase;color:var(--muted);">${label} · ${exp.expiry}</span>
      <span style="font-weight:800;color:${sc};font-size:.95rem;">${exp.signal}</span>
    </div>
    <div style="font-size:.8rem;display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;margin-bottom:.5rem;">
      <span style="color:var(--muted);">PCR</span><span style="font-weight:700;">${exp.pcr_text||exp.pcr||"—"}</span>
      <span style="color:var(--muted);">Max Pain</span><span style="font-weight:700;">${exp.max_pain?"&#8377;"+Number(exp.max_pain).toLocaleString():"—"}</span>
      <span style="color:var(--muted);">Total CE OI</span><span>${(exp.total_call_oi||0).toLocaleString()}</span>
      <span style="color:var(--muted);">Total PE OI</span><span>${(exp.total_put_oi||0).toLocaleString()}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;">
      <div><div style="font-size:.68rem;font-weight:700;text-transform:uppercase;color:var(--down);margin-bottom:.25rem;">CE Resistance</div>
        ${(exp.ce_resistance||[]).map(r=>`<div style="display:flex;justify-content:space-between;background:rgba(239,68,68,.08);border-radius:5px;padding:.2rem .4rem;margin-bottom:.2rem;font-size:.78rem;"><span style="font-weight:700;">&#8377;${r.strike.toLocaleString()}</span><span style="color:var(--muted);">OI ${(r.oi/1e5).toFixed(1)}L</span></div>`).join("")}
      </div>
      <div><div style="font-size:.68rem;font-weight:700;text-transform:uppercase;color:var(--up);margin-bottom:.25rem;">PE Support</div>
        ${(exp.pe_support||[]).map(r=>`<div style="display:flex;justify-content:space-between;background:rgba(34,197,94,.08);border-radius:5px;padding:.2rem .4rem;margin-bottom:.2rem;font-size:.78rem;"><span style="font-weight:700;">&#8377;${r.strike.toLocaleString()}</span><span style="color:var(--muted);">OI ${(r.oi/1e5).toFixed(1)}L</span></div>`).join("")}
      </div>
    </div>
  </div>`;
}

async function fetchAndRenderIndex(symbol, displayName) {
  const r = await fetch(`${NSE_PROXY}/options/${symbol}`, {headers:{"ngrok-skip-browser-warning":"1"}});
  const rec = await r.json();
  if (!r.ok) throw new Error(rec.error || `Proxy returned ${r.status}`);
  if (rec.error) throw new Error(rec.error);

  const spot = parseFloat(rec.underlyingValue || 0);
  const expiries = (rec.expiryDates || []);
  const rows = rec.data || [];

  // Always compare against Mumbai time (IST = UTC+5:30) regardless of viewer's location
  const nowIST = new Date(new Date().toLocaleString("en-US", {timeZone:"Asia/Kolkata"}));
  nowIST.setHours(0,0,0,0);
  const today = nowIST;

  const _mo = {Jan:'01',Feb:'02',Mar:'03',Apr:'04',May:'05',Jun:'06',Jul:'07',Aug:'08',Sep:'09',Oct:'10',Nov:'11',Dec:'12'};
  const parsed = expiries.map(e => {
    const parts = e.split('-');
    const iso = parts.length===3 ? `${parts[2]}-${_mo[parts[1]]||parts[1]}-${parts[0].padStart(2,'0')}` : e;
    return { str: e, dt: new Date(iso) };
  }).filter(e => !isNaN(e.dt) && e.dt >= today).sort((a,b)=>a.dt-b.dt);
  if (!parsed.length) throw new Error("No future expiries (got: "+expiries.slice(0,3).join(', ')+")");

  const weeklyStr  = parsed[0].str;
  const monthlyStr = parsed.find(e=>e.dt.getMonth()===today.getMonth())?.str || parsed[parsed.length-1].str;

  const weekly  = processChain(rows, spot, weeklyStr);
  const monthly = monthlyStr !== weeklyStr ? processChain(rows, spot, monthlyStr) : weekly;

  let html = `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;">
      <div class="card-title" style="margin:0;">${displayName}</div>
      <div style="font-size:1rem;font-weight:800;">&#8377;${spot.toLocaleString()}</div>
    </div>
    ${renderExpiry("Weekly", weekly)}
    ${monthlyStr !== weeklyStr ? renderExpiry("Monthly", monthly) : ""}
  </div>`;
  return html;
}

function showOptions() {
  if (!NSE_PROXY) {
    alert("NSE_PROXY_URL not configured in Vercel env vars.");
    return;
  }
  document.getElementById("options-overlay").style.display = "block";
  document.getElementById("options-loading").style.display = "block";
  document.getElementById("options-content").innerHTML     = "";

  Promise.all([
    fetchAndRenderIndex("NIFTY",     "Nifty 50"),
    fetchAndRenderIndex("BANKNIFTY", "Bank Nifty"),
  ]).then(([n, bn]) => {
    document.getElementById("options-loading").style.display = "none";
    document.getElementById("options-content").innerHTML = n + bn +
      `<div style="grid-column:1/-1;font-size:.72rem;color:var(--muted);text-align:center;margin-top:.4rem;">Updated ${new Date().toLocaleTimeString()}</div>`;
  }).catch(e => {
    document.getElementById("options-loading").innerHTML =
      `<span style="color:var(--sell)">Failed: ${e.message}<br><span style="font-size:.7rem;opacity:.7;">Check NSE_PROXY_URL in Vercel env vars</span></span>`;
  });
}
document.getElementById("options-overlay").addEventListener("click", function(e){ if(e.target===this) this.style.display="none"; });

// ── R:R Calculator ────────────────────────────────────────────
function calcRR() {
  const entry   = parseFloat(document.getElementById("rr-entry").value);
  const sl      = parseFloat(document.getElementById("rr-sl").value);
  const t1      = parseFloat(document.getElementById("rr-t1").value);
  const t2      = parseFloat(document.getElementById("rr-t2").value);
  const capital = parseFloat(document.getElementById("rr-capital").value) || 0;
  const riskPct = parseFloat(document.getElementById("rr-risk-pct").value) || 1;
  const el      = document.getElementById("rr-result");

  if (!entry || !sl || (!t1 && !t2)) { el.innerHTML = ""; return; }

  const risk = Math.abs(entry - sl);
  if (risk === 0) { el.innerHTML = "<div style='color:var(--sell)'>SL cannot equal entry</div>"; return; }

  const isLong = entry > sl;
  const rr1 = t1 ? (Math.abs(t1 - entry) / risk) : null;
  const rr2 = t2 ? (Math.abs(t2 - entry) / risk) : null;
  const bestRR = rr2 || rr1;
  const go = bestRR >= 2;

  // Position sizing
  let posHtml = "";
  if (capital > 0) {
    const riskAmt  = capital * riskPct / 100;
    const qty      = Math.floor(riskAmt / risk);
    const invest   = qty * entry;
    const maxLoss  = qty * risk;
    const profitT1 = t1 ? qty * Math.abs(t1 - entry) : 0;
    const profitT2 = t2 ? qty * Math.abs(t2 - entry) : 0;
    const sym = entry > 100 ? "₹" : "$";
    posHtml = `
      <div style="margin-top:.8rem;padding:.7rem;background:var(--bg);border-radius:8px;font-size:.78rem;">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;">
          <span style="color:var(--muted);">Qty</span><span style="font-weight:700;">${qty} shares</span>
          <span style="color:var(--muted);">Investment</span><span>${sym}${invest.toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          <span style="color:var(--muted);">Max Loss</span><span style="color:var(--sell);">${sym}${maxLoss.toLocaleString(undefined,{maximumFractionDigits:0})}</span>
          ${profitT1?`<span style="color:var(--muted);">Profit T1</span><span style="color:var(--buy);">${sym}${profitT1.toLocaleString(undefined,{maximumFractionDigits:0})}</span>`:""}
          ${profitT2?`<span style="color:var(--muted);">Profit T2</span><span style="color:var(--buy);">${sym}${profitT2.toLocaleString(undefined,{maximumFractionDigits:0})}</span>`:""}
        </div>
      </div>`;
  }

  el.innerHTML = `
    <div style="text-align:center;padding:1rem;border-radius:12px;background:${go?"rgba(74,222,128,.15)":"rgba(239,68,68,.15)"};border:2px solid ${go?"var(--buy)":"var(--sell)"};">
      <div style="font-size:2rem;font-weight:900;color:${go?"var(--buy)":"var(--sell)"};">${go?"GO ✓":"SKIP ✗"}</div>
      <div style="font-size:.78rem;color:var(--muted);margin-top:.2rem;">${isLong?"Long":"Short"} · Risk per share: ${risk.toFixed(2)}</div>
    </div>
    <div style="display:grid;grid-template-columns:${t1&&t2?"1fr 1fr":"1fr"};gap:.6rem;margin-top:.8rem;">
      ${rr1?`<div style="padding:.6rem;background:var(--bg);border-radius:8px;text-align:center;">
        <div style="font-size:.7rem;color:var(--muted);">Target 1 R:R</div>
        <div style="font-size:1.3rem;font-weight:800;color:${rr1>=1.5?"var(--buy)":"var(--sell)"};">${rr1.toFixed(2)}:1</div>
      </div>`:""}
      ${rr2?`<div style="padding:.6rem;background:var(--bg);border-radius:8px;text-align:center;">
        <div style="font-size:.7rem;color:var(--muted);">Target 2 R:R</div>
        <div style="font-size:1.3rem;font-weight:800;color:${rr2>=2?"var(--buy)":"var(--sell)"};">${rr2.toFixed(2)}:1</div>
      </div>`:""}
    </div>
    ${posHtml}`;
}

document.getElementById("rr-overlay").addEventListener("click", function(e) {
  if (e.target === this) this.style.display = "none";
});

// ── Live Portfolio P/L ─────────────────────────────────────────
function loadPortfolio() {
  fetch("/api/portfolio")
    .then(r => r.json())
    .then(data => {
      // Update sidebar badges with live pnl_pct + show $ amount
      const byTicker = {};
      (data.positions || []).forEach(p => { byTicker[p.ticker] = p; });

      UNIVERSE.us["⭐ My Portfolio"].forEach(s => {
        const p = byTicker[s.t];
        if (p && p.pnl_pct != null) {
          s.pl     = p.pnl_pct;
          s.pnlUsd = p.pnl_usd;
          s.pre    = p.pre;
        }
      });
      renderSidebar(currentTab, document.getElementById("stock-search").value);

      // Overall summary bar
      const el = document.getElementById("portfolio-overall");
      if (el && data.total_pct != null) {
        const pnlSign  = data.total_pnl_usd >= 0 ? "+" : "-";
        const pnlColor = data.total_pnl_usd >= 0 ? "var(--buy)" : "var(--sell)";
        const curVal   = (data.total_invested + data.total_pnl_usd).toFixed(0);
        el.innerHTML =
          `<div style="font-size:.68rem;color:var(--muted);">$${data.total_invested.toFixed(0)} invested</div>` +
          `<div style="font-size:.72rem;color:${pnlColor};font-weight:700;">` +
          `P&amp;L ${pnlSign}$${Math.abs(data.total_pnl_usd).toFixed(0)} / ${pnlSign}€${Math.abs(data.total_pnl_eur).toFixed(0)} ` +
          `(${data.total_pct >= 0 ? "+" : ""}${data.total_pct.toFixed(1)}%)` +
          `</div>`;
      }
    }).catch(() => {});
}

// Init
renderSidebar("us","");
loadPortfolio();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)
