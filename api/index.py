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
_cache      = {}
_cache_date = {}   # tracks which date each ticker was last trained
_lock       = threading.Lock()

STOCKS     = ["AAPL", "MSFT", "TSLA", "NVDA", "^NSEI", "BDMD"]
START_DATE = "2022-01-01"   # 2 years keeps training fast for serverless
SR_START   = "2018-01-01"   # 6+ years of history for support/resistance

# ── Market scan universes ───────────────────────────────────────────────────────
US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "AMD",
    "JPM", "BAC", "GS", "MS", "V", "MA",
    "XOM", "CVX", "PFE", "JNJ", "UNH",
    "SPY", "QQQ", "ARKK", "BDMD",
]
PENNY_UNIVERSE = [
    "SNDL", "CLOV", "ABAT", "IMPP", "ZENA", "PHUN",
    "NAKD", "EXPR", "BBBY", "AMC", "GME", "WISH",
    "NKLA", "WKHS", "RIDE", "GOEV", "XELA", "GFAI",
    "BNGO", "SNGX", "OBSV", "CRBP", "ADMA", "AEYE",
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


def get_sentiment(ticker: str) -> tuple[float, list]:
    """Return (avg_sentiment_score, list_of_headlines)"""
    try:
        news = yf.Ticker(ticker).news or []
        headlines, scores = [], []
        for a in news[:10]:
            # yfinance changed its news structure — try all known formats
            title = (
                a.get("title")
                or a.get("content", {}).get("title")
                or a.get("headline")
                or ""
            )
            if not title:
                continue
            score = TextBlob(title).sentiment.polarity
            headlines.append({"title": title, "score": round(score, 3)})
            scores.append(score)
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
        resistances = [r for r in resistances if r["price"] > current_price]
        supports    = [s for s in supports    if s["price"] < current_price]

        # Sort by proximity first, then take top 3 regardless of distance
        # (filter extreme outliers beyond 40% — those are from a different price era)
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
        macd      = ema12 - ema26
        macd_hist = float((macd - macd.ewm(span=9).mean()).iloc[-1])

        rm  = close.rolling(20).mean()
        rs  = close.rolling(20).std()
        bb_series = (close - (rm - 2*rs)) / (4*rs)
        bb_pos    = float(bb_series.iloc[-1])

        vol_ratio = float((volume / volume.rolling(10).mean()).iloc[-1])
        price     = round(float(close.iloc[-1]), 2)

        # ── Z-scores: compare current value vs this stock's own history ──
        # This prevents flagging persistently strong stocks (UNH, HCLTECH)
        # as overbought just because their normal RSI runs high.
        rsi_mean  = float(rsi_series.mean())
        rsi_std   = float(rsi_series.std())
        rsi_z     = (rsi - rsi_mean) / rsi_std if rsi_std > 0 else 0.0

        bb_mean   = float(bb_series.mean())
        bb_std    = float(bb_series.std())
        bb_z      = (bb_pos - bb_mean) / bb_std if bb_std > 0 else 0.0

        sma_series = close.rolling(10).mean() / close.rolling(50).mean()
        sma_mean   = float(sma_series.mean())
        sma_std    = float(sma_series.std())
        sma_z      = (sma_ratio - sma_mean) / sma_std if sma_std > 0 else 0.0

        pcr = get_pcr(ticker)

        # ── Buy score (z-score based) ────────────────────────────────
        # RSI z < -1.5 means genuinely oversold *for this stock*
        buy = 0.0
        if rsi_z < -2.0:   buy += 3.0
        elif rsi_z < -1.5: buy += 1.5
        elif rsi_z < -1.0: buy += 0.5
        if macd_hist > 0:      buy += 1.5
        if bb_z < -1.5:        buy += 2.0
        elif bb_z < -1.0:      buy += 1.0
        if sma_z > 0.5:        buy += 1.0   # short-term trend above its own norm
        if vol_ratio > 2.0:    buy += 1.0
        elif vol_ratio > 1.5:  buy += 0.5
        if ret5 < -0.05:       buy += 1.5
        if pcr is not None:
            if pcr > 1.3:      buy += 1.5
            elif pcr > 1.0:    buy += 0.5

        # ── Sell score (z-score based) ───────────────────────────────
        # RSI z > +1.5 means genuinely overbought *for this stock*
        sell = 0.0
        if rsi_z > 2.0:    sell += 3.0
        elif rsi_z > 1.5:  sell += 1.5
        elif rsi_z > 1.0:  sell += 0.5
        if macd_hist < 0:      sell += 1.5
        if bb_z > 1.5:         sell += 2.0
        elif bb_z > 1.0:       sell += 1.0
        if sma_z < -0.5:       sell += 1.0  # short-term trend below its own norm
        if vol_ratio > 2.0:    sell += 1.0
        elif vol_ratio > 1.5:  sell += 0.5
        if ret5 > 0.08:        sell += 1.5
        if pcr is not None:
            if pcr < 0.6:      sell += 1.5
            elif pcr < 0.8:    sell += 0.5

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
            "rsi_z":      round(rsi_z, 2),
            "rsi_mean":   round(rsi_mean, 1),
            "bb_pos":     round(bb_pos, 3),
            "bb_z":       round(bb_z, 2),
            "macd_hist":  round(macd_hist, 4),
            "sma_ratio":  round(sma_ratio, 4),
            "sma_z":      round(sma_z, 2),
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

    MIN_BUY_SCORE  = 4.0
    MIN_SELL_SCORE = 4.0

    def pick(results):
        if not results:
            return None, None, [], []
        by_buy  = sorted(results, key=lambda x: x["buy_score"],  reverse=True)
        by_sell = sorted(results, key=lambda x: x["sell_score"], reverse=True)

        best_buy  = by_buy[0]  if by_buy[0]["buy_score"]   >= MIN_BUY_SCORE  else None
        best_sell = by_sell[0] if by_sell[0]["sell_score"] >= MIN_SELL_SCORE else None

        # Avoid showing the same ticker as both buy and sell
        if best_buy and best_sell and best_buy["ticker"] == best_sell["ticker"]:
            if best_buy["buy_score"] >= best_sell["sell_score"]:
                best_sell = next((r for r in by_sell[1:] if r["sell_score"] >= MIN_SELL_SCORE), None)
            else:
                best_buy  = next((r for r in by_buy[1:]  if r["buy_score"]  >= MIN_BUY_SCORE),  None)

        # Always return top 3 watch candidates with S/R regardless of threshold
        watch_buy  = [r for r in by_buy[:3]  if not best_buy  or r["ticker"] != best_buy["ticker"]][:3]
        watch_sell = [r for r in by_sell[:3] if not best_sell or r["ticker"] != best_sell["ticker"]][:3]

        return best_buy, best_sell, watch_buy, watch_sell

    us_buy,    us_sell,    us_watch_buy,    us_watch_sell    = pick(us_results)
    india_buy, india_sell, india_watch_buy, india_watch_sell = pick(india_results)

    us_regime    = get_market_regime("SPY")
    india_regime = get_market_regime("^NSEI")

    return jsonify(sanitize({
        "us":    {"buy": us_buy,    "sell": us_sell,    "watch_buy": us_watch_buy,    "watch_sell": us_watch_sell,    "scanned": len(us_results),    "regime": us_regime},
        "india": {"buy": india_buy, "sell": india_sell, "watch_buy": india_watch_buy, "watch_sell": india_watch_sell, "scanned": len(india_results), "regime": india_regime},
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


@app.route("/api/penny")
def api_penny():
    """Scan penny stock universe and return ranked buy/sell picks."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(scan_ticker, t): t for t in PENNY_UNIVERSE}
        for f in as_completed(futures):
            r = f.result()
            if r and r["price"] <= 10:   # enforce penny stock price cap
                results.append(r)

    # Rank top 3 buys and top 3 sells
    top_buys  = sorted([r for r in results if r["buy_score"]  >= 3.0], key=lambda x: x["buy_score"],  reverse=True)[:3]
    top_sells = sorted([r for r in results if r["sell_score"] >= 3.0], key=lambda x: x["sell_score"], reverse=True)[:3]

    return jsonify(sanitize({
        "buys":         top_buys,
        "sells":        top_sells,
        "scanned":      len(results),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


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

  .sr-section { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); }
  .sr-title { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 2px; }
  .sr-subtitle { font-size: 0.68rem; color: var(--muted); margin-bottom: 0.4rem; }
  .sr-price-label { font-size: 0.78rem; color: var(--muted); margin-bottom: 0.6rem; }
  .sr-group-label { font-size: 0.72rem; font-weight: 700; margin-bottom: 0.35rem; }
  .resistance-label { color: var(--down); }
  .support-label    { color: var(--up); }
  .sr-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.3rem; font-size: 0.78rem; }
  .sr-price  { font-weight: 700; width: 56px; }
  .sr-pct    { color: var(--muted); width: 68px; }
  .sr-touches { color: var(--muted); font-size: 0.7rem; width: 60px; }
  .sr-bar-track { flex: 1; background: var(--border); border-radius: 3px; height: 4px; }
  .sr-bar-fill  { height: 100%; border-radius: 3px; }
  .resistance-fill { background: var(--down); }
  .support-fill    { background: var(--up); }
  .sr-empty { font-size: 0.75rem; color: var(--muted); font-style: italic; }

  /* ── Signal Guide ── */
  #guide-section { margin-bottom: 2rem; }
  .guide-toggle {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1.2rem;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text);
    width: 100%;
    text-align: left;
  }
  .guide-toggle:hover { border-color: var(--accent); }
  .guide-toggle .chevron { transition: transform 0.2s; font-style: normal; }
  .guide-toggle.open .chevron { transform: rotate(180deg); }
  .guide-body {
    display: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.4rem;
  }
  .guide-body.open { display: block; }
  .guide-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.4rem; }
  .guide-group h4 { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: var(--accent); margin-bottom: 0.7rem; }
  .guide-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem; padding: 0.4rem 0; border-bottom: 1px solid var(--border); font-size: 0.8rem; }
  .guide-row:last-child { border-bottom: none; }
  .guide-signal { font-weight: 600; color: var(--text); min-width: 130px; }
  .guide-desc { color: var(--muted); line-height: 1.4; }
  .guide-pts { font-weight: 700; white-space: nowrap; }
  .guide-pts.bull { color: var(--up); }
  .guide-pts.bear { color: var(--down); }
  .guide-note { font-size: 0.75rem; color: var(--muted); margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--border); line-height: 1.6; }

  /* ── Regime Banner ── */
  .regime-banner {
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    font-size: 0.82rem;
    border: 1px solid;
  }
  .regime-banner.up      { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.3);  }
  .regime-banner.down    { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.3);  }
  .regime-banner.neutral { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); }
  .regime-icon  { font-size: 1.2rem; flex-shrink: 0; margin-top: 1px; }
  .regime-label { font-weight: 700; margin-bottom: 2px; }
  .regime-desc  { color: var(--muted); line-height: 1.4; }
  .regime-stats { display: flex; gap: 0.8rem; margin-top: 0.4rem; font-size: 0.72rem; color: var(--muted); }
  .regime-stats span { background: var(--bg); border-radius: 4px; padding: 0.1rem 0.4rem; }

  /* ── Market Scan ── */
  #scan-section { margin-bottom: 2.5rem; }
  .scan-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.2rem; margin-top: 1rem; }
  .scan-card { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.4rem; }
  .scan-card.buy-card  { border-top: 3px solid var(--up); }
  .scan-card.sell-card { border-top: 3px solid var(--down); }
  .scan-market-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.3rem; }
  .scan-action { font-size: 1rem; font-weight: 800; margin-bottom: 0.6rem; }
  .scan-action.buy  { color: var(--up); }
  .scan-action.sell { color: var(--down); }
  .scan-ticker { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.2rem; }
  .scan-price  { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.8rem; }
  .scan-score-row { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
  .scan-reason { margin-top: 0.8rem; font-size: 0.78rem; color: var(--muted); line-height: 1.5; }
  .scan-reason span { display: inline-block; background: var(--bg); border-radius: 4px; padding: 0.15rem 0.4rem; margin: 0.1rem 0.1rem 0.1rem 0; font-size: 0.72rem; }
  .scan-reason span.bull { color: var(--up); }
  .scan-reason span.bear { color: var(--down); }
  #scan-loading { text-align:center; padding: 2rem; color: var(--muted); font-size: 0.9rem; display:none; }
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
  <div style="display:flex;gap:0.6rem">
    <button id="penny-btn" onclick="runPennyScan()" style="background:#f59e0b;color:#000;border:none;padding:0.5rem 1.2rem;border-radius:8px;cursor:pointer;font-size:0.9rem;font-weight:600;transition:opacity 0.2s;">💰 Penny Scan</button>
    <button id="scan-btn" onclick="runScan()" style="background:#0ea5e9;color:#fff;border:none;padding:0.5rem 1.2rem;border-radius:8px;cursor:pointer;font-size:0.9rem;font-weight:600;transition:opacity 0.2s;">⚡ Market Scan</button>
    <button id="refresh-btn" onclick="loadPredictions()">↻ Refresh</button>
  </div>
</header>

<div id="status-bar">Loading predictions…</div>

<main>
  <div id="guide-section">
    <button class="guide-toggle" onclick="toggleGuide(this)">
      <span>📖 Signal &amp; Indicator Guide — what does each number mean?</span>
      <i class="chevron">▼</i>
    </button>
    <div class="guide-body">
      <div class="guide-cols">

        <div class="guide-group">
          <h4>Technical Indicators</h4>
          <div class="guide-row">
            <span class="guide-signal">RSI (14)</span>
            <span class="guide-desc">Relative Strength Index. Measures if a stock is overbought or oversold over 14 days.</span>
          </div>
          <div class="guide-row" style="padding-left:1rem">
            <span class="guide-signal" style="color:var(--up)">RSI &lt; 30</span>
            <span class="guide-desc">Oversold — potential bounce / buy zone</span>
          </div>
          <div class="guide-row" style="padding-left:1rem">
            <span class="guide-signal" style="color:var(--down)">RSI &gt; 70</span>
            <span class="guide-desc">Overbought — potential pullback / sell zone</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">MACD Histogram</span>
            <span class="guide-desc">Difference between fast &amp; slow momentum. Positive = bullish momentum building. Negative = bearish.</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">SMA Ratio</span>
            <span class="guide-desc">10-day avg ÷ 50-day avg. &gt;1 = short-term trend above long-term (Bullish). &lt;1 = Bearish.</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">BB Position</span>
            <span class="guide-desc">Where price sits inside Bollinger Bands. 0 = at lower band (oversold). 1 = at upper band (overbought). 0.5 = middle.</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">Vol Ratio</span>
            <span class="guide-desc">Today's volume ÷ 10-day average. 2x = twice normal trading activity — confirms momentum.</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">Volatility</span>
            <span class="guide-desc">10-day std dev of daily returns. Higher = bigger price swings, higher risk.</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">Return 1d / 5d</span>
            <span class="guide-desc">Price change over last 1 or 5 trading days as a %. Not importance — actual move.</span>
          </div>
          <div class="guide-row">
            <span class="guide-signal">Put/Call Ratio</span>
            <span class="guide-desc">Options market sentiment. &gt;1.2 = more puts (fear) = contrarian buy. &lt;0.7 = more calls (greed) = contrarian sell.</span>
          </div>
        </div>

        <div class="guide-group">
          <h4>Buy Signal Scoring (max 12 pts)</h4>
          <div class="guide-row"><span class="guide-signal">RSI &lt; 30</span><span class="guide-desc">Heavily oversold</span><span class="guide-pts bull">+3.0</span></div>
          <div class="guide-row"><span class="guide-signal">RSI 30–40</span><span class="guide-desc">Moderately oversold</span><span class="guide-pts bull">+1.5</span></div>
          <div class="guide-row"><span class="guide-signal">RSI 40–50</span><span class="guide-desc">Mildly oversold</span><span class="guide-pts bull">+0.5</span></div>
          <div class="guide-row"><span class="guide-signal">MACD &gt; 0</span><span class="guide-desc">Bullish momentum</span><span class="guide-pts bull">+1.5</span></div>
          <div class="guide-row"><span class="guide-signal">BB Pos &lt; 0.2</span><span class="guide-desc">Near lower band (dip zone)</span><span class="guide-pts bull">+2.0</span></div>
          <div class="guide-row"><span class="guide-signal">BB Pos 0.2–0.4</span><span class="guide-desc">Below midband</span><span class="guide-pts bull">+1.0</span></div>
          <div class="guide-row"><span class="guide-signal">SMA Ratio &gt; 1</span><span class="guide-desc">Short-term trend bullish</span><span class="guide-pts bull">+1.0</span></div>
          <div class="guide-row"><span class="guide-signal">Vol &gt; 2x</span><span class="guide-desc">Volume confirms move</span><span class="guide-pts bull">+1.0</span></div>
          <div class="guide-row"><span class="guide-signal">5d Return &lt; -5%</span><span class="guide-desc">Sharp dip — bounce candidate</span><span class="guide-pts bull">+1.5</span></div>
          <div class="guide-row"><span class="guide-signal">PCR &gt; 1.3</span><span class="guide-desc">Extreme fear in options</span><span class="guide-pts bull">+1.5</span></div>
        </div>

        <div class="guide-group">
          <h4>Sell Signal Scoring (max 12 pts)</h4>
          <div class="guide-row"><span class="guide-signal">RSI &gt; 70</span><span class="guide-desc">Heavily overbought</span><span class="guide-pts bear">+3.0</span></div>
          <div class="guide-row"><span class="guide-signal">RSI 60–70</span><span class="guide-desc">Moderately overbought</span><span class="guide-pts bear">+1.5</span></div>
          <div class="guide-row"><span class="guide-signal">RSI 55–60</span><span class="guide-desc">Mildly overbought</span><span class="guide-pts bear">+0.5</span></div>
          <div class="guide-row"><span class="guide-signal">MACD &lt; 0</span><span class="guide-desc">Bearish momentum</span><span class="guide-pts bear">+1.5</span></div>
          <div class="guide-row"><span class="guide-signal">BB Pos &gt; 0.8</span><span class="guide-desc">Near upper band (extended)</span><span class="guide-pts bear">+2.0</span></div>
          <div class="guide-row"><span class="guide-signal">BB Pos 0.6–0.8</span><span class="guide-desc">Above midband</span><span class="guide-pts bear">+1.0</span></div>
          <div class="guide-row"><span class="guide-signal">SMA Ratio &lt; 1</span><span class="guide-desc">Short-term trend bearish</span><span class="guide-pts bear">+1.0</span></div>
          <div class="guide-row"><span class="guide-signal">Vol &gt; 2x</span><span class="guide-desc">Volume confirms move</span><span class="guide-pts bear">+1.0</span></div>
          <div class="guide-row"><span class="guide-signal">5d Return &gt; +8%</span><span class="guide-desc">Extended run — pullback risk</span><span class="guide-pts bear">+1.5</span></div>
          <div class="guide-row"><span class="guide-signal">PCR &lt; 0.6</span><span class="guide-desc">Extreme greed in options</span><span class="guide-pts bear">+1.5</span></div>

          <h4 style="margin-top:1.2rem">Feature Importance Chart</h4>
          <div class="guide-row"><span class="guide-signal">Bar width %</span><span class="guide-desc">How much the ML model relies on that indicator when predicting UP or DOWN. Not the actual value of the indicator.</span></div>
          <div class="guide-row"><span class="guide-signal">Model Confidence</span><span class="guide-desc">How certain the RandomForest model is about tomorrow's direction. Based on vote share across 50 decision trees.</span></div>
          <div class="guide-row"><span class="guide-signal">Backtest Accuracy</span><span class="guide-desc">How often the model was correct on historical test data (last 20% of price history).</span></div>
        </div>

      </div>
      <div class="guide-note">
        ⚠️ <strong>Disclaimer:</strong> This dashboard is for educational and research purposes only. Signals are generated from technical indicators and ML models — they do not constitute financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.
      </div>
    </div>
  </div>

  <div id="scan-section" style="display:none">
    <p class="section-title">⚡ Best Market Picks</p>
    <div id="scan-loading"><div class="spinner"></div>Scanning 40+ stocks across US &amp; India markets…</div>
    <div class="scan-grid" id="scan-cards"></div>
  </div>

  <div id="penny-section" style="display:none">
    <p class="section-title">💰 Penny Stock Picks <span style="font-size:0.72rem;color:var(--muted);font-weight:400">(US stocks ≤$10 · high risk, high reward)</span></p>
    <div id="penny-loading" style="display:none;text-align:center;padding:1.5rem;color:var(--muted);font-size:0.9rem"><div class="spinner"></div>Scanning penny stocks…</div>
    <div id="penny-buy-wrap">
      <p style="font-size:0.8rem;font-weight:700;color:var(--up);margin-bottom:0.6rem">↑ TOP BUY SETUPS</p>
      <div class="scan-grid" id="penny-buys"></div>
    </div>
    <div id="penny-sell-wrap" style="margin-top:1.2rem">
      <p style="font-size:0.8rem;font-weight:700;color:var(--down);margin-bottom:0.6rem">↓ TOP SELL / AVOID</p>
      <div class="scan-grid" id="penny-sells"></div>
    </div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:0.8rem;padding:0.6rem;background:rgba(239,68,68,0.08);border-radius:8px;border:1px solid rgba(239,68,68,0.2)">
      ⚠️ Penny stocks are extremely volatile and speculative. Low volume means prices can move dramatically on small orders. Never risk more than you can afford to lose entirely.
    </div>
  </div>

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

        <div class="sr-section">
          <div class="sr-title">Historical Support &amp; Resistance</div>
          <div class="sr-subtitle">Based on ${p.as_of.slice(0,4) - 2018}+ years of swing highs/lows · touches = times tested</div>
          <div class="sr-price-label">Current: <strong>$${p.current_price}</strong></div>

          <div class="sr-group">
            <div class="sr-group-label resistance-label">⬆ Resistance (price ceiling)</div>
            ${p.resistance.length ? p.resistance.map(r => `
              <div class="sr-row resistance-row">
                <span class="sr-price">$${r.price}</span>
                <span class="sr-pct">+${r.pct_away}% away</span>
                <span class="sr-touches">${r.touches} touches</span>
                <div class="sr-bar-track"><div class="sr-bar-fill resistance-fill" style="width:${Math.min(r.touches*10,100)}%"></div></div>
              </div>`).join("") : '<div class="sr-empty">No resistance found above current price</div>'}
          </div>

          <div class="sr-group" style="margin-top:0.6rem">
            <div class="sr-group-label support-label">⬇ Support (price floor)</div>
            ${p.support.length ? p.support.map(s => `
              <div class="sr-row support-row">
                <span class="sr-price">$${s.price}</span>
                <span class="sr-pct">-${s.pct_away}% away</span>
                <span class="sr-touches">${s.touches} touches</span>
                <div class="sr-bar-track"><div class="sr-bar-fill support-fill" style="width:${Math.min(s.touches*10,100)}%"></div></div>
              </div>`).join("") : '<div class="sr-empty">No support found below current price</div>'}
          </div>
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

async function runPennyScan() {
  const btn    = document.getElementById("penny-btn");
  const sec    = document.getElementById("penny-section");
  const loader = document.getElementById("penny-loading");

  btn.disabled = true;
  btn.textContent = "Scanning…";
  sec.style.display = "block";
  loader.style.display = "block";
  document.getElementById("penny-buys").innerHTML  = "";
  document.getElementById("penny-sells").innerHTML = "";

  try {
    const res  = await fetch("/api/penny");
    const data = await res.json();
    loader.style.display = "none";
    renderPenny(data);
  } catch(e) {
    loader.style.display = "none";
    document.getElementById("penny-buys").innerHTML = `<div class="error-msg">Scan failed: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "💰 Penny Scan";
  }
}

function renderPenny(data) {
  const buysEl  = document.getElementById("penny-buys");
  const sellsEl = document.getElementById("penny-sells");

  if (!data.buys.length)  buysEl.innerHTML  = `<div class="scan-card" style="color:var(--muted);font-size:0.85rem;padding:1rem">No strong buy setups found right now</div>`;
  if (!data.sells.length) sellsEl.innerHTML = `<div class="scan-card" style="color:var(--muted);font-size:0.85rem;padding:1rem">No strong sell setups found right now</div>`;

  data.buys.forEach(s  => buysEl.innerHTML  += pennyCard(s, "buy"));
  data.sells.forEach(s => sellsEl.innerHTML += pennyCard(s, "sell"));

  const meta = document.createElement("div");
  meta.style.cssText = "font-size:0.72rem;color:var(--muted);margin-top:0.5rem";
  meta.textContent = `Scanned ${data.scanned} penny stocks · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("penny-sell-wrap").appendChild(meta);
}

function pennyCard(s, action) {
  const score   = action === "buy" ? s.buy_score : s.sell_score;
  const maxScore = 12;
  const pct     = Math.min(score / maxScore * 100, 100).toFixed(0);
  const volWarn = s.vol_ratio < 0.5 ? `<span style="color:var(--neutral);font-size:0.7rem">⚠️ Low volume (${s.vol_ratio}x avg) — low liquidity</span>` : "";
  return `
    <div class="scan-card ${action}-card">
      <div class="scan-market-label">🇺🇸 US Penny · ${action === "buy" ? "↑ BUY Setup" : "↓ SELL / Avoid"}</div>
      <div class="scan-action ${action}">${action === "buy" ? "↑" : "↓"} ${s.ticker}</div>
      <div class="scan-price">Price: <strong>$${s.price}</strong> &nbsp;|&nbsp; 5d: ${s.ret5 > 0 ? "+" : ""}${s.ret5}%</div>
      <div class="scan-score-row"><span>Signal Strength</span><span>${score} / ${maxScore}</span></div>
      <div class="bar-track"><div class="bar-fill ${action === "buy" ? "up" : "down"}" style="width:${pct}%"></div></div>
      <div class="scan-reason" style="margin-top:0.5rem">${scanReasons(s, action) || "<span>Multiple signals aligned</span>"}</div>
      ${volWarn}
      ${verdict(s)}
      ${s.pcr ? `<div style="font-size:0.72rem;color:var(--muted);margin-top:0.4rem">Put/Call Ratio: ${s.pcr}</div>` : ""}
      <div class="sr-section" style="margin-top:0.6rem;padding-top:0.5rem">
        <div style="font-size:0.68rem;color:var(--down);font-weight:700;margin-bottom:0.2rem">⬆ R: ${srRows(s.resistance, "resistance")}</div>
        <div style="font-size:0.68rem;color:var(--up);font-weight:700;margin-top:0.2rem">⬇ S: ${srRows(s.support, "support")}</div>
      </div>
    </div>`;
}

function toggleGuide(btn) {
  btn.classList.toggle("open");
  btn.nextElementSibling.classList.toggle("open");
}

async function runScan() {
  const btn    = document.getElementById("scan-btn");
  const sec    = document.getElementById("scan-section");
  const cards  = document.getElementById("scan-cards");
  const loader = document.getElementById("scan-loading");

  btn.disabled = true;
  btn.textContent = "Scanning…";
  sec.style.display = "block";
  loader.style.display = "block";
  cards.innerHTML = "";

  try {
    const res  = await fetch("/api/scan");
    const data = await res.json();
    loader.style.display = "none";
    renderScan(data);
  } catch(e) {
    loader.style.display = "none";
    cards.innerHTML = `<div class="error-msg">Scan failed: ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "⚡ Market Scan";
  }
}

function scanReasons(s, action) {
  const tags = [];
  const rz = s.rsi_z ?? 0;
  const bz = s.bb_z  ?? 0;
  const sz = s.sma_z ?? 0;
  const baseline = s.rsi_mean ? ` (stock avg: ${s.rsi_mean})` : "";

  if (action === "buy") {
    if (rz < -2.0)         tags.push({t:`RSI ${s.rsi} — very oversold vs own history${baseline}`, c:"bull"});
    else if (rz < -1.5)    tags.push({t:`RSI ${s.rsi} — oversold vs own history${baseline}`, c:"bull"});
    else if (rz < -1.0)    tags.push({t:`RSI ${s.rsi} — mildly low vs own history${baseline}`, c:"bull"});
    if (s.macd_hist > 0)   tags.push({t:`MACD bullish`, c:"bull"});
    if (bz < -1.5)         tags.push({t:`BB unusually low vs own history`, c:"bull"});
    else if (bz < -1.0)    tags.push({t:`Near BB lower`, c:"bull"});
    if (sz > 0.5)          tags.push({t:`SMA trend above norm`, c:"bull"});
    if (s.vol_ratio > 1.5) tags.push({t:`Vol spike ${s.vol_ratio}x`, c:"bull"});
    if (s.ret5 < -3)       tags.push({t:`-${Math.abs(s.ret5)}% dip in 5d`, c:"bull"});
    if (s.pcr && s.pcr > 1.2) tags.push({t:`PCR ${s.pcr} (fear)`, c:"bull"});
  } else {
    if (rz > 2.0)          tags.push({t:`RSI ${s.rsi} — very overbought vs own history${baseline}`, c:"bear"});
    else if (rz > 1.5)     tags.push({t:`RSI ${s.rsi} — overbought vs own history${baseline}`, c:"bear"});
    else if (rz > 1.0)     tags.push({t:`RSI ${s.rsi} — mildly high vs own history${baseline}`, c:"bear"});
    if (s.macd_hist < 0)   tags.push({t:`MACD bearish`, c:"bear"});
    if (bz > 1.5)          tags.push({t:`BB unusually high vs own history`, c:"bear"});
    else if (bz > 1.0)     tags.push({t:`Near BB upper`, c:"bear"});
    if (sz < -0.5)         tags.push({t:`SMA trend below norm`, c:"bear"});
    if (s.vol_ratio > 1.5) tags.push({t:`Vol spike ${s.vol_ratio}x`, c:"bear"});
    if (s.ret5 > 5)        tags.push({t:`+${s.ret5}% run in 5d`, c:"bear"});
    if (s.pcr && s.pcr < 0.7) tags.push({t:`PCR ${s.pcr} (greed)`, c:"bear"});
  }
  return tags.map(t => `<span class="${t.c}">${t.t}</span>`).join("");
}

function srRows(levels, type) {
  const isRes = type === "resistance";
  if (!levels || !levels.length) {
    const msg = isRes ? "No ceiling — near all-time highs" : "No floor — high downside risk";
    return `<span style="font-size:0.7rem;color:${isRes ? "var(--neutral)" : "var(--down)"}">${msg}</span>`;
  }
  const allFar = levels.every(l => l.pct_away > 10);
  const warn = allFar ? `<span style="font-size:0.68rem;color:var(--down)">⚠ All far</span> ` : "";
  const rows = levels.map(l => {
    const far = l.pct_away > 10;
    const col = far ? "var(--muted)" : (isRes ? "var(--down)" : "var(--up)");
    const sign = isRes ? "+" : "-";
    return `<span style="color:${col};font-size:0.75rem;margin-right:0.6rem"><strong>${l.price}</strong> ${sign}${l.pct_away}%${far ? " ↗far" : ""} · ${l.touches}×</span>`;
  }).join("");
  return warn + rows;
}

function scanCard(market, action, s) {
  if (!s) return `<div class="scan-card ${action}-card"><div class="scan-market-label">${market} · Best ${action.toUpperCase()}</div><div class="scan-action ${action}">${action === "buy" ? "↑ BUY" : "↓ SELL"}</div><div style="color:var(--muted);font-size:0.85rem;margin-top:0.5rem">No strong ${action} signal right now<br><span style="font-size:0.72rem">Score below threshold (4.0 / 12) — market may be ranging</span></div></div>`;
  const score = action === "buy" ? s.buy_score : s.sell_score;
  const maxScore = 12;
  const pct = Math.min(score / maxScore * 100, 100).toFixed(0);
  const arrow = action === "buy" ? "↑ BUY" : "↓ SELL";
  return `
    <div class="scan-card ${action}-card">
      <div class="scan-market-label">${market} · Best ${action.toUpperCase()}</div>
      <div class="scan-action ${action}">${arrow}</div>
      <div class="scan-ticker">${s.ticker.replace(".NS","")}</div>
      <div class="scan-price">Current Price: <strong>${s.price}</strong> &nbsp;|&nbsp; 5d: ${s.ret5 > 0 ? "+" : ""}${s.ret5}%</div>
      <div class="scan-score-row"><span>Signal Strength</span><span>${score} / ${maxScore}</span></div>
      <div class="bar-track"><div class="bar-fill ${action === "buy" ? "up" : "down"}" style="width:${pct}%"></div></div>
      <div class="scan-reason">${scanReasons(s, action) || "<span>Multiple signals aligned</span>"}</div>
      ${s.pcr ? `<div style="font-size:0.72rem;color:var(--muted);margin-top:0.5rem">Put/Call Ratio: ${s.pcr}</div>` : ""}
      ${verdict(s)}

      <div class="sr-section" style="margin-top:0.7rem;padding-top:0.6rem">
        <div style="font-size:0.68rem;color:var(--down);font-weight:700;margin-bottom:0.2rem">⬆ R: ${srRows(s.resistance, "resistance")}</div>
        <div style="font-size:0.68rem;color:var(--up);font-weight:700;margin-top:0.25rem">⬇ S: ${srRows(s.support, "support")}</div>
      </div>
    </div>`;
}

function verdict(s) {
  // Combine buy and sell scores into a single decisive verdict
  const net = (s.buy_score || 0) - (s.sell_score || 0);
  const macdBull = s.macd_hist > 0;
  const rz = s.rsi_z ?? 0;
  const bz = s.bb_z  ?? 0;

  let label, color, detail;

  if (net >= 5) {
    label = "STRONG BUY";  color = "#22c55e";
    detail = "Multiple strong bullish signals aligned";
  } else if (net >= 2.5) {
    label = "BUY";         color = "#4ade80";
    detail = "More buy signals than sell — lean long";
  } else if (net >= 1 && macdBull) {
    label = "WEAK BUY";    color = "#86efac";
    detail = "Mild buy edge with bullish momentum — watch for confirmation";
  } else if (net <= -5) {
    label = "STRONG SELL"; color = "#ef4444";
    detail = "Multiple strong bearish signals aligned";
  } else if (net <= -2.5) {
    label = "SELL";        color = "#f87171";
    detail = "More sell signals than buy — lean short";
  } else if (net <= -1 && !macdBull) {
    label = "WEAK SELL";   color = "#fca5a5";
    detail = "Mild sell edge with bearish momentum — watch for breakdown";
  } else {
    label = "HOLD / WAIT"; color = "#f59e0b";
    detail = "Signals mixed or insufficient — no clear edge right now";
  }

  return `
    <div style="margin-top:0.7rem;padding:0.45rem 0.8rem;border-radius:8px;background:${color}18;border:1px solid ${color}44;display:flex;justify-content:space-between;align-items:center;gap:0.5rem">
      <div>
        <div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em">Verdict</div>
        <div style="font-size:0.9rem;font-weight:800;color:${color}">${label}</div>
      </div>
      <div style="font-size:0.7rem;color:var(--muted);text-align:right">
        <span style="color:#22c55e">B ${s.buy_score}</span> &nbsp;
        <span style="color:#ef4444">S ${s.sell_score}</span> &nbsp;
        <span style="color:${color};font-weight:700">Net ${net >= 0 ? "+" : ""}${net.toFixed(1)}</span>
      </div>
    </div>`;
}

function fmt(val, suffix="%") {
  if (val === null || val === undefined || isNaN(val)) return "n/a";
  return (val > 0 ? "+" : "") + val + suffix;
}

function regimeBanner(region, r) {
  const icons = { bullish:"📈", bearish:"📉", selloff:"🔻", rally:"⚡", overbought:"🔥", ranging:"↔️", unknown:"❓" };
  const icon  = icons[r.regime] || "❓";
  return `
    <div class="regime-banner ${r.color}" style="grid-column:1/-1">
      <span class="regime-icon">${icon}</span>
      <div>
        <div class="regime-label">${region} Market Regime: ${r.label}</div>
        <div class="regime-desc">${r.desc}</div>
        <div class="regime-stats">
          <span>RSI ${r.rsi ?? "n/a"}</span>
          <span>5d ${fmt(r.ret5)}</span>
          <span>vs SMA20 ${fmt(r.vs_sma20)}</span>
          <span>vs SMA50 ${fmt(r.vs_sma50)}</span>
        </div>
      </div>
    </div>`;
}

function watchCard(s, action) {
  const score = action === "buy" ? s.buy_score : s.sell_score;
  return `
    <div class="scan-card ${action}-card" style="opacity:0.85">
      <div class="scan-market-label">Watch · ${action === "buy" ? "↑ BUY candidate" : "↓ SELL candidate"}</div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
        <span style="font-size:1.1rem;font-weight:800">${s.ticker.replace(".NS","")}</span>
        <span style="font-size:0.78rem;color:var(--muted)">Score ${score}/12</span>
      </div>
      <div class="scan-price">Price: <strong>${s.price}</strong> &nbsp;|&nbsp; 5d: ${s.ret5 > 0 ? "+" : ""}${s.ret5}%</div>
      <div class="scan-reason" style="margin:0.4rem 0">${scanReasons(s, action) || "<span>Watching for setup</span>"}</div>
      ${verdict(s)}
      <div class="sr-section" style="margin-top:0.5rem;padding-top:0.5rem">
        <div style="font-size:0.68rem;color:var(--down);font-weight:700;margin-bottom:0.2rem">⬆ R: ${srRows(s.resistance, "resistance")}</div>
        <div style="font-size:0.68rem;color:var(--up);font-weight:700;margin-top:0.2rem">⬇ S: ${srRows(s.support, "support")}</div>
      </div>
    </div>`;
}

function renderScan(data) {
  const cards = document.getElementById("scan-cards");

  let html = "";

  // US section
  html += regimeBanner("🇺🇸 US", data.us.regime);
  html += scanCard("🇺🇸 US", "buy",  data.us.buy);
  html += scanCard("🇺🇸 US", "sell", data.us.sell);

  // US watchlist when no strong signal
  if (!data.us.buy || !data.us.sell) {
    const watches = [...(data.us.watch_buy||[]), ...(data.us.watch_sell||[])].slice(0,3);
    if (watches.length) {
      html += `<div style="grid-column:1/-1;font-size:0.78rem;color:var(--muted);margin-top:0.3rem;margin-bottom:-0.4rem">No signal above threshold — top candidates with S&amp;R levels to watch:</div>`;
      watches.forEach(s => { html += watchCard(s, s.buy_score >= s.sell_score ? "buy" : "sell"); });
    }
  }

  // India section
  html += regimeBanner("🇮🇳 India", data.india.regime);
  html += scanCard("🇮🇳 India", "buy",  data.india.buy);
  html += scanCard("🇮🇳 India", "sell", data.india.sell);

  // India watchlist when no strong signal
  if (!data.india.buy || !data.india.sell) {
    const watches = [...(data.india.watch_buy||[]), ...(data.india.watch_sell||[])].slice(0,3);
    if (watches.length) {
      html += `<div style="grid-column:1/-1;font-size:0.78rem;color:var(--muted);margin-top:0.3rem;margin-bottom:-0.4rem">No signal above threshold — top candidates with S&amp;R levels to watch:</div>`;
      watches.forEach(s => { html += watchCard(s, s.buy_score >= s.sell_score ? "buy" : "sell"); });
    }
  }

  cards.innerHTML = html;

  const meta = document.createElement("div");
  meta.style.cssText = "font-size:0.72rem;color:var(--muted);margin-top:0.8rem;text-align:center;grid-column:1/-1";
  meta.textContent = `Scanned ${data.us.scanned} US + ${data.india.scanned} India stocks · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("scan-cards").appendChild(meta);
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