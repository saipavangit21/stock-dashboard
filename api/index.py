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
_cache      = {}
_cache_date = {}
_lock       = threading.Lock()

# ── Zerodha Kite session (persists across warm invocations) ────────────────────
_kite_access_token = None

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
        # Two modes: oversold dip OR momentum breakout
        buy = 0.0

        # Dip/oversold signals
        if rsi_z < -2.0:   buy += 3.0
        elif rsi_z < -1.5: buy += 1.5
        elif rsi_z < -1.0: buy += 0.5
        if bb_z < -1.5:        buy += 2.0
        elif bb_z < -1.0:      buy += 1.0
        if ret5 < -0.05:       buy += 1.5
        if pcr is not None:
            if pcr > 1.3:      buy += 1.5
            elif pcr > 1.0:    buy += 0.5

        # Momentum/breakout signals (work in bullish markets)
        if macd_hist > 0:      buy += 1.5
        if sma_z > 0.5:        buy += 1.0   # short-term trend above its own norm
        if vol_ratio > 2.0:    buy += 1.5   # strong volume confirms breakout
        elif vol_ratio > 1.5:  buy += 0.75
        if rsi_z > 0.5 and macd_hist > 0 and sma_z > 0.3:
            buy += 1.5   # momentum confluence: RSI rising + MACD bull + SMA bull
        if ret5 > 0.03 and vol_ratio > 1.5:
            buy += 1.0   # rising on above-avg volume = healthy momentum

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
        result = []
        for dt, row in raw.iterrows():
            label = dt.strftime("%H:%M") if interval == "1h" else str(dt.date())
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

        # ── Scoring (same logic as scan_ticker) ─────────────────────
        buy = sell = 0.0

        # RSI z-score
        if   rsi_z < -2.0: buy  += 3.0
        elif rsi_z < -1.5: buy  += 1.5
        elif rsi_z < -1.0: buy  += 0.5
        if   rsi_z >  2.0: sell += 3.0
        elif rsi_z >  1.5: sell += 1.5
        elif rsi_z >  1.0: sell += 0.5

        # MACD
        if mh > 0: buy  += 1.5
        else:      sell += 1.5

        # BB z-score
        if   bb_z < -1.5: buy  += 2.0
        elif bb_z < -1.0: buy  += 1.0
        if   bb_z >  1.5: sell += 2.0
        elif bb_z >  1.0: sell += 1.0

        # SMA trend
        if   sma_z >  0.5: buy  += 1.0
        elif sma_z < -0.5: sell += 1.0

        # Volume
        if vol_r > 2.0:
            if mh > 0: buy  += 1.5
            else:      sell += 1.0
        elif vol_r > 1.5:
            if mh > 0: buy  += 0.75

        # Recent return
        if ret5 < -5:   buy  += 1.5
        if ret5 >  8:   sell += 1.5

        # Momentum confluence
        if rsi_z > 0.5 and mh > 0 and sma_z > 0.3:
            buy += 1.5
        if ret5 > 3 and vol_r > 1.5:
            buy += 1.0

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
            entry     = f"${round(price, 2)} (current) or on dip to ${nearest_sup}" if nearest_sup else f"${price}"
            target    = f"${nearest_res}" if nearest_res else f"${round(price * 1.08, 2)} (+8%)"
            stop_loss = f"${round(nearest_sup * 0.985, 2)}" if nearest_sup else f"${round(price * 0.95, 2)} (-5%)"
        elif verdict == "SELL":
            entry     = f"${price} (exit now) or at bounce to ${nearest_res}" if nearest_res else f"${price}"
            target    = f"${nearest_sup}" if nearest_sup else f"${round(price * 0.92, 2)} (-8%)"
            stop_loss = f"${round(nearest_res * 1.015, 2)}" if nearest_res else f"${round(price * 1.05, 2)} (+5%)"
        else:
            entry     = f"Wait — watch ${nearest_sup} support / ${nearest_res} resistance" if nearest_sup and nearest_res else f"${price}"
            target    = f"${nearest_res}" if nearest_res else "—"
            stop_loss = f"${nearest_sup}" if nearest_sup else "—"

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
            "ticker":  ticker,
            "price":   price,
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


@app.route("/api/kite-login")
def kite_login():
    """Redirect user to Zerodha OAuth login page."""
    import os
    from kiteconnect import KiteConnect
    api_key = os.environ.get("KITE_API_KEY", "")
    if not api_key:
        return "KITE_API_KEY not set in Vercel env vars.", 503
    kite = KiteConnect(api_key=api_key)
    return __import__("flask").redirect(kite.login_url())


@app.route("/api/kite-callback")
def kite_callback():
    """Exchange Zerodha request_token for access_token and store in memory."""
    import os
    from kiteconnect import KiteConnect
    global _kite_access_token

    api_key    = os.environ.get("KITE_API_KEY",    "")
    api_secret = os.environ.get("KITE_API_SECRET", "")
    req_token  = request.args.get("request_token", "")
    status     = request.args.get("status", "")

    if status != "success" or not req_token:
        return "Login failed or cancelled.", 400

    try:
        kite = KiteConnect(api_key=api_key)
        data = kite.generate_session(req_token, api_secret=api_secret)
        _kite_access_token = data["access_token"]
        return __import__("flask").redirect("/?options=1")
    except Exception as e:
        return f"Token exchange failed: {e}", 500


@app.route("/api/options")
def api_options():
    """
    NIFTY / BankNifty options chain via Zerodha Kite Connect.
    Access token is obtained via /api/kite-login OAuth flow and stored in memory.
    Env vars needed: KITE_API_KEY, KITE_API_SECRET (permanent, set once in Vercel).
    """
    import os
    from datetime import date as _date
    from kiteconnect import KiteConnect

    global _kite_access_token

    api_key = os.environ.get("KITE_API_KEY", "")
    if not api_key:
        return jsonify({"needs_auth": True, "login_url": "/api/kite-login",
                        "error": "KITE_API_KEY not set in Vercel env vars"})

    if not _kite_access_token:
        return jsonify({"needs_auth": True, "login_url": "/api/kite-login",
                        "error": "Not authenticated — click Connect Zerodha"})

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(_kite_access_token)

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
        elif pcr >= 1.5:   signal, pcr_text = "BUY",  f"{pcr} — Strong support (high PE OI)"
        elif pcr >= 1.1:   signal, pcr_text = "BUY",  f"{pcr} — Moderate support"
        elif pcr <= 0.6:   signal, pcr_text = "SELL", f"{pcr} — Strong resistance (high CE OI)"
        elif pcr <= 0.9:   signal, pcr_text = "SELL", f"{pcr} — Moderate resistance"
        else:              signal, pcr_text = "HOLD", f"{pcr} — Neutral"

        return {
            "expiry": expiry_str, "signal": signal, "pcr": pcr, "pcr_text": pcr_text,
            "total_call_oi": total_ce, "total_put_oi": total_pe, "max_pain": mp_strike,
            "ce_resistance": [{"strike":k,"oi":v["oi"],"vol":v["vol"],"ltp":v["ltp"]} for k,v in top_ce],
            "pe_support":    [{"strike":k,"oi":v["oi"],"vol":v["vol"],"ltp":v["ltp"]} for k,v in top_pe],
        }

    def analyse_index(nse_symbol, kite_symbol, display_name):
        """
        nse_symbol  = 'NIFTY' or 'BANKNIFTY'
        kite_symbol = 'NSE:NIFTY 50' or 'NSE:NIFTY BANK'
        """
        try:
            # Spot price
            q = kite.quote([kite_symbol])
            spot = round(float(q[kite_symbol]["last_price"]), 2)

            # NFO instruments — fetch once, filter locally
            instruments = kite.instruments("NFO")
            today = _date.today()

            # Keep only options for this underlying
            opts = [i for i in instruments
                    if i["name"] == nse_symbol
                    and i["instrument_type"] in ("CE", "PE")
                    and i["expiry"] >= today]

            if not opts:
                return {"error": f"No {nse_symbol} options found in NFO instruments", "spot": spot}

            expiries = sorted(set(o["expiry"] for o in opts))
            weekly_exp  = expiries[0]
            monthly_exp = next((e for e in expiries if e.month == today.month), weekly_exp)

            def fetch_expiry(exp_date):
                exp_opts = [o for o in opts if o["expiry"] == exp_date
                            and spot * 0.85 <= o["strike"] <= spot * 1.15]
                if not exp_opts:
                    exp_opts = [o for o in opts if o["expiry"] == exp_date][:80]

                # Batch quote (Kite allows ~500 per call; use NFO:SYMBOL format)
                inst_keys = [f"NFO:{o['tradingsymbol']}" for o in exp_opts]
                quotes = {}
                for i in range(0, len(inst_keys), 500):
                    quotes.update(kite.quote(inst_keys[i:i+500]))

                calls, puts = {}, {}
                for o in exp_opts:
                    key = f"NFO:{o['tradingsymbol']}"
                    q   = quotes.get(key, {})
                    oi  = int(q.get("oi", 0) or 0)
                    vol = int(q.get("volume", 0) or 0)
                    ltp = float(q.get("last_price", 0) or 0)
                    td  = {"oi": oi, "vol": vol, "ltp": ltp}
                    if o["instrument_type"] == "CE":
                        calls[int(o["strike"])] = td
                    else:
                        puts[int(o["strike"])]  = td

                return compute_chain(calls, puts, str(exp_date))

            out = {"spot": spot, "name": display_name}
            try:    out["weekly"]  = fetch_expiry(weekly_exp)
            except Exception as e: out["weekly"] = {"error": str(e)}

            if monthly_exp != weekly_exp:
                try:    out["monthly"] = fetch_expiry(monthly_exp)
                except Exception as e: out["monthly"] = {"error": str(e)}
            else:
                out["monthly"] = out["weekly"]

            return out
        except Exception as e:
            # Token may have expired — clear so UI shows login button again
            if "token" in str(e).lower() or "403" in str(e):
                global _kite_access_token
                _kite_access_token = None
            return {"error": str(e)}

    return jsonify(sanitize({
        "NIFTY":     analyse_index("NIFTY",     "NSE:NIFTY 50",   "Nifty 50"),
        "BANKNIFTY": analyse_index("BANKNIFTY", "NSE:NIFTY BANK", "Bank Nifty"),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }))


@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE, stocks=STOCKS)


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
    <button class="hdr-btn" onclick="showScan('penny')">Penny Scan</button>
    <button class="hdr-btn" onclick="showOptions()">India Options</button>
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
    Technology:  [{t:"AAPL",n:"Apple"},{t:"MSFT",n:"Microsoft"},{t:"NVDA",n:"NVIDIA"},{t:"AMD",n:"AMD"},{t:"META",n:"Meta"},{t:"GOOGL",n:"Alphabet"},{t:"AMZN",n:"Amazon"},{t:"TSLA",n:"Tesla"}],
    Finance:     [{t:"JPM",n:"JP Morgan"},{t:"BAC",n:"Bank of America"},{t:"GS",n:"Goldman Sachs"},{t:"MS",n:"Morgan Stanley"},{t:"V",n:"Visa"},{t:"MA",n:"Mastercard"}],
    Healthcare:  [{t:"UNH",n:"UnitedHealth"},{t:"PFE",n:"Pfizer"},{t:"JNJ",n:"J&J"}],
    Energy:      [{t:"XOM",n:"ExxonMobil"},{t:"CVX",n:"Chevron"}],
    ETFs:        [{t:"SPY",n:"S&P 500 ETF"},{t:"QQQ",n:"Nasdaq ETF"},{t:"ARKK",n:"ARK Innovation"}],
    Speculative: [{t:"BDMD",n:"Baird Medical"}],
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
      html += `<div class="stock-item${sel}" onclick="selectStock('${s.t}','${s.n.replace(/'/g,"\\\\'")}')" >
        <div>
          <div style="font-weight:700;">${s.t.replace(".NS","")}</div>
          <div class="stk-name">${s.n}</div>
        </div>
      </div>`;
    }
  }
  list.innerHTML = html || `<div style="padding:.8rem;color:var(--muted);font-size:.82rem;">No results</div>`;
}

function filterStocks(val) { renderSidebar(currentTab, val); }

function selectStock(ticker, name) {
  currentTicker = ticker;
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

    document.getElementById("hdr-price").textContent = `$${last.close.toLocaleString()}`;
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
        plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`$${c.raw.toLocaleString()}`}}},
        scales:{
          x:{grid:{color:"rgba(255,255,255,.05)"},ticks:{color:"#8892a4",maxTicksLimit:8,font:{size:10}}},
          y:{grid:{color:"rgba(255,255,255,.05)"},ticks:{color:"#8892a4",font:{size:10},callback:v=>`$${v.toLocaleString()}`},position:"right"},
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
          <span class="sr-price">$${r.price.toLocaleString()}</span>
          <span class="sr-pct">+${r.pct_away}% away</span>
          <span class="sr-touches">${r.touches}x</span>
        </div>`;
      });
    }
    if (data.support && data.support.length) {
      srHtml += `<div class="sr-heading" style="margin-top:.5rem;">Support</div>`;
      data.support.forEach(s => {
        srHtml += `<div class="sr-level sup">
          <span class="sr-price">$${s.price.toLocaleString()}</span>
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
  document.getElementById("scan-overlay").style.display = "block";
  document.getElementById("scan-title").textContent     = type === "penny" ? "Penny Stock Scan" : "Market Scan";
  document.getElementById("scan-loading").style.display = "block";
  document.getElementById("scan-cards").innerHTML       = "";
  document.getElementById("regime-us-banner").innerHTML    = "";
  document.getElementById("regime-india-banner").innerHTML = "";
  fetch(type === "penny" ? "/api/penny" : "/api/scan")
    .then(r => r.json())
    .then(data => {
      document.getElementById("scan-loading").style.display = "none";
      if (type === "penny") renderPenny(data);
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
    srHtml += `<div class="sc-sr-line res">R: $${r.resistance[0].price} (+${r.resistance[0].pct_away}%)</div>`;
  if (r.support && r.support[0])
    srHtml += `<div class="sc-sr-line sup">S: $${r.support[0].price} (-${r.support[0].pct_away}%)</div>`;
  return `<div class="scan-card">
    <div class="sc-top">
      <span class="sc-ticker">${r.ticker.replace(".NS","")}</span>
      <span class="sc-badge ${v}">${v}</span>
    </div>
    <div class="sc-price">$${r.price.toLocaleString()}</div>
    <div class="sc-row">Buy <span>${r.buy_score}</span> &nbsp; Sell <span>${r.sell_score}</span></div>
    <div class="sc-row">RSI <span>${r.rsi}</span> (z <span>${r.rsi_z>0?"+":""}${r.rsi_z}</span>)</div>
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

function renderPenny(data) {
  let html = "";
  (data.buys ||[]).forEach(r=>{ html += `<div><div style="font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">Penny Buy</div>${scanCard(r,"BUY")}</div>`; });
  (data.sells||[]).forEach(r=>{ html += `<div><div style="font-size:.7rem;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;">Penny Sell</div>${scanCard(r,"SELL")}</div>`; });
  document.getElementById("scan-cards").innerHTML = html || `<div style="color:var(--muted);">No penny signals found</div>`;
  const meta = document.createElement("div");
  meta.style.cssText = "font-size:.72rem;color:var(--muted);margin-top:.8rem;grid-column:1/-1;text-align:center;";
  meta.textContent   = `Scanned ${data.scanned||0} penny stocks · ${new Date(data.generated_at).toLocaleTimeString()}`;
  document.getElementById("scan-cards").appendChild(meta);
}

// ── India Options ─────────────────────────────────────────────
function showOptions() {
  document.getElementById("options-overlay").style.display = "block";
  document.getElementById("options-loading").style.display = "block";
  document.getElementById("options-content").innerHTML     = "";
  fetch("/api/options")
    .then(r => r.json())
    .then(data => {
      document.getElementById("options-loading").style.display = "none";
      if (data.needs_auth) {
        document.getElementById("options-content").innerHTML =
          `<div style="grid-column:1/-1;text-align:center;padding:2rem;">
            <div style="font-size:1rem;color:var(--muted);margin-bottom:1.2rem;">${data.error || "Connect your Zerodha account to view live options data."}</div>
            <a href="/api/kite-login" style="display:inline-block;background:var(--accent);color:#fff;padding:.7rem 1.8rem;border-radius:10px;font-weight:700;font-size:.95rem;text-decoration:none;">
              Connect Zerodha (Kite)
            </a>
          </div>`;
        return;
      }
      let html = "";
      for (const key of ["NIFTY","BANKNIFTY"]) {
        const d = data[key];
        if (!d || d.error) {
          html += `<div class="card"><div class="card-title">${key}</div><span style="color:var(--muted)">${d?.error||"No data"}</span></div>`;
          continue;
        }
        html += `<div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;">
            <div class="card-title" style="margin:0;">${d.name||key}</div>
            <div style="font-size:1rem;font-weight:800;">&#8377;${(d.spot||0).toLocaleString()}</div>
          </div>`;
        for (const [label, exp] of [["Weekly (Day)", d.weekly], ["Monthly", d.monthly]]) {
          if (!exp) continue;
          if (exp.error) { html += `<div style="color:var(--muted);font-size:.8rem;margin-bottom:.8rem;">${label}: ${exp.error}</div>`; continue; }
          const sc = exp.signal === "BUY" ? "var(--up)" : exp.signal === "SELL" ? "var(--down)" : "var(--neutral)";
          const bg = exp.signal === "BUY" ? "rgba(34,197,94,.12)" : exp.signal === "SELL" ? "rgba(239,68,68,.12)" : "rgba(245,158,11,.12)";
          html += `<div style="background:${bg};border-radius:10px;padding:.7rem .9rem;margin-bottom:.8rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem;">
              <span style="font-size:.75rem;font-weight:700;text-transform:uppercase;color:var(--muted);">${label} · ${exp.expiry}</span>
              <span style="font-weight:800;color:${sc};font-size:.95rem;">${exp.signal}</span>
            </div>
            <div style="font-size:.8rem;display:grid;grid-template-columns:1fr 1fr;gap:.3rem .8rem;margin-bottom:.5rem;">
              <span style="color:var(--muted);">PCR</span><span style="font-weight:700;">${exp.pcr_text||exp.pcr||"—"}</span>
              <span style="color:var(--muted);">Max Pain</span><span style="font-weight:700;">${exp.max_pain ? "&#8377;"+exp.max_pain.toLocaleString() : "—"}</span>
              <span style="color:var(--muted);">Total CE OI</span><span>${(exp.total_call_oi||0).toLocaleString()}</span>
              <span style="color:var(--muted);">Total PE OI</span><span>${(exp.total_put_oi||0).toLocaleString()}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem;">
              <div>
                <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;color:var(--down);margin-bottom:.25rem;">CE Resistance (Calls)</div>
                ${(exp.ce_resistance||[]).map(r=>`<div style="display:flex;justify-content:space-between;background:rgba(239,68,68,.08);border-radius:5px;padding:.2rem .4rem;margin-bottom:.2rem;font-size:.78rem;">
                  <span style="font-weight:700;">&#8377;${r.strike.toLocaleString()}</span>
                  <span style="color:var(--muted);">OI ${(r.oi/1e5).toFixed(1)}L</span>
                </div>`).join("")}
              </div>
              <div>
                <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;color:var(--up);margin-bottom:.25rem;">PE Support (Puts)</div>
                ${(exp.pe_support||[]).map(r=>`<div style="display:flex;justify-content:space-between;background:rgba(34,197,94,.08);border-radius:5px;padding:.2rem .4rem;margin-bottom:.2rem;font-size:.78rem;">
                  <span style="font-weight:700;">&#8377;${r.strike.toLocaleString()}</span>
                  <span style="color:var(--muted);">OI ${(r.oi/1e5).toFixed(1)}L</span>
                </div>`).join("")}
              </div>
            </div>
          </div>`;
        }
        html += `</div>`;
      }
      document.getElementById("options-content").innerHTML = html;
      const meta = document.createElement("div");
      meta.style.cssText = "grid-column:1/-1;font-size:.72rem;color:var(--muted);text-align:center;margin-top:.4rem;";
      meta.textContent = `Updated ${new Date(data.generated_at).toLocaleTimeString()}`;
      document.getElementById("options-content").appendChild(meta);
    })
    .catch(e => {
      document.getElementById("options-loading").innerHTML =
        `<span style="color:var(--down)">Failed: ${e.message}</span>`;
    });
}
document.getElementById("options-overlay").addEventListener("click", function(e){ if(e.target===this) this.style.display="none"; });

// Auto-open options overlay after Zerodha OAuth redirect
if (new URLSearchParams(window.location.search).get("options") === "1") {
  history.replaceState({}, "", "/");
  showOptions();
}

// Init
renderSidebar("us","");
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True, port=5000)
