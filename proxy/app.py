import os
from flask import Flask, jsonify, make_response

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    return response

@app.route("/options/<symbol>", methods=["OPTIONS"])
@app.route("/health", methods=["OPTIONS"])
def preflight(symbol=None):
    return make_response("", 204)

def fetch_chain(symbol):
    try:
        from nsepython import nse_optionchain_scrapper
        return nse_optionchain_scrapper(symbol)
    except Exception:
        pass

    # Fallback: manual session
    import time, requests
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/124.0.0.0 Safari/537.36")
    hdrs = {
        "User-Agent": UA, "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br", "DNT": "1",
    }
    s = requests.Session()
    s.headers.update(hdrs)
    s.get("https://www.nseindia.com", timeout=15, headers={
        **hdrs, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    time.sleep(1.5)
    s.get("https://www.nseindia.com/option-chain", timeout=15, headers={
        **hdrs, "Referer": "https://www.nseindia.com",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    })
    time.sleep(1)
    r = s.get(
        f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
        timeout=20,
        headers={
            **hdrs,
            "Accept":           "application/json, text/plain, */*",
            "Referer":          "https://www.nseindia.com/option-chain",
            "X-Requested-With": "XMLHttpRequest",
            "sec-fetch-dest":   "empty",
            "sec-fetch-mode":   "cors",
            "sec-fetch-site":   "same-origin",
        }
    )
    if r.status_code != 200:
        raise ValueError(f"NSE HTTP {r.status_code}")
    return r.json()

@app.route("/options/<symbol>")
def options(symbol):
    symbol = symbol.upper()
    if symbol not in ("NIFTY", "BANKNIFTY"):
        return jsonify({"error": "Only NIFTY and BANKNIFTY supported"}), 400
    try:
        data = fetch_chain(symbol)
        if not data or "records" not in data:
            keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            return jsonify({"error": f"Unexpected NSE response. Keys: {keys}"}), 502
        return jsonify(data["records"])
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
