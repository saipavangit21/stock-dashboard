import os
import time
from flask import Flask, jsonify, make_response
import requests

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

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

BASE_HEADERS = {
    "User-Agent":      UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "DNT":             "1",
}

def nse_session():
    s = requests.Session()
    s.headers.update(BASE_HEADERS)

    # Step 1: hit homepage to get initial cookies
    s.get("https://www.nseindia.com", timeout=15, headers={
        **BASE_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Upgrade-Insecure-Requests": "1",
    })
    time.sleep(1)

    # Step 2: hit the option-chain page to get additional cookies NSE sets on that page
    s.get("https://www.nseindia.com/option-chain", timeout=15, headers={
        **BASE_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": "https://www.nseindia.com",
        "Upgrade-Insecure-Requests": "1",
    })
    time.sleep(1)

    return s

def fetch_chain(symbol):
    s = nse_session()
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    r = s.get(url, timeout=20, headers={
        **BASE_HEADERS,
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/option-chain",
        "X-Requested-With": "XMLHttpRequest",
        "sec-fetch-dest":   "empty",
        "sec-fetch-mode":   "cors",
        "sec-fetch-site":   "same-origin",
    })
    if r.status_code != 200:
        raise ValueError(f"NSE returned HTTP {r.status_code}")
    return r.json()

@app.route("/options/<symbol>")
def options(symbol):
    symbol = symbol.upper()
    if symbol not in ("NIFTY", "BANKNIFTY"):
        return jsonify({"error": "Only NIFTY and BANKNIFTY supported"}), 400
    try:
        data = fetch_chain(symbol)
        if not data or "records" not in data:
            return jsonify({"error": "NSE returned no data", "raw": str(data)[:300]}), 502
        return jsonify(data["records"])
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/health")
def health():
    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
