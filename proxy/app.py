import os
import time
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
    # curl_cffi impersonates Chrome TLS fingerprint — bypasses Akamai bot detection
    from curl_cffi import requests as cf
    s = cf.Session(impersonate="chrome124")

    s.get("https://www.nseindia.com", timeout=15)
    time.sleep(1.5)
    s.get("https://www.nseindia.com/option-chain", timeout=15, headers={
        "Referer": "https://www.nseindia.com",
    })
    time.sleep(1)

    r = s.get(
        f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
        timeout=20,
        headers={
            "Accept":           "application/json, text/plain, */*",
            "Referer":          "https://www.nseindia.com/option-chain",
            "X-Requested-With": "XMLHttpRequest",
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
