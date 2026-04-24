import os
import json
import time
import tempfile
import subprocess
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


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def curl_get(url, cookie_file, referer=None, extra_headers=None, output=None):
    cmd = [
        "curl", url,
        "-H", f"User-Agent: {UA}",
        "-H", "Accept-Language: en-US,en;q=0.9",
        "-H", "Accept-Encoding: gzip, deflate, br",
        "--compressed",
        "--silent",
        "--max-time", "20",
        "--cookie", cookie_file,
        "--cookie-jar", cookie_file,
    ]
    if referer:
        cmd += ["-H", f"Referer: {referer}"]
    if extra_headers:
        for h in extra_headers:
            cmd += ["-H", h]
    if output:
        cmd += ["--output", output]
        result = subprocess.run(cmd, check=True)
        return None
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout


def fetch_chain(symbol):
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        cookie_file = f.name

    try:
        # Step 1: homepage — NSE sets initial session cookies
        curl_get(
            "https://www.nseindia.com/",
            cookie_file,
            extra_headers=["Accept: text/html,application/xhtml+xml,*/*;q=0.8"],
            output=os.devnull,
        )
        time.sleep(1.5)

        # Step 2: option-chain page — Akamai challenge runs, bm_sv cookie set
        curl_get(
            "https://www.nseindia.com/option-chain",
            cookie_file,
            referer="https://www.nseindia.com/",
            extra_headers=["Accept: text/html,application/xhtml+xml,*/*;q=0.8"],
            output=os.devnull,
        )
        time.sleep(1)

        # Step 3: API call with all cookies in place
        body = curl_get(
            f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
            cookie_file,
            referer="https://www.nseindia.com/option-chain",
            extra_headers=[
                "Accept: application/json, text/plain, */*",
                "X-Requested-With: XMLHttpRequest",
            ],
        )

        return json.loads(body)

    finally:
        try:
            os.unlink(cookie_file)
        except Exception:
            pass


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
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"curl failed: {e}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
