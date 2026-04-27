import os
import json
import requests as req_lib
from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/store/<symbol>", methods=["OPTIONS"])
@app.route("/options/<symbol>", methods=["OPTIONS"])
@app.route("/health", methods=["OPTIONS"])
def preflight(symbol=None):
    return make_response("", 204)

_cache = {}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


@app.route("/fetch")
def fetch_with_cookies():
    """
    User pastes this in NSE console (bypasses CSP via window.open, not fetch):
      window.open('http://localhost:8000/fetch?c='+encodeURIComponent(document.cookie));
    This endpoint receives the real browser cookies and uses them server-side.
    """
    cookies_str = request.args.get("c", "")
    cookies = {}
    for pair in cookies_str.split(";"):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            cookies[k.strip()] = v.strip()

    results = {}
    for symbol in ["NIFTY", "BANKNIFTY"]:
        try:
            s = req_lib.Session()
            s.headers.update({
                "User-Agent":        UA,
                "Accept":            "application/json, text/plain, */*",
                "Accept-Language":   "en-US,en;q=0.9",
                "Accept-Encoding":   "gzip, deflate",
                "Referer":           "https://www.nseindia.com/option-chain",
                "X-Requested-With":  "XMLHttpRequest",
            })
            s.cookies.update(cookies)
            r = s.get(
                f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
                timeout=15,
            )
            data = r.json()
            if "records" in data:
                _cache[symbol] = data["records"]
                results[symbol] = "✓ OK"
            else:
                results[symbol] = f"✗ No records — keys: {list(data.keys())}"
        except Exception as e:
            results[symbol] = f"✗ Error: {e}"

    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in results.items())
    ok = all("✓" in v for v in results.values())
    return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem;">
    <h2>{"✓ NSE data updated!" if ok else "⚠ Partial result"}</h2>
    <table border="1" cellpadding="8">{rows}</table>
    <p style="color:#888;margin-top:1rem;">You can close this tab and use the dashboard.</p>
    </body></html>"""


@app.route("/paste-form", methods=["POST"])
def paste_form():
    """Receive NSE data via HTML form POST (not blocked by CSP unlike fetch)."""
    try:
        data = json.loads(request.form.get("data", "{}"))
        loaded = []
        for symbol, records in data.items():
            _cache[symbol.upper()] = records
            loaded.append(symbol.upper())
        return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem;">
        <h2>&#10003; Loaded: {', '.join(loaded)}</h2>
        <p>Close this tab and click <b>India Options</b> on the dashboard.</p>
        </body></html>"""
    except Exception as e:
        return f"Error: {e}", 400


@app.route("/paste")
def paste():
    """Read NSE JSON from clipboard (user copies via browser console on NSE page)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        data = json.loads(text)
        loaded = []
        for symbol, records in data.items():
            _cache[symbol.upper()] = records
            loaded.append(symbol.upper())
        return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem;">
        <h2>&#10003; Loaded: {', '.join(loaded)}</h2>
        <p>You can close this tab and click <b>India Options</b> on the dashboard.</p>
        </body></html>"""
    except Exception as e:
        return f"""<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem;">
        <h2>&#9888; Error: {e}</h2>
        <p>Make sure you ran the console script on NSE first and allowed clipboard access.</p>
        </body></html>""", 400


@app.route("/store/<symbol>", methods=["POST"])
def store(symbol):
    symbol = symbol.upper()
    data = request.get_json(force=True)
    if data:
        _cache[symbol] = data
        return jsonify({"ok": True})
    return jsonify({"error": "No data"}), 400


@app.route("/options/<symbol>")
def options(symbol):
    symbol = symbol.upper()
    if symbol not in ("NIFTY", "BANKNIFTY"):
        return jsonify({"error": "Only NIFTY and BANKNIFTY supported"}), 400
    if symbol in _cache:
        return jsonify(_cache[symbol])
    return jsonify({"error": f"No data yet. Open NSE option-chain and run the console command."}), 502


@app.route("/health")
def health():
    return jsonify({"status": "ok", "cached": list(_cache.keys())})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n NSE Proxy on http://localhost:{port}")
    print("\n On NSE option-chain page, open console (F12) and paste:")
    print(f"  window.open('http://localhost:{port}/fetch?c='+encodeURIComponent(document.cookie));")
    app.run(host="0.0.0.0", port=port)
