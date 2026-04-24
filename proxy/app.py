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
    """
    Use patchright (patched Playwright) so Chrome TLS fingerprint matches a
    real browser — Akamai cannot distinguish it from a genuine Chrome session.
    """
    from patchright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--single-process"],
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()

        # Step 1: homepage — picks up initial Akamai cookies
        page.goto("https://www.nseindia.com", wait_until="domcontentloaded",
                  timeout=30000)
        page.wait_for_timeout(2000)

        # Step 2: option-chain page — Akamai JS runs, bm_sv cookie is set
        page.goto("https://www.nseindia.com/option-chain",
                  wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)

        # Step 3: API call from inside the browser (all cookies already valid)
        result = page.evaluate(f"""
            async () => {{
                const r = await fetch(
                    '/api/option-chain-indices?symbol={symbol}',
                    {{
                        headers: {{
                            'Accept': 'application/json, text/plain, */*',
                            'X-Requested-With': 'XMLHttpRequest',
                            'Referer': 'https://www.nseindia.com/option-chain'
                        }}
                    }}
                );
                return await r.json();
            }}
        """)

        browser.close()
        return result


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
