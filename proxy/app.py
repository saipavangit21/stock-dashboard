import os
import json
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

# In-memory cache — bookmarklet POSTs here, dashboard GETs from here
_cache = {}

@app.route("/store/<symbol>", methods=["POST"])
def store(symbol):
    """Bookmarklet POSTs NSE data here from inside the user's real browser."""
    symbol = symbol.upper()
    data = request.get_json(force=True)
    if data:
        _cache[symbol] = data
        return jsonify({"ok": True, "symbol": symbol})
    return jsonify({"error": "No data received"}), 400

@app.route("/options/<symbol>")
def options(symbol):
    symbol = symbol.upper()
    if symbol not in ("NIFTY", "BANKNIFTY"):
        return jsonify({"error": "Only NIFTY and BANKNIFTY supported"}), 400
    if symbol in _cache:
        return jsonify(_cache[symbol])
    return jsonify({
        "error": f"No data yet for {symbol}. Open NSE option-chain in your browser and click the bookmarklet."
    }), 502

@app.route("/health")
def health():
    cached = list(_cache.keys())
    return jsonify({"status": "ok", "cached": cached})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n NSE Proxy running on http://localhost:{port}")
    print("\n Bookmarklet — drag this to your bookmarks bar:")
    print("""
  javascript:(async()=>{
    const base='http://localhost:8000';
    for(const s of['NIFTY','BANKNIFTY']){
      try{
        const d=await fetch('/api/option-chain-indices?symbol='+s).then(r=>r.json());
        await fetch(base+'/store/'+s,{method:'POST',body:JSON.stringify(d['records']||d),headers:{'Content-Type':'application/json'}});
      }catch(e){alert('Error for '+s+': '+e);}
    }
    alert('NSE options data updated!');
  })();
    """)
    print(f"\n Steps:")
    print(f"  1. Open https://www.nseindia.com/option-chain in your browser")
    print(f"  2. Click the bookmarklet")
    print(f"  3. Click India Options on the dashboard\n")
    app.run(host="0.0.0.0", port=port)
