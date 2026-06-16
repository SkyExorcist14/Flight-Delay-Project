"""
Flask entry point for the DATA 603 Flight Intelligence web app.
Run locally with: python app.py
Then open: http://127.0.0.1:5000
"""
from pathlib import Path
import os
from flask import Flask, Response, send_file, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

@app.route("/")
def index():
    return send_file(BASE_DIR / "webapp.html")

@app.route("/api/results")
def api_results():
    results_path = BASE_DIR / "results.json"
    if not results_path.exists():
        return Response('{"error":"results.json not found"}', status=404, mimetype="application/json")
    return Response(results_path.read_text(encoding="utf-8"), mimetype="application/json")

@app.route("/results.json")
def results_json():
    return send_from_directory(BASE_DIR, "results.json", mimetype="application/json")

@app.route("/processed/<path:filename>")
def processed_file(filename):
    return send_from_directory(BASE_DIR / "Processed", filename)

@app.route("/health")
def health():
    return {"status": "ok", "app": "SkySense Flight Intelligence"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
