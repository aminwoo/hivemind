"""
Vercel API route to check engine/endpoint health.
"""

import os
import json
import requests
from http.server import BaseHTTPRequestHandler

RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID')
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Check RunPod endpoint health."""
        try:
            response = requests.get(
                f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/health",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self._send_json({
                    "status": "ok",
                    "workers": {
                        "ready": data.get("workers", {}).get("ready", 0),
                        "running": data.get("workers", {}).get("running", 0),
                        "idle": data.get("workers", {}).get("idle", 0)
                    }
                })
            else:
                self._send_json({
                    "status": "error",
                    "message": "Endpoint not available"
                })
                
        except Exception as e:
            self._send_json({
                "status": "error", 
                "message": str(e)
            })
            
    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
