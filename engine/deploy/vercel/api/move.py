"""
Vercel API route to proxy requests to RunPod serverless endpoint.
This keeps the RunPod API key secure on the server side.
"""

import os
import json
import requests
from http.server import BaseHTTPRequestHandler

RUNPOD_ENDPOINT_ID = os.environ.get('RUNPOD_ENDPOINT_ID')
RUNPOD_API_KEY = os.environ.get('RUNPOD_API_KEY')
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def do_POST(self):
        """Forward move request to RunPod."""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
            
            # Validate required fields
            if 'fen' not in data:
                self._send_error(400, "Missing 'fen' parameter")
                return
                
            # Build RunPod request
            runpod_payload = {
                "input": {
                    "action": data.get("action", "move"),
                    "fen": data["fen"],
                    "nodes": data.get("nodes", 800),
                    "movetime": data.get("movetime", 0)
                }
            }
            
            if "moves" in data:
                runpod_payload["input"]["moves"] = data["moves"]
                
            # Call RunPod (synchronous)
            response = requests.post(
                f"{RUNPOD_BASE_URL}/runsync",
                headers={
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=runpod_payload,
                timeout=60
            )
            
            if response.status_code != 200:
                self._send_error(response.status_code, f"RunPod error: {response.text}")
                return
                
            result = response.json()
            
            # Extract output from RunPod response
            if result.get("status") == "COMPLETED":
                output = result.get("output", {})
                self._send_json(output)
            elif result.get("status") == "FAILED":
                self._send_error(500, result.get("error", "Unknown error"))
            else:
                self._send_error(500, f"Unexpected status: {result.get('status')}")
                
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
        except requests.Timeout:
            self._send_error(504, "Request timeout")
        except Exception as e:
            self._send_error(500, str(e))
            
    def _send_json(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
        
    def _send_error(self, code, message):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
