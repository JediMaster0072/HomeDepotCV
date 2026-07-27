import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HEALTH_PORT = 8080


# Pipeline order: Startup health server
# Description: Handles Kubernetes liveness and readiness HTTP requests.
class HealthCheckHandler(BaseHTTPRequestHandler):
    # Pipeline order: Health-check request
    # Description: Routes health-check paths to JSON responses or returns 404 for unknown paths.
    def do_GET(self):
        if self.path == "/health/liveness":
            self._respond({"status": "alive"})
        elif self.path == "/health/readiness":
            self._respond({"status": "ready"})
        else:
            self.send_response(404)
            self.end_headers()

    # Pipeline order: Health-check request
    # Description: Writes one JSON health-check response.
    def _respond(self, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # Pipeline order: Health-check request
    # Description: Suppresses default HTTP server request logging.
    def log_message(self, fmt, *args):  # pragma: no cover
        pass


# Pipeline order: Startup
# Description: Starts the background HTTP server used for container health checks.
def start_health_server(port=HEALTH_PORT):
    server = HTTPServer(("", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
