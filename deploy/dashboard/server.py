"""Dashboard server (runs on the user's laptop).

Receives heartbeats from every node (edge + cloud), keeps an in-memory
roster, and serves a single-page HTML UI that polls /api/nodes every
2 seconds.

Endpoints:
  POST /register        node says hello (first heartbeat)
  POST /heartbeat       periodic heartbeat
  GET  /api/nodes       JSON roster (used by the HTML)
  GET  /api/summary     JSON aggregated counters
  GET  /                HTML dashboard
  GET  /health          {"status": "ok"}

Run:
  cd paper2/deploy
  pip install -r dashboard/requirements.txt
  python -m dashboard.server
  # then point browser at http://localhost:9000
  # and set DASHBOARD_URL=http://<laptop-tailscale>:9000 on each node
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.protocol import Heartbeat  # noqa: E402


HEARTBEAT_TIMEOUT_S = float(os.environ.get("HEARTBEAT_TIMEOUT_S", "60"))
HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.environ.get("DASHBOARD_PORT", "9000"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [dashboard] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard")


# ---------------------------------------------------------------------------
# Roster (in-memory, thread-safe)
# ---------------------------------------------------------------------------

class Roster:
    def __init__(self):
        self._lock = threading.Lock()
        self._nodes: Dict[str, Dict] = {}        # node_id -> last heartbeat dict
        self._first_seen: Dict[str, float] = {}  # node_id -> epoch seconds
        self._last_seen: Dict[str, float] = {}   # node_id -> epoch seconds

    def upsert(self, hb: Heartbeat) -> bool:
        """Insert or update a node. Returns True iff this is a new node."""
        now = time.time()
        with self._lock:
            new = hb.node_id not in self._nodes
            self._nodes[hb.node_id] = hb.dict()
            self._last_seen[hb.node_id] = now
            if new:
                self._first_seen[hb.node_id] = now
            return new

    def snapshot(self) -> Dict[str, Dict]:
        now = time.time()
        with self._lock:
            out = {}
            for node_id, hb in self._nodes.items():
                last = self._last_seen[node_id]
                age = now - last
                out[node_id] = {
                    **hb,
                    "last_seen_age_s": age,
                    "online": age < HEARTBEAT_TIMEOUT_S,
                    "first_seen": self._first_seen.get(node_id, last),
                }
            return out


roster = Roster()


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(title="plant-deploy dashboard", version="1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.post("/register")
def register(hb: Heartbeat):
    is_new = roster.upsert(hb)
    if is_new or hb.is_register:
        log.info("REGISTER node_id=%s role=%s host=%s:%d",
                 hb.node_id, hb.role, hb.hostname, hb.port)
    return {"ok": True, "is_new": is_new}


@app.post("/heartbeat")
def heartbeat(hb: Heartbeat):
    is_new = roster.upsert(hb)
    if is_new:
        log.info("first heartbeat (no /register seen): node_id=%s role=%s",
                 hb.node_id, hb.role)
    return {"ok": True}


@app.get("/api/nodes")
def api_nodes():
    return JSONResponse(roster.snapshot())


@app.get("/api/summary")
def api_summary():
    snap = roster.snapshot()
    online_edges = [n for n in snap.values() if n["online"] and n["role"] == "edge"]
    online_clouds = [n for n in snap.values() if n["online"] and n["role"] == "cloud"]
    total_requests = sum(n["stats"]["requests_total"] for n in snap.values())
    total_routed = sum(n["stats"]["requests_routed"] for n in snap.values()
                       if n["role"] == "edge")
    total_bw = sum(n["stats"]["bandwidth_total_kb"] for n in snap.values()
                   if n["role"] == "edge")
    edge_total = sum(n["stats"]["requests_total"] for n in snap.values()
                     if n["role"] == "edge")
    return {
        "nodes_total": len(snap),
        "edges_online": len(online_edges),
        "clouds_online": len(online_clouds),
        "edges_offline": sum(1 for n in snap.values()
                             if not n["online"] and n["role"] == "edge"),
        "clouds_offline": sum(1 for n in snap.values()
                              if not n["online"] and n["role"] == "cloud"),
        "total_requests": total_requests,
        "total_routed": total_routed,
        "edge_total_requests": edge_total,
        "global_offload_rate": (total_routed / edge_total) if edge_total else 0.0,
        "total_bandwidth_kb": total_bw,
        "heartbeat_timeout_s": HEARTBEAT_TIMEOUT_S,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/style.css")
def style():
    return FileResponse(STATIC_DIR / "style.css", media_type="text/css")


@app.get("/app.js")
def appjs():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


def main() -> None:
    import uvicorn
    log.info("dashboard listening on %s:%d  (heartbeat timeout = %.0fs)",
             HOST, PORT, HEARTBEAT_TIMEOUT_S)
    uvicorn.run("dashboard.server:app", host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
