"""Operations console for the edge--cloud deployment.

Serves a single-page UI that follows a fleet run as it happens: it re-reads
the run's events.csv / meta.json (written by client/fleet_sim.py) and polls
every edge node and the cloud node for health.

  python experiments/console/server.py --run-dir results/fleet_run \
      --image-dir <plantwild_test> --edge-host raspberrypi.local \
      --cloud-url http://jetson.local:8001 --port 9100
  -> http://localhost:9100
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from collections import defaultdict
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="plant-console")
CFG: dict = {}
_nodes_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def _read_run() -> tuple[dict, list[dict]]:
    run = Path(CFG["run_dir"])
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    rows: list[dict] = []
    if (run / "events.csv").exists():
        with (run / "events.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return meta, rows


@app.get("/api/state")
def state():
    meta, rows = _read_run()
    farms_meta = meta.get("farms", {})
    farms = {f: {"type": m.get("type"), "crops": m.get("crops"), "planned": m.get("n_images", 0),
                 "n": 0, "esc": 0, "cloud_ok": 0, "correct": 0, "edge_correct": 0,
                 "up_kb": 0.0, "cloud_only_kb": 0.0, "hourly": defaultdict(int)}
             for f, m in sorted(farms_meta.items())}
    hourly = defaultdict(lambda: defaultdict(int))
    lat_local, lat_esc, edge_ms, cloud_ms, net_ms = [], [], [], [], []
    for r in rows:
        f = farms.setdefault(r["farm"], {"type": "?", "crops": [], "planned": 0, "n": 0, "esc": 0,
                                         "cloud_ok": 0, "correct": 0, "edge_correct": 0, "up_kb": 0.0,
                                         "cloud_only_kb": 0.0, "hourly": defaultdict(int)})
        h = int(r["sim_day"]) * 24 + int(r["sim_hour"])
        f["n"] += 1
        f["correct"] += int(r["final_correct"])
        f["edge_correct"] += int(r["edge_correct"])
        f["up_kb"] += float(r["bandwidth_kb"])
        f["cloud_only_kb"] += float(r["cloud_only_kb"])
        edge_ms.append(float(r["edge_ms"]))
        if int(r["route_decision"]):
            f["esc"] += 1
            f["hourly"][h] += 1
            hourly[h][r["farm"]] += 1
            if int(r["cloud_ok"]):
                f["cloud_ok"] += 1
                lat_esc.append(float(r["latency_ms"]))
                cloud_ms.append(float(r["cloud_ms"]))
                net_ms.append(float(r["network_ms"]))
        else:
            lat_local.append(float(r["latency_ms"]))
    n = len(rows)
    tot = {k: sum(f[k] for f in farms.values()) for k in ("n", "esc", "correct", "edge_correct", "up_kb", "cloud_only_kb")}
    delays = [float(r.get("answer_delay_h") or 0) for r in rows if int(r["route_decision"]) and int(r["cloud_ok"])]
    answer_all = [float(r["latency_ms"]) for r in rows]   # every image gets an immediate edge answer
    flushes = []
    fl = Path(CFG["run_dir"]) / "flushes.csv"
    if fl.exists():
        with fl.open(encoding="utf-8") as fh:
            flushes = list(csv.DictReader(fh))
    last = rows[-1] if rows else None
    recent = [{k: r.get(k, "") for k in ("sim_time", "farm", "image", "true_label", "edge_label", "edge_margin",
                                         "threshold", "route_decision", "cloud_ok", "final_label", "final_correct",
                                         "latency_ms", "bandwidth_kb", "answer_delay_h")}
              for r in rows[-14:][::-1]]
    days = int(meta.get("args", {}).get("days", 7))
    return {
        "meta": {"disclosure": meta.get("disclosure"), "n_planned": meta.get("n_images", 0), "days": days,
                 "started": meta.get("started_real"), "finished": meta.get("finished_real"),
                 "sim_start": meta.get("args", {}).get("sim_start")},
        "clock": {"sim_time": last["sim_time"] if last else None,
                  "sim_hour_index": (int(last["sim_day"]) * 24 + int(last["sim_hour"])) if last else 0},
        "totals": {**tot, "esc_rate": tot["esc"] / n if n else 0.0, "acc": tot["correct"] / n if n else 0.0,
                   "edge_acc": tot["edge_correct"] / n if n else 0.0,
                   "reduction": (tot["cloud_only_kb"] / tot["up_kb"]) if tot["up_kb"] else None},
        "farms": {k: {**{kk: vv for kk, vv in v.items() if kk != "hourly"},
                      "hourly": [v["hourly"].get(h, 0) for h in range(days * 24)]} for k, v in farms.items()},
        "hourly": [{"h": h, **{f: hourly[h].get(f, 0) for f in farms}} for h in range(days * 24)],
        "latency": {"local_p50": _pct(lat_local, 50), "esc_p50": _pct(lat_esc, 50), "esc_p95": _pct(lat_esc, 95),
                    "edge_p50": _pct(edge_ms, 50), "cloud_p50": _pct(cloud_ms, 50), "net_p50": _pct(net_ms, 50),
                    "answer_p50": _pct(answer_all, 50),
                    "delay_h_mean": (sum(delays) / len(delays)) if delays else None,
                    "delay_h_max": max(delays) if delays else None},
        "flushes": {"n": len(flushes), "last": flushes[-1] if flushes else None,
                    "images": sum(int(f["n"]) for f in flushes)},
        "recent": recent,
    }


@app.get("/api/nodes")
def nodes():
    with _lock:
        if time.time() - _nodes_cache["t"] < 4 and _nodes_cache["data"] is not None:
            return _nodes_cache["data"]
    out = []
    for i, f in enumerate(CFG["farms"]):
        url = f"http://{CFG['edge_host']}:{CFG['base_port'] + i}/health"
        try:
            h = httpx.get(url, timeout=2.5).json()
            out.append({"id": h.get("node_id", f"farm-{f}"), "role": "edge", "online": True,
                        "detail": f"MobileNetV3-L · τ={h.get('threshold')}"})
        except Exception:
            out.append({"id": f"farm-{f}", "role": "edge", "online": False, "detail": "unreachable"})
    try:
        h = httpx.get(CFG["cloud_url"].rstrip("/") + "/health", timeout=2.5).json()
        # show the device, not the host name (not the board's host name)
        label = "jetson-nano" if "jetson-nano" in h.get("device", "") else h.get("node_id", "cloud")
        out.append({"id": label, "role": "cloud", "online": True,
                    "detail": f"host {h.get('node_id')} · {h.get('mode', '')} mode · {h.get('n_requests', 0)} images"})
    except Exception:
        out.append({"id": "cloud", "role": "cloud", "online": False, "detail": "unreachable"})
    with _lock:
        _nodes_cache.update(t=time.time(), data=out)
    return out


@app.get("/img")
def img(path: str):
    root = Path(CFG["image_dir"]).resolve()
    p = (root / path).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


NO_CACHE = {"Cache-Control": "no-store"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.get("/static/{name}")
def static(name: str):
    p = (STATIC / name).resolve()
    if p.parent != STATIC.resolve() or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, headers=NO_CACHE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--edge-host", default="raspberrypi.local")
    ap.add_argument("--farms", default="A,B,C,D")
    ap.add_argument("--base-port", type=int, default=8101)
    ap.add_argument("--cloud-url", default="http://jetson.local:8001")
    ap.add_argument("--port", type=int, default=9100)
    a = ap.parse_args()
    CFG.update(run_dir=a.run_dir, image_dir=a.image_dir, edge_host=a.edge_host,
               farms=[f.strip() for f in a.farms.split(",") if f.strip()],
               base_port=a.base_port, cloud_url=a.cloud_url)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
