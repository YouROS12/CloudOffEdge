"""Fleet simulator: a time-compressed multi-farm deployment against real
edge nodes.

Replays a labelled image set as if N farms captured it over a simulated
period (default: 4 farms, 7 days, captures during daylight hours). Every
image is a REAL round trip through an edge node (and, if the edge routes
it, through the real cloud). Only the idle time between captures is
compressed: one tick = one simulated hour = --real-seconds-per-tick of
wall time, stretched if processing that hour's captures takes longer.

Requests are sent sequentially (round-robin across farms), so the measured
edge latency is that of one request on an otherwise idle device -- the same
as a real single-farm Pi. When the farms are N edge processes on ONE
physical Pi (edge/setup_farms.sh), report them as "N virtual edge nodes
emulated on one Raspberry Pi", not as N devices.

Usage:
  python fleet_sim.py \
    --image-dir /path/to/plantwild_test \
    --edge-host raspberrypi.local --farms A,B,C,D --base-port 8101 \
    --cloud-url http://jetson.local:8001 \
    --days 7 --real-seconds-per-tick 1.0 \
    --out-dir results/fleet_run

Outputs in --out-dir:
  events.csv    one row per image (sim time, farm, routing, latency, bytes)
  meta.json     full configuration, farm->crop map, disclosure statement
  summary.json  per-farm and per-farm-per-day aggregates
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rasp_deploy"))
from shared.image_io import encode_b64, escalation_payload  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Four plausible farm types over the PlantWild crops. Uneven sizes and
# different crop mixes are deliberate: farms differ in how often the edge
# model is unsure, which is what the per-farm results should show.
THEMED_FARMS = {
    "A": ("orchard", ["apple", "banana", "blueberry", "cherry", "citrus",
                      "coffee", "grape", "grapevine", "maple", "peach",
                      "plum", "raspberry", "strawberry"]),
    "B": ("solanaceae", ["bell pepper", "eggplant", "potato", "tobacco",
                         "tomato"]),
    "C": ("field crops", ["bean", "corn", "garlic", "ginger", "rice",
                          "soybean"]),
    "D": ("market garden", ["basil", "broccoli", "cabbage", "carrot",
                            "cauliflower", "celery", "cucumber", "lettuce",
                            "squash", "zucchini"]),
}

EVENT_FIELDS = [
    "sim_time", "sim_day", "sim_hour", "farm", "edge_node",
    "image", "true_label",
    "edge_label", "edge_confidence", "edge_margin", "threshold",
    "route_decision", "routed", "cloud_ok", "source", "final_label",
    "edge_correct", "final_correct",
    "bandwidth_kb", "cloud_only_kb",
    "latency_ms", "edge_ms", "cloud_ms", "network_ms", "client_rtt_ms",
    "cloud_job_id", "answer_delay_h", "cloud_batch_n",
    "real_time",
]
FLUSH_FIELDS = ["sim_time", "n", "load_ms_total", "load_ms", "compute_ms_median", "wall_s", "real_time"]


def crop_of(classname: str, crops: list[str]) -> str | None:
    # Longest match first so "grapevine" wins over "grape".
    for c in sorted(crops, key=len, reverse=True):
        if classname == c or classname.startswith(c + " "):
            return c
    return None


def assign_farms(images: list[tuple[Path, str]], farms: list[str],
                 mode: str, rng: random.Random) -> tuple[dict, dict]:
    """Return ({farm: [(path, label), ...]}, {farm: description})."""
    by_farm: dict[str, list] = {f: [] for f in farms}
    if mode == "themed":
        if sorted(farms) != sorted(THEMED_FARMS):
            raise SystemExit("--assign themed needs exactly farms A,B,C,D")
        crop_to_farm = {c: f for f, (_, crops) in THEMED_FARMS.items() for c in crops}
        all_crops = list(crop_to_farm)
        for p, lab in images:
            c = crop_of(lab, all_crops)
            if c is None:
                raise SystemExit(f"class '{lab}' has no farm in THEMED_FARMS")
            by_farm[crop_to_farm[c]].append((p, lab))
        desc = {f: {"type": t, "crops": crops} for f, (t, crops) in THEMED_FARMS.items()}
    else:
        shuffled = images[:]
        rng.shuffle(shuffled)
        for i, item in enumerate(shuffled):
            by_farm[farms[i % len(farms)]].append(item)
        desc = {f: {"type": "random split", "crops": "all"} for f in farms}
    return by_farm, desc


def build_schedule(by_farm: dict, days: int, day_start: int, day_end: int,
                   rng: random.Random) -> dict[int, list]:
    """Spread each farm's images over the daylight hours of `days` days.
    Returns {sim_hour_index: [(farm, path, label), ...]} with farms
    interleaved round-robin inside each hour."""
    slots = [d * 24 + h for d in range(days) for h in range(day_start, day_end)]
    per_hour: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for farm, items in by_farm.items():
        items = items[:]
        rng.shuffle(items)
        for i, (p, lab) in enumerate(items):
            # Even spread: image i lands in slot floor(i * S / n).
            slot = slots[(i * len(slots)) // max(1, len(items))]
            per_hour[slot][farm].append((p, lab))
    schedule: dict[int, list] = {}
    for hour, farms in per_hour.items():
        queues = {f: list(v) for f, v in sorted(farms.items())}
        merged = []
        while any(queues.values()):
            for f in list(queues):
                if queues[f]:
                    p, lab = queues[f].pop(0)
                    merged.append((f, p, lab))
        schedule[hour] = merged
    return schedule


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q / 100 * (len(xs) - 1)))))
    return xs[k]


def summarize(rows: list[dict]) -> dict:
    def agg(rs: list[dict]) -> dict:
        n = len(rs)
        if n == 0:
            return {"n": 0}
        f = lambda k: [float(r[k]) for r in rs]  # noqa: E731
        routed = [r for r in rs if int(r["routed"])]
        local = [r for r in rs if not int(r["route_decision"])]
        up = sum(f("bandwidth_kb"))
        base = sum(f("cloud_only_kb"))
        return {
            "n": n,
            "route_decision_rate": sum(int(r["route_decision"]) for r in rs) / n,
            "offload_rate": len(routed) / n,
            "cloud_failures": sum(int(r["route_decision"]) and not int(r["cloud_ok"]) for r in rs),
            "edge_acc": sum(int(r["edge_correct"]) for r in rs) / n,
            "final_acc": sum(int(r["final_correct"]) for r in rs) / n,
            "upload_mb": up / 1024,
            "cloud_only_mb": base / 1024,
            "kb_per_image": up / n,
            "cloud_only_kb_per_image": base / n,
            "bandwidth_reduction_x": (base / up) if up > 0 else float("inf"),
            "edge_ms_p50": pct(f("edge_ms"), 50),
            "edge_ms_p95": pct(f("edge_ms"), 95),
            "local_latency_ms_p50": pct([float(r["latency_ms"]) for r in local], 50),
            "routed_latency_ms_p50": pct([float(r["latency_ms"]) for r in routed], 50),
            "routed_latency_ms_p95": pct([float(r["latency_ms"]) for r in routed], 95),
            "cloud_ms_p50": pct([float(r["cloud_ms"]) for r in routed], 50),
            "cloud_ms_p95": pct([float(r["cloud_ms"]) for r in routed], 95),
            "network_ms_p50": pct([float(r["network_ms"]) for r in routed], 50),
            "answer_delay_h_mean": (statistics.mean(float(r.get("answer_delay_h") or 0) for r in routed)
                                    if routed else float("nan")),
            "answer_delay_h_max": max((float(r.get("answer_delay_h") or 0) for r in routed), default=float("nan")),
        }

    by_farm = defaultdict(list)
    by_farm_day = defaultdict(list)
    by_hour = defaultdict(int)
    for r in rows:
        by_farm[r["farm"]].append(r)
        by_farm_day[(r["farm"], int(r["sim_day"]))].append(r)
        if int(r["routed"]):
            by_hour[int(r["sim_day"]) * 24 + int(r["sim_hour"])] += 1
    return {
        "overall": agg(rows),
        "per_farm": {f: agg(rs) for f, rs in sorted(by_farm.items())},
        "per_farm_day": {
            f"{f}/day{d}": {k: v for k, v in agg(rs).items()
                            if k in ("n", "offload_rate", "upload_mb", "cloud_only_mb", "final_acc")}
            for (f, d), rs in sorted(by_farm_day.items())
        },
        "cloud_requests_per_sim_hour": {
            "max": max(by_hour.values(), default=0),
            "mean_over_active_hours": statistics.mean(by_hour.values()) if by_hour else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True,
                    help="Root with <class_name>/<image> layout (labels from dirnames).")
    ap.add_argument("--edge-host", required=True, help="Host running the farm edge nodes.")
    ap.add_argument("--farms", default="A,B,C,D")
    ap.add_argument("--base-port", type=int, default=8101,
                    help="Farm i listens on base-port + i (see edge/setup_farms.sh).")
    ap.add_argument("--cloud-url", default="",
                    help="Cloud server URL for a pre-flight health check.")
    ap.add_argument("--allow-no-cloud", action="store_true",
                    help="Run even if the cloud is unreachable (routed images fall back to edge).")
    ap.add_argument("--assign", choices=["themed", "random"], default="themed")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--day-start", type=int, default=7, help="First capture hour (sim clock).")
    ap.add_argument("--day-end", type=int, default=19, help="Captures stop at this hour.")
    ap.add_argument("--real-seconds-per-tick", type=float, default=1.0,
                    help="Wall-clock seconds per simulated hour (minimum; stretches if busy).")
    ap.add_argument("--max-images-per-farm", type=int, default=0, help="0 = all.")
    ap.add_argument("--sim-start", default="2026-06-01T00:00:00",
                    help="Simulated wall clock at tick 0.")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="Skip images already in out-dir/events.csv.")
    ap.add_argument("--flush-hours", default="12,18",
                    help="Deferred escalation: the cloud processes its queue after the captures "
                         "of these simulated hours (answers available at the next hour).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    farms = [f.strip() for f in args.farms.split(",") if f.strip()]
    root = Path(args.image_dir)
    images = sorted((p, p.parent.name) for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"no images under {root}")

    by_farm, farm_desc = assign_farms(images, farms, args.assign, rng)
    if args.max_images_per_farm:
        for f in by_farm:
            rng.shuffle(by_farm[f])
            by_farm[f] = sorted(by_farm[f][:args.max_images_per_farm])
    schedule = build_schedule(by_farm, args.days, args.day_start, args.day_end, rng)
    n_total = sum(len(v) for v in schedule.values())

    # --- pre-flight -------------------------------------------------------
    clients = {}
    payload: dict[str, tuple[int, int]] = {}  # farm -> (escalation short side, JPEG quality)
    for i, f in enumerate(farms):
        url = f"http://{args.edge_host}:{args.base_port + i}"
        c = httpx.Client(base_url=url, timeout=args.timeout)
        try:
            h = c.get("/health").json()
        except Exception as e:
            raise SystemExit(f"farm {f} unreachable at {url}: {e}")
        print(f"[farm {f}] {url} node_id={h.get('node_id')} threshold={h.get('threshold')} "
              f"payload={h.get('escalation_short_side')}px q{h.get('jpeg_quality')} cloud={h.get('cloud_url')}")
        clients[f] = (c, h.get("node_id", f"farm-{f}"))
        payload[f] = (int(h.get("escalation_short_side", 0)), int(h.get("jpeg_quality", args.jpeg_quality)))
    if args.cloud_url:
        try:
            ch = httpx.get(args.cloud_url.rstrip("/") + "/health", timeout=15).json()
            print(f"[cloud] {args.cloud_url} {ch}")
        except Exception as e:
            if not args.allow_no_cloud:
                raise SystemExit(f"cloud unreachable at {args.cloud_url}: {e} "
                                 f"(pass --allow-no-cloud to run edge-only)")
            print(f"[cloud] UNREACHABLE ({e}) -- routed images will fall back to edge")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev_path = out / "events.csv"
    done: set[tuple[str, str]] = set()
    if args.resume and ev_path.exists():
        with ev_path.open(encoding="utf-8") as fh:
            done = {(r["farm"], r["image"]) for r in csv.DictReader(fh)}
        print(f"[resume] {len(done)} images already done")

    sim0 = datetime.fromisoformat(args.sim_start)
    meta = {
        "description": "Time-compressed multi-farm deployment replay.",
        "disclosure": (f"{len(farms)} virtual edge nodes (farms {','.join(farms)}) "
                       f"emulated as separate processes on one physical edge device "
                       f"({args.edge_host}); 1 simulated hour = "
                       f">= {args.real_seconds_per_tick} s wall clock. Every inference "
                       f"and every edge->cloud transfer is a real measurement; only the "
                       f"idle time between captures is compressed."),
        "args": vars(args),
        "farms": {f: {**farm_desc[f], "n_images": len(by_farm[f]),
                      "edge_node": clients[f][1]} for f in farms},
        "n_images": n_total,
        "started_real": datetime.now(timezone.utc).isoformat(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"[plan] {n_total} images, {len(farms)} farms, {args.days} sim days, "
          f"{len(schedule)} active sim hours")
    for f in farms:
        print(f"   farm {f} ({farm_desc[f]['type']}): {len(by_farm[f])} images")

    new_file = not (args.resume and ev_path.exists())
    fh = ev_path.open("w" if new_file else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=EVENT_FIELDS)
    if new_file:
        w.writeheader()

    t_start = time.time()
    n_done = len(done)
    day_rows: list[dict] = []
    flush_hours = {int(h) for h in args.flush_hours.split(",") if h.strip()}
    pending: dict[str, tuple[dict, int]] = {}  # cloud job id -> (row, capture hour)
    fl_path = out / "flushes.csv"
    fl_fh = fl_path.open("w" if new_file else "a", newline="", encoding="utf-8")
    fw = csv.DictWriter(fl_fh, fieldnames=FLUSH_FIELDS)
    if new_file:
        fw.writeheader()

    def flush_cloud(hour: int) -> None:
        """The cloud node's batch job: diagnose every queued escalation."""
        t0 = time.time()
        res = httpx.post(args.cloud_url.rstrip("/") + "/flush", timeout=7200).json()
        wall = time.time() - t0
        for r_ in res["results"]:
            item = pending.pop(r_["job_id"], None)
            if item is None:
                continue
            row_, h_cap = item
            row_.update(cloud_ok=1, source="cloud", final_label=r_["label"],
                        final_correct=int(r_["label"] == row_["true_label"]),
                        cloud_ms=f"{r_['latency_ms']:.1f}", answer_delay_h=hour + 1 - h_cap,
                        cloud_batch_n=res["n"])
            w.writerow(row_)
            day_rows.append(row_)
        fh.flush()
        load = res.get("load_ms", {})
        comp = [r_["latency_ms"] for r_ in res["results"]]
        fw.writerow({"sim_time": (sim0 + timedelta(hours=hour + 1)).isoformat(), "n": res["n"],
                     "load_ms_total": f"{sum(load.values()):.0f}", "load_ms": json.dumps(load),
                     "compute_ms_median": f"{statistics.median(comp):.0f}" if comp else "",
                     "wall_s": f"{wall:.1f}", "real_time": datetime.now(timezone.utc).isoformat()})
        fl_fh.flush()
        print(f"   [cloud flush @ sim {hour + 1:3d}h] {res['n']} images in {wall:.0f}s "
              f"(load {sum(load.values()) / 1000:.0f}s)")

    try:
        for hour in range(args.days * 24):
            t_tick = time.time()
            for farm, path, label in schedule.get(hour, []):
                rel = str(path.relative_to(root)).replace("\\", "/")
                if (farm, rel) in done:
                    continue
                # The farm sends the photo exactly as captured (original file
                # bytes). The always-cloud baseline uploads the same payload the
                # edge uploads when it escalates (same resolution policy and
                # JPEG quality, read from the node's /health), so the two
                # upload paths are directly comparable.
                b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                client, node = clients[farm]
                _, cloud_only_kb = encode_b64(
                    escalation_payload(Image.open(path).convert("RGB"), payload[farm][0]),
                    quality=payload[farm][1])
                t0 = time.perf_counter()
                try:
                    r = client.post("/predict", json={"image_b64": b64,
                                                      "request_id": f"{farm}-{path.stem}"})
                    r.raise_for_status()
                    resp = r.json()
                except Exception as e:
                    print(f"[err] farm {farm} {rel}: {e}")
                    continue
                rtt = (time.perf_counter() - t0) * 1000
                ep = resp["edge_prediction"]
                cp = resp.get("cloud_prediction") or {}
                decision = ep["margin"] < resp["threshold"]
                cloud_ms = float(cp.get("latency_ms", 0.0))
                edge_ms = float(resp.get("edge_ms", 0.0))
                net_ms = (resp["latency_ms"] - edge_ms - cloud_ms) if resp["routed"] else 0.0
                sim_t = sim0 + timedelta(hours=hour)
                row = {
                    "sim_time": sim_t.isoformat(), "sim_day": hour // 24,
                    "sim_hour": hour % 24, "farm": farm, "edge_node": node,
                    "image": rel, "true_label": label,
                    "edge_label": ep["label"],
                    "edge_confidence": f"{ep['confidence']:.6f}",
                    "edge_margin": f"{ep['margin']:.6f}",
                    "threshold": resp["threshold"],
                    "route_decision": int(decision),
                    "routed": int(resp["routed"]),
                    "cloud_ok": int(resp["source"] == "cloud"),
                    "source": resp["source"], "final_label": resp["label"],
                    "edge_correct": int(ep["label"] == label),
                    "final_correct": int(resp["label"] == label),
                    "bandwidth_kb": f"{resp['bandwidth_kb']:.3f}",
                    "cloud_only_kb": f"{cloud_only_kb:.3f}",
                    "latency_ms": f"{resp['latency_ms']:.1f}",
                    "edge_ms": f"{edge_ms:.1f}", "cloud_ms": f"{cloud_ms:.1f}",
                    "network_ms": f"{net_ms:.1f}", "client_rtt_ms": f"{rtt:.1f}",
                    "cloud_job_id": resp.get("cloud_job_id") or "", "answer_delay_h": 0,
                    "cloud_batch_n": "",
                    "real_time": datetime.now(timezone.utc).isoformat(),
                }
                n_done += 1
                if resp.get("cloud_job_id"):
                    pending[resp["cloud_job_id"]] = (row, hour)   # written once the cloud answers
                    continue
                w.writerow(row)
                fh.flush()
                day_rows.append(row)

            if pending and ((hour % 24) in flush_hours or hour == args.days * 24 - 1):
                flush_cloud(hour)
            if hour % 24 == 23 and day_rows:
                s = summarize(day_rows)["overall"]
                el = time.time() - t_start
                print(f"[sim day {hour // 24 + 1}/{args.days}] {s['n']} imgs | "
                      f"offload {s['offload_rate']:.1%} | up {s['upload_mb']:.1f} MB "
                      f"(cloud-only {s['cloud_only_mb']:.1f}) | acc {s['final_acc']:.1%} | "
                      f"{n_done}/{n_total} done, {el / 60:.1f} min wall")
                day_rows = []
            spare = args.real_seconds_per_tick - (time.time() - t_tick)
            if spare > 0:
                time.sleep(spare)
    finally:
        for row_, _ in pending.values():   # never answered (cloud failure): keep edge label
            row_.update(cloud_ok=0, source="edge")
            w.writerow(row_)
        fh.close()
        fl_fh.close()

    with ev_path.open(encoding="utf-8") as fh2:
        rows = list(csv.DictReader(fh2))
    summary = summarize(rows)
    summary["wall_clock_minutes"] = (time.time() - t_start) / 60
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    meta["finished_real"] = datetime.now(timezone.utc).isoformat()
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print("\n farm | type          |    n | offload | up MB | cloud-only MB |  red. | final acc")
    for f in farms:
        s = summary["per_farm"].get(f, {"n": 0})
        if not s["n"]:
            continue
        print(f"   {f}  | {farm_desc[f]['type']:<13} | {s['n']:4d} | {s['offload_rate']:6.1%} | "
              f"{s['upload_mb']:5.1f} | {s['cloud_only_mb']:13.1f} | "
              f"{s['bandwidth_reduction_x']:4.1f}x | {s['final_acc']:.1%}")
    o = summary["overall"]
    print(f"  all |               | {o['n']:4d} | {o['offload_rate']:6.1%} | {o['upload_mb']:5.1f} | "
          f"{o['cloud_only_mb']:13.1f} | {o['bandwidth_reduction_x']:4.1f}x | {o['final_acc']:.1%}")
    print(f"\n wrote {out}/events.csv, summary.json, meta.json")


if __name__ == "__main__":
    main()
