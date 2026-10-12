"""Memory and power profile of one batched cloud flush on the Jetson Nano.

Replays replay-sized batches (the median batch of the seven-day replay, 48
escalated images, same 448 px / q90 payload) through the running cloud server
(`plant-cloud-trt`, batch mode) while `tegrastats` samples RAM and the board's
power rails (INA3221: POM_5V_IN, POM_5V_GPU, POM_5V_CPU) every 100 ms.

Timeline per run (all timestamps on the Nano's clock):
  idle baseline -> enqueue N images -> POST /flush (run on the Nano) -> idle tail

  python experiments/scripts/measure_nano_flush.py --runs 2 --n 48
Outputs results/nano_flush_profile/: tegrastats.log, runs.json (markers, flush
responses' load/compute timings).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
sys.path[:0] = [str(PAPER2 / "rasp_deploy"), str(PAPER2 / "jetson_deploy")]
from shared.image_io import encode_b64, escalation_payload  # noqa: E402


def ssh(host: str, cmd: str, timeout: float = 120) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh {cmd!r} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="user@jetson.local")
    ap.add_argument("--cloud-url", default="http://jetson.local:8001")
    ap.add_argument("--events", default=str(PAPER2 / "results" / "fleet_run" / "events.csv"))
    ap.add_argument("--image-dir", default=str(PAPER2 / "data" / "plantwild_test"))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--idle-s", type=float, default=30)
    ap.add_argument("--short-side", type=int, default=448)
    ap.add_argument("--out", default=str(PAPER2 / "results" / "nano_flush_profile"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.events, encoding="utf-8") as fh:
        esc = [r["image"] for r in csv.DictReader(fh) if r["route_decision"] == "1"]
    payloads = []
    for rel in esc[: args.n * args.runs]:
        img = Image.open(Path(args.image_dir) / rel).convert("RGB")
        b64, kb = encode_b64(escalation_payload(img, args.short_side), quality=90)
        payloads.append((rel, b64, kb))
    print(f"{len(payloads)} payloads, mean {sum(p[2] for p in payloads) / len(payloads):.1f} KB")

    health = httpx.get(args.cloud_url + "/health", timeout=10).json()
    assert health["mode"] == "batch" and health["queued"] == 0, health

    # tegrastats at 10 Hz, each line prefixed with the Nano's epoch time
    log_remote = "/tmp/nano_flush_tegrastats.log"
    ssh(args.host, f"sudo -n pkill -x tegrastats; rm -f {log_remote}; "
                   f"nohup bash -c 'sudo -n tegrastats --interval 100 | while IFS= read -r l; do "
                   f"echo \"$(date +%s.%N) $l\"; done > {log_remote}' >/dev/null 2>&1 &")
    runs = []
    try:
        for k in range(args.runs):
            batch = payloads[k * args.n:(k + 1) * args.n]
            t_idle = float(ssh(args.host, "date +%s.%N"))
            time.sleep(args.idle_s)
            t_enq0 = float(ssh(args.host, "date +%s.%N"))
            with httpx.Client(timeout=60) as c:
                for rel, b64, kb in batch:
                    c.post(args.cloud_url + "/enqueue",
                           json={"image_b64": b64, "image_kb": kb, "request_id": rel}).raise_for_status()
            # flush on the Nano itself so start/end markers share its clock
            res = ssh(args.host, "date +%s.%N; curl -s -X POST localhost:8001/flush -o /tmp/nano_flush.json; "
                                 "date +%s.%N; python3 -c \"import json; d=json.load(open('/tmp/nano_flush.json')); "
                                 "print(json.dumps({'n': d['n'], 'load_ms': d['load_ms'], "
                                 "'timings': [r['timings_ms'] for r in d['results']]}))\"", timeout=3600)
            t0, t1, payload = res.split("\n", 2)
            d = json.loads(payload)
            time.sleep(args.idle_s)
            t_end = float(ssh(args.host, "date +%s.%N"))
            runs.append({"run": k, "n": d["n"], "t_idle": t_idle, "t_enqueue": t_enq0,
                         "t_flush_start": float(t0), "t_flush_end": float(t1), "t_end": t_end,
                         "load_ms": d["load_ms"], "timings_ms": d["timings"],
                         "payload_kb_mean": sum(p[2] for p in batch) / len(batch)})
            print(f"run {k}: {d['n']} images, flush {float(t1) - float(t0):.0f} s, load "
                  + " ".join(f"{n.split('_')[0]}={v / 1000:.1f}s" for n, v in d["load_ms"].items()))
    finally:
        ssh(args.host, "sudo -n pkill -x tegrastats || true")
        (out / "tegrastats.log").write_text(ssh(args.host, f"cat {log_remote}", timeout=300) + "\n",
                                            encoding="utf-8")
        (out / "runs.json").write_text(json.dumps({"host": args.host, "interval_ms": 100,
                                                   "runs": runs}, indent=1), encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
