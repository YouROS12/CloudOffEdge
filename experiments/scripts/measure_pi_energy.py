"""Energy of edge inference on the Raspberry Pi 5, from the PMIC rail readings.

The Pi 5's power-management IC reports voltage and current per supply rail
(`vcgencmd pmic_read_adc`); their products summed over rails give the power of
the SoC, memory and on-board supplies. Peripherals powered directly from the
5 V input (USB devices, fan) are not included, so this is a lower bound on the
board's draw. Sampled at ~10 Hz on the Pi while the workstation sends N test
images, one at a time, to farm-A's edge service with force_route=edge (pure
edge path, nothing escalated).

  python experiments/scripts/measure_pi_energy.py --n 300
  -> results/pi_energy/{pmic.log, run.json}
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import subprocess
import time
from pathlib import Path

import httpx

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
SAMPLER = r"""nohup bash -c 'while true; do t=$(date +%s.%N); vcgencmd pmic_read_adc | awk -v t="$t" \
 "{split(\$1,a,\"_\"); n=\$1; sub(/_[AV]\$/,\"\",n); split(\$2,b,\"=\"); v=b[2]+0; \
   if (\$1 ~ /_A\$/) I[n]=v; else if (\$1 ~ /_V\$/) U[n]=v} \
   END {p=0; for (k in I) if (k in U) p+=I[k]*U[k]; printf \"%s %.4f\\n\", t, p}"; sleep 0.07; done' \
 > /tmp/pmic.log 2>&1 &"""


def ssh(host: str, cmd: str, timeout: float = 60) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="user@raspberrypi.local")
    ap.add_argument("--edge-url", default="http://raspberrypi.local:8101")
    ap.add_argument("--events", default=str(PAPER2 / "results" / "fleet_run" / "events.csv"))
    ap.add_argument("--image-dir", default=str(PAPER2 / "data" / "plantwild_test"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--idle-s", type=float, default=30)
    ap.add_argument("--out", default=str(PAPER2 / "results" / "pi_energy"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.events, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["farm"] == "A"][: args.n]
    payloads = [base64.b64encode((Path(args.image_dir) / r["image"]).read_bytes()).decode("ascii") for r in rows]

    ssh(args.host, "pkill -f '[w]hile true; do t=' ; rm -f /tmp/pmic.log; true")
    ssh(args.host, SAMPLER)
    try:
        time.sleep(2)
        t_idle = float(ssh(args.host, "date +%s.%N"))
        time.sleep(args.idle_s)
        t0 = float(ssh(args.host, "date +%s.%N"))
        edge_ms = []
        with httpx.Client(base_url=args.edge_url, timeout=60) as c:
            for i, b64 in enumerate(payloads):
                r = c.post("/predict", json={"image_b64": b64, "request_id": f"energy-{i}", "force_route": "edge"})
                r.raise_for_status()
                edge_ms.append(r.json().get("edge_ms"))
        t1 = float(ssh(args.host, "date +%s.%N"))
        time.sleep(10)
    finally:
        ssh(args.host, "pkill -f '[w]hile true; do t=' ; true")
        (out / "pmic.log").write_text(ssh(args.host, "cat /tmp/pmic.log") + "\n", encoding="utf-8")

    samples = [tuple(map(float, l.split())) for l in (out / "pmic.log").read_text().splitlines() if len(l.split()) == 2]
    idle = [p for t, p in samples if t_idle + 2 <= t < t0]
    load = [(t, p) for t, p in samples if t0 <= t <= t1]
    p_idle = sum(idle) / len(idle)
    e = sum((load[i + 1][0] - load[i][0]) * (load[i + 1][1] + load[i][1]) / 2 for i in range(len(load) - 1))
    dur = t1 - t0
    res = {"n": len(payloads), "duration_s": dur, "samples_idle": len(idle), "samples_load": len(load),
           "p_idle_w": p_idle, "p_load_mean_w": sum(p for _, p in load) / len(load),
           "p_load_max_w": max(p for _, p in load), "energy_j": e, "energy_per_image_j": e / len(payloads),
           "incremental_per_image_j": (e - p_idle * dur) / len(payloads),
           "edge_ms_mean": sum(edge_ms) / len(edge_ms), "t_idle": t_idle, "t0": t0, "t1": t1,
           "note": "PMIC rails only (SoC, memory, on-board supplies); excludes 5 V peripherals"}
    (out / "run.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
