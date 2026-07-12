# Paper 2 — Two-device deployment

Pi (edge) + Jetson (cloud), connected over Tailscale, batch-classifying a
folder of plant images. End-to-end story:

```
[test images]
     |
     v                          Tailscale (WireGuard)
+----------+   POST /predict    +----------+
|   client | -----------------> |   edge   | --margin? hold edge result
+----------+                    |  (Pi)    |
                                +----+-----+
                                     | margin < 0.20
                                     | POST /classify
                                     v
                                +----------+
                                |  cloud   |
                                | (Jetson) |
                                +----------+
```

Routing decision lives on the edge: if `margin = top1_prob - top2_prob`
falls below `MARGIN_THRESHOLD` (default 0.20, validated on PlantWild test
to hit ~20.5% offload at 95% of cloud accuracy), the edge re-encodes
the JPEG and POSTs it to the cloud. Otherwise, it returns its own
prediction. Cloud-call failures are caught and the edge prediction is
used as fallback (`source: edge`, `routed: false`).

---

## Multi-node + dashboard

The cloud server is stateless, so you can attach **as many Pis as you want** —
each Pi just needs `CLOUD_URL` pointing at a Jetson/desktop on your Tailnet.
There's no cluster config to update; new nodes self-register.

To see what's connected, run the dashboard on your **laptop** (not on the
Pi/Jetson — keep observability off the constrained devices):

```bash
cd paper2
make -C deploy dashboard-deps
make -C deploy dashboard            # serves http://0.0.0.0:9000
make -C deploy open-dashboard       # opens browser
```

Then on each Pi/Jetson, set in `/opt/plant-deploy/deploy/config/.env`:

```
DASHBOARD_URL=http://<your-laptop-tailscale-name>:9000
```

…and `sudo systemctl restart plant-edge` (or `plant-cloud`). Each node's
`HeartbeatThread` POSTs `/register` once on startup and `/heartbeat` every
15s with stats (uptime, requests, routed%, avg margin, avg bandwidth, avg
latency). The dashboard partitions nodes into edge / cloud / offline and
auto-marks anyone silent >60s as offline. Node IDs default to
`socket.gethostname()`; override with `NODE_ID=…` in `.env` if needed.

If `DASHBOARD_URL` is unset or unreachable, nodes silently skip heartbeats —
no impact on serving traffic.

---

## Hardware notes

| Device | Role | Constraints baked in |
|---|---|---|
| Raspberry Pi 4 / 5    | edge  | CPU-only torch wheels; MobileNetV3-L runs in 50–150 ms |
| Jetson Nano (4GB)     | cloud | Sequential backbone loading is mandatory (3×~600MB fp16 backbones don't fit simultaneously). Per-image latency ~15–40 s. |
| Jetson Orin (8GB+)    | cloud | Resident mode works; per-image latency ~0.5 s. |
| Desktop GPU (≥6 GB)   | cloud | Resident mode; per-image latency ~0.2 s. |

If the Nano is too slow for your demo, point the edge's `CLOUD_URL` at a
desktop on the same Tailscale net — nothing else changes.

---

## One-time prep on your dev machine

You need to ship two checkpoints + the class-name list to both devices.

```powershell
# 1. Build the tarball from your trained outputs/
python deploy\package_checkpoints.py `
  --edge-ckpt  outputs\edge_model_v3l\best.pt `
  --cloud-ckpt outputs\cloud_head\qualitygate_fusion_3L.pt `
  --out deploy\checkpoints.tar.gz

# 2. Host it somewhere reachable from both devices. Easiest: a private GitHub
#    Release. (Or scp it directly to each device — see "Without CHECKPOINTS_URL"
#    below.)
gh release create v1-deploy deploy\checkpoints.tar.gz
# -> note the asset URL, e.g.:
#    https://github.com/<you>/<repo>/releases/download/v1-deploy/checkpoints.tar.gz
```

---

## On the Raspberry Pi (edge)

```bash
sudo apt-get update && sudo apt-get install -y git curl
# Tailscale (skip if already installed):
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Pull, install, start
git clone https://github.com/<you>/<repo>.git ~/plant-deploy
cd ~/plant-deploy/paper2
CHECKPOINTS_URL=https://.../checkpoints.tar.gz bash deploy/setup.sh edge

# Then edit /opt/plant-deploy/deploy/config/.env and set CLOUD_URL to the
# Jetson's Tailscale MagicDNS name, then:
sudo systemctl restart plant-edge
journalctl -u plant-edge -f
```

## On the Jetson Nano (cloud)

```bash
sudo apt-get update && sudo apt-get install -y git curl
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

git clone https://github.com/<you>/<repo>.git ~/plant-deploy
cd ~/plant-deploy/paper2
CHECKPOINTS_URL=https://.../checkpoints.tar.gz bash deploy/setup.sh cloud
journalctl -u plant-cloud -f
```

The first start downloads the three backbones via torch.hub / timm /
open_clip — about 1.5 GB. Subsequent starts use the local cache.

> **Jetson Nano JetPack-4 caveat:** `setup.sh cloud` installs torch from
> NVIDIA's wheel. torchvision must be **built from source** on JetPack 4
> (no prebuilt aarch64 wheel for the matched torch). The script prints a
> message; clone `pytorch/vision` at the matching branch (e.g. `v0.12.0`
> for torch 1.11) and `python setup.py install` inside the venv.

## Without CHECKPOINTS_URL (manual placement)

If you don't want to host a tarball:

```bash
# On dev machine
scp -r outputs/edge_model_v3l/best.pt pi@<pi>:/tmp/
scp outputs/cloud_head/qualitygate_fusion_3L.pt nvidia@<jetson>:/tmp/
```

On each device, place files into `/opt/plant-deploy/deploy/checkpoints/`
named `edge_v3l.pt` (Pi) and `qualitygate_fusion_3L.pt` (Jetson), plus
`class_names.txt` (one class name per line, label-id order — generated by
`package_checkpoints.py`).

---

## Run a batch inference

Anywhere on your Tailnet — laptop, dev machine, or even the Pi itself:

```bash
cd paper2
pip install -r deploy/client/requirements.txt

python deploy/client/batch_predict.py \
  --image-dir /path/to/PlantWild/test \
  --edge-url http://<pi-tailscale-name>:8000 \
  --out-csv  results/deploy_run.csv \
  --label-from-dirname
```

Live progress shows offload rate, average bandwidth, average latency, and
top-1 accuracy (if labels are available). The CSV is written incrementally
so a crash mid-run doesn't lose data.

Verify against expected paper-2 numbers:

| Metric | Expected on full PlantWild test |
|---|---|
| Offload rate     | ~20.5% |
| Avg bandwidth    | ~51 KB/sample |
| Top-1 accuracy   | ~76.3% |

---

## Quick sanity check (single image, no network)

```bash
# Run both servers locally (different ports), point edge at cloud:
DEPLOY_ROLE=cloud SERVER_PORT=8001 python -m cloud.server &
DEPLOY_ROLE=edge  SERVER_PORT=8000 CLOUD_URL=http://localhost:8001 python -m edge.server &

# Then:
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$(base64 -w0 sample.jpg)\"}" | jq
```

---

## Layout

```
deploy/
  setup.sh                           # autodetect + dispatch
  package_checkpoints.py             # one-time tarball build (dev machine)
  edge/
    server.py                        # FastAPI: POST /predict
    setup.sh                         # Pi installer
    requirements.txt
  cloud/
    server.py                        # FastAPI: POST /classify
    setup.sh                         # Jetson installer (wheel-aware)
    requirements.txt
  client/
    batch_predict.py                 # walk folder -> POST /predict -> CSV
    requirements.txt
  shared/
    config.py protocol.py routing.py image_io.py
    stats.py heartbeat.py            # roster telemetry
  dashboard/
    server.py requirements.txt
    static/index.html static/style.css static/app.js
  systemd/
    plant-edge.service plant-cloud.service
    plant-dashboard.service          # optional, for laptop-as-server
  config/
    config.example.yaml .env.example
  checkpoints/                       # populated by setup.sh / package_checkpoints.py
    edge_v3l.pt qualitygate_fusion_3L.pt class_names.txt
```

## Logs and ops

```bash
# Live logs
journalctl -u plant-edge  -f
journalctl -u plant-cloud -f

# Restart
sudo systemctl restart plant-edge
sudo systemctl restart plant-cloud

# Status
systemctl status plant-edge
systemctl status plant-cloud

# Health check
curl http://<pi-ts>:8000/health
curl http://<jetson-ts>:8001/health
```

## Tuning the routing threshold

`MARGIN_THRESHOLD` in `deploy/config/.env` (or `edge.margin_threshold` in
`config.yaml`). Higher = more aggressive escalation = higher accuracy +
higher bandwidth. Calibrate on val:

| Threshold | ~Offload | ~Acc | Notes |
|---:|---:|---:|---|
| 0.10 | 12% | ~73% | aggressive edge, low bw |
| 0.20 | 20% | ~76% | **default — at 95% of cloud** |
| 0.30 | 28% | ~77% | conservative, more bw |
| 0.50 | 49% | ~79% | nearly cloud-only |
