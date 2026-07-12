# CloudOffEdge — selective edge→cloud plant-disease inference

Two-device deployment: a **Raspberry Pi** (edge, MobileNetV3-Large) routes only
its low-confidence images to a **Jetson Nano** (cloud, QualityGate fusion over
DINOv2-L + DINOv3-L + CLIP-L), over Tailscale. Confident edge predictions never
leave the Pi, cutting bandwidth ~4.9× versus always-cloud while retaining ~95%
of cloud accuracy on PlantWild.

```
[test images]
     |
     v                          Tailscale (WireGuard)
+----------+   POST /predict    +----------+
|  client  | -----------------> |   edge   | --margin high? keep edge result
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

Routing lives on the edge: if `margin = top1_prob - top2_prob` falls below
`MARGIN_THRESHOLD` (default 0.20 → ~20.5% offload at 95% of cloud accuracy),
the edge re-encodes the JPEG and POSTs it to the cloud; otherwise it returns
its own prediction. Cloud-call failures fall back to the edge prediction.

## Checkpoints are bundled

`deploy/checkpoints/` already contains the trained weights, so `setup.sh`
skips any download:

| File | Size | Used by |
|---|---|---|
| `edge_v3l.pt` | 17 MB | Pi (edge) |
| `qualitygate_fusion_3L.pt` | 9 MB | Jetson (cloud) fusion head |
| `class_names.txt` | 89 classes | both |

The cloud's three backbones (~1.5 GB) still download once via torch.hub /
timm / open_clip on the Jetson's first start, then cache locally.

## On the Raspberry Pi (edge)

```bash
sudo apt-get update && sudo apt-get install -y git curl
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up

git clone https://github.com/YouROS12/CloudOffEdge.git ~/CloudOffEdge
cd ~/CloudOffEdge
bash deploy/setup.sh edge

# set CLOUD_URL to the Jetson's Tailscale MagicDNS name:
#   /opt/plant-deploy/deploy/config/.env
sudo systemctl restart plant-edge
journalctl -u plant-edge -f
```

## On the Jetson Nano (cloud)

```bash
sudo apt-get update && sudo apt-get install -y git curl
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up

git clone https://github.com/YouROS12/CloudOffEdge.git ~/CloudOffEdge
cd ~/CloudOffEdge
bash deploy/setup.sh cloud
journalctl -u plant-cloud -f
```

> **JetPack-4 caveat:** on the Nano, torchvision has no prebuilt aarch64 wheel
> matching NVIDIA's torch — build it from source (clone `pytorch/vision` at the
> matching branch, e.g. `v0.12.0` for torch 1.11, `python setup.py install`
> inside the venv). The script prints a reminder.

## Run a batch inference (from any Tailnet host)

```bash
pip install -r deploy/client/requirements.txt

python deploy/client/batch_predict.py \
  --image-dir /path/to/PlantWild/test \
  --edge-url http://<pi-tailscale-name>:8000 \
  --out-csv  deploy_run.csv \
  --label-from-dirname
```

Expected on full PlantWild test: **~20.5% offload · ~51 KB/sample · ~76.3% top-1**.

## Dashboard (run on your laptop, not the devices)

```bash
make -C deploy dashboard-deps
make -C deploy dashboard          # http://0.0.0.0:9000
```

Set `DASHBOARD_URL=http://<laptop-tailscale-name>:9000` in each device's
`/opt/plant-deploy/deploy/config/.env`, restart the service, and nodes
self-register with heartbeat telemetry (offload %, avg margin/bandwidth/latency).

## Tuning

`MARGIN_THRESHOLD` in `deploy/config/.env` — higher = more escalation = higher
accuracy + bandwidth.

| Threshold | ~Offload | ~Acc |
|---:|---:|---:|
| 0.10 | 12% | ~73% |
| 0.20 | 20% | ~76% (default, 95% of cloud) |
| 0.30 | 28% | ~77% |
| 0.50 | 49% | ~79% |

See `deploy/README.md` for full ops details (logs, systemd, manual checkpoint
placement, multi-Pi setups).
