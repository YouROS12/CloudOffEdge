# rasp_deploy — the edge tier (Raspberry Pi 5)

Everything that runs on the farm device: an HTTP edge service that classifies each
photograph with MobileNetV3-Large on the CPU, computes the margin between its two most
probable classes, answers immediately, and — only when the margin is below τ — resizes
the image (448 px shorter side, JPEG q90) and places it on the cloud node's queue.

```
edge/server.py          edge service (FastAPI): POST /predict, GET /health, GET /stats
edge/setup.sh           venv + PyTorch (CPU) + checkpoint + .env + systemd (one farm)
edge/setup_farms.sh     N virtual farms on one Pi: one service per farm, ports 8101..
shared/                 config loader, routing rule, image payload, wire protocol, heartbeat
config/                 config.example.yaml, .env.example (copied to config/ on install)
systemd/                plant-edge.service, plant-edge-farm@.service (template, one per farm)
install_pi.sh           one-shot installer for a fresh Raspberry Pi OS (64-bit)
probe_device.py         prints the device facts reported in the paper's Table 1
```

Tested on: Raspberry Pi 5 Model B (16 GB), Debian 13, Python 3.13, PyTorch 2.6 (CPU).
Dependencies are pinned per Python version in `edge/requirements.txt` (appendix item A1).

## Install

```bash
# on the Pi, with this repository checked out (or copied) at ~/CloudOffEdge
cd ~/CloudOffEdge
mkdir -p rasp_deploy/checkpoints
cp artifacts/edge_model_v3l/best.pt rasp_deploy/checkpoints/edge_v3l.pt   # replay 1 model
cp artifacts/class_names.txt        rasp_deploy/checkpoints/class_names.txt
CLOUD_URL=http://<jetson-address>:8001 bash rasp_deploy/install_pi.sh
```

The installer copies `rasp_deploy/` to `/opt/plant-deploy/deploy` (the path the systemd
units use), creates `/opt/plant-deploy/.venv`, writes `config/.env` and starts
`plant-edge`. Then add to `/opt/plant-deploy/deploy/config/.env`:

```
ESCALATION_MODE=deferred        # queue escalations on the cloud node (batched cloud tier)
MARGIN_THRESHOLD=0.138424       # validation-calibrated tau of the replay-1 model
```

and start the four virtual farms used in the paper (A–D on ports 8101–8104):

```bash
bash /opt/plant-deploy/deploy/edge/setup_farms.sh
curl -s localhost:8101/health
```

The four farms are four processes on one physical Pi; report them as emulated edge
nodes, not as four devices (Section 4.4 of the paper).

## The two configurations of the paper

| | Replay 1 | Replay 2 |
|---|---|---|
| Checkpoint (`checkpoints/`) | `artifacts/edge_model_v3l/best.pt` | `artifacts/edge_model_v3l_kd/best.pt` (distilled) |
| `MARGIN_THRESHOLD` | 0.138424 | 0.071893 |

Replay 2 was deployed without touching replay 1's files: the distilled checkpoint was
copied to `checkpoints/edge_v3l_kd.pt`, a copy of `config/config.yaml` with that
`ckpt_path` was selected through `DEPLOY_CONFIG` in `config/.env`, and the threshold was
changed; reverting restores the previous `.env` (docs/ADAPTATIONS.md, item C6).

## Check the deployment

Before measuring, verify that the deployed node reproduces the offline model image by
image (paper, Table 5):

```bash
python experiments/client/check_edge_fidelity.py --image-dir data/plantwild_test \
  --offline artifacts/edge_v3l_predictions_test.csv --edge-url http://<pi-address>:8101 --n 300
```

Expected: 300/300 label agreement (replay 1: margin difference mean 8×10⁻⁴; replay 2:
`artifacts/edge_v3l_kd_predictions_test.csv`, 300/300, mean 6×10⁻⁴).
