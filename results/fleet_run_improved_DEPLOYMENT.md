# Replay 2 — improved configuration (2026-10-12)

Same replay as `results/fleet_run/` (same arguments, images, schedule, payload,
flush hours), with two components changed. Everything else is identical.

| Component | Replay 1 (`fleet_run/`) | Replay 2 (`fleet_run_improved/`) |
|---|---|---|
| Edge model | MobileNetV3-L, `outputs/edge_model_v3l/best.pt` (Pi: `checkpoints/edge_v3l.pt`, SHA-256 `6b9acfe5…c17c`) | MobileNetV3-L distilled from the ensemble (focal + KL to ensemble probabilities, T=2, α=0.5), `outputs/edge_model_v3l_kd/best.pt` (Pi: `checkpoints/edge_v3l_kd.pt`, SHA-256 `36e0f03d04159a663193cfd28e38b95dac6e6ccf8b12856ceb0e722f139ca9bd`) |
| Threshold τ | 0.138424 (`results/threshold_calibration.json`) | 0.071893 (`results/threshold_calibration_kd.json`), same rule |
| Cloud engines | `/opt/plant-deploy/trt` (DINOv2 fp16, DINOv3 fp32, CLIP fp32) | `/opt/plant-deploy/trt_clip16` (DINOv2 fp16, DINOv3 fp32, CLIP fp16; DINOv2/DINOv3 engines are the same files, linked) |

Offline prediction for replay 2 at τ = 0.071893: test escalation 12.35 %,
delivered accuracy 75.36 % (94.0 % of cloud), reduction 8.1×.

Checks before the run
- Edge fidelity: 300/300 label agreement with the distilled model's offline
  predictions, margin |diff| mean 6.2e-4, max 3.7e-3 (`results/edge_fidelity_kd.json`).
- CLIP fp16 engines: cosine ≥ 0.99982 to the fp32 engines on the 24 reference
  images; pipeline top-1 agreement 23/24, as fp32 (`results/clip_fp16_check.json`).
- All four farm services report `threshold 0.071893`; cloud `/health` OK.

How it was deployed and how to revert: `docs/ADAPTATIONS.md`, item C6.

Command
```
python experiments/client/fleet_sim.py --image-dir outputs/plantwild_test --edge-host raspberrypi.local \
  --farms A,B,C,D --base-port 8101 --cloud-url http://jetson.local:8001 --assign themed \
  --days 7 --day-start 7 --day-end 19 --real-seconds-per-tick 1.0 --sim-start 2026-06-01T00:00:00 \
  --jpeg-quality 90 --seed 0 --timeout 300 --flush-hours 12,18 --out-dir results/fleet_run_improved
```
