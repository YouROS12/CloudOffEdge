# results — measured data behind the paper

`experiments/scripts/make_paper_numbers.py` reads these files and writes every measured
value of the manuscript into `paper/numbers.tex`; nothing is transcribed by hand.

| Path | Content | Paper |
|---|---|---|
| `fleet_run/` | replay 1 on the devices: `events.csv` (one row per image: farm, simulated time, margin, routing decision, bytes uploaded, edge/network/cloud latency, diagnoses), `flushes.csv` (cloud batches: size, load and compute time), `summary.json`, `meta.json` (arguments) | §6.6, Tab. 12, Fig. 11–14 |
| `fleet_run_improved/` + `fleet_run_improved_DEPLOYMENT.md` | replay 2 (distilled edge, fp16 CLIP), same format; deployment record with checksums and checks | §6.7, Tab. 13 |
| `nano_flush_profile/`, `nano_flush_profile_clip16/` | `tegrastats` RAM and board power (≈8 Hz) during two 48-image batches, with batch markers | §5.4, Fig. 5, §6.5 |
| `pi_energy/` | Raspberry Pi 5 PMIC rail power during 300 edge inferences | §6.5 |
| `activation_ranges.json` | largest residual-stream and linear-layer activation per block, 64 test images | §5.3, Fig. 4 |
| `trt_verify_report.json`, `trt_verify_engine_only.json` | TensorRT parity on the Nano: complete pipeline / engines fed workstation tensors | §5.5, Tab. 5 |
| `trt_engine_sizes.json`, `clip_fp16_check.json` | engine sizes per backbone; CLIP fp16 vs fp32 re-check | §5.4, §5.3 |
| `threshold_calibration.json`, `threshold_calibration_kd.json` | validation-calibrated τ and resulting test operating point, both edge models | §3.4, §6.7 |
| `edge_fidelity_kd.json` | deployed distilled edge vs offline predictions, 300 images | §6.7 |
| `edge_loop/` | distillation result, calibrated operating points, one-week loop pilot | §6.4, Tab. 10 |
| `option_A_3arch/`, `option_A_v3l/` | escalation sweeps (margin, max-probability, entropy, oracle) and architecture benchmark | §6.1–6.2, Tab. 6–8 |
| `option_B_verifier*/` | cloud-supervised verifier: training logs, AUC, sweeps | §6.3, Tab. 9 |
| `phase0*/` | routing-signal study with MobileNetV3-Small (learned edge-side routers) | §6.1 |
| `device_probe_raspberrypi.json` | hardware/software facts of the edge device | Tab. 1 |

The replay logs record the testbed's private overlay-network (Tailscale) addresses as
they were used; they are not reachable from outside that network.
