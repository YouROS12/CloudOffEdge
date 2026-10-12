# experiments — workstation side

Everything that does not run on the devices: the fleet emulator that drives the
seven-day replay, the operations console, and the scripts behind every result.
Install: `pip install -r experiments/requirements.txt`. The scripts use the PlantWild
split in `artifacts/plantwild_split.csv` through `scripts/plantwild.py`; set
`PLANTWILD_ROOT` to the dataset's `images/` directory (see `data/README.md`).

## Replay on the devices (Section 4.4, 6.6, 6.7)

```bash
python experiments/scripts/plantwild.py --materialize-test data/plantwild_test
python experiments/client/fleet_sim.py --image-dir data/plantwild_test \
  --edge-host <pi-address> --farms A,B,C,D --base-port 8101 \
  --cloud-url http://<jetson-address>:8001 --assign themed --days 7 --day-start 7 --day-end 19 \
  --real-seconds-per-tick 1.0 --sim-start 2026-06-01T00:00:00 --jpeg-quality 90 --seed 0 \
  --flush-hours 12,18 --out-dir results/fleet_run_new
python experiments/console/server.py --run-dir results/fleet_run_new --image-dir data/plantwild_test \
  --edge-host <pi-address> --cloud-url http://<jetson-address>:8001 --port 9100   # http://localhost:9100
```

`client/check_edge_fidelity.py` verifies a deployed edge node against the offline
predictions before a run; `client/batch_predict.py` sends a folder to one node.

## Scripts → results → paper

| Script | Produces | Paper |
|---|---|---|
| `scripts/train_edge_model.py` | edge models (`artifacts/edge_model*/`) | §3.3 |
| `scripts/predict_edge.py`, `predict_edge_val.py --edge-only` | per-image edge predictions (`artifacts/*_predictions_*.csv`) | §3.4, Tab. 5 |
| `scripts/calibrate_threshold.py` | validation-calibrated τ (`results/threshold_calibration*.json`) | §3.4 |
| `scripts/benchmark_archs.py`, `sweep_three_arch.py`, `sweep_v3l.py`, `oracle_routing.py` | escalation curves, GMACs (`results/option_A_*`) | §6.1–6.2, Tab. 6–8, Fig. 6–7 |
| `scripts/sweep_routing_signals.py` | learned edge-side routers (`results/phase0_routing_signals/`) | §6.1 |
| `scripts/extract_edge_embeddings.py`, `train_verifier.py`, `sweep_verifier.py` | cloud-supervised verifier (`results/option_B_*`) | §6.3, Tab. 9, Fig. 10 |
| `scripts/cloud_soft_labels.py` | ensemble probabilities (`artifacts/cloud_soft_labels.pt`; needs paper 1's cached features) | §6.4 |
| `scripts/train_edge_distill.py offline` / `loop` | distilled edge model; one-week loop pilot (`results/edge_loop/`) | §6.4, Tab. 10 |
| `scripts/eval_kd_operating_point.py` | operating points of both edge models (`results/edge_loop/kd_operating_point.json`) | Tab. 10 |
| `scripts/profile_activation_ranges.py` | per-block activation ranges (`results/activation_ranges.json`) | §5.3, Fig. 4 |
| `scripts/measure_nano_flush.py` | Nano RAM and power during a batch (`results/nano_flush_profile*/`) | §5.4, Fig. 5 |
| `scripts/measure_pi_energy.py` | Pi energy per inference (`results/pi_energy/`) | §6.5 |
| `scripts/make_paper_numbers.py` | `paper/numbers.tex` — every measured value as a LaTeX macro | all |
| `scripts/make_paper_figures.py` | `paper/figures/*` | all figures |
| `scripts/make_docx.py` | `paper/main.docx` | — |

The measurement scripts reach the devices over SSH (`--host user@device`) and HTTP;
defaults use the placeholders `raspberrypi.local` and `jetson.local`.
