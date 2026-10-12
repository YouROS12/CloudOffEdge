# Software adaptations for embedded deployment

Running log of every change needed to make the reference (desktop) pipeline
run on the embedded testbed. Source for the paper's appendix
(`paper/manuscript/main.tex`, Appendix A). Add an entry whenever a new
adaptation is made; keep the *problem → cause → adaptation → verification*
structure.

Testbed: Raspberry Pi 5 (Debian 13, Python 3.13) as edge; Jetson Nano 4 GB
(JetPack 4.6.1 / L4T R32.7.1, Ubuntu 18.04, Python 3.6.9, CUDA 10.2,
TensorRT 8.2.1) as cloud; Windows 11 workstation (Python 3.12, PyTorch 2.11,
timm 1.0.26, open_clip 3.3.0) for export and orchestration.

---

## A. Edge node (Raspberry Pi 5)

| # | Problem | Cause | Adaptation | Verification |
|---|---|---|---|---|
| A1 | Edge dependencies fail to install | Original pins (torch 2.2.1, numpy 1.26.4, pydantic 2.6.4, pillow 10.2) have no Python 3.13 / aarch64 wheels; Raspberry Pi OS now ships Debian 13 (Python 3.13) | Two pin sets selected by PEP 508 markers (`python_version < "3.13"` keeps the original pins; ≥3.13 uses torch 2.6.0, torchvision 0.21.0, numpy 2.2.3, pydantic 2.10.6, pillow 11.1.0, fastapi 0.115.6, uvicorn 0.32.1) | Clean install; `/health` OK; checkpoint loads (`weights_only=False` explicit) |
| A2 | One physical Pi must act as several farms | Single systemd service per device | systemd template unit `plant-edge-farm@.service` + `setup_farms.sh`: N independent processes, each with its own `NODE_ID` and port (8101…), sharing the venv and checkpoint | 4 nodes report distinct `node_id` on `/health` |
| A3 | Latency could not be split into edge / network / cloud | Edge response only carried total wall time | Added `edge_ms` (preprocess + forward) to the edge response; cloud returns its own `latency_ms` and per-backbone timings; network = total − edge − cloud | Per-image breakdown in `events.csv` |
| A5 | Deployed edge would see different pixels than the evaluated model | Edge server resized with bicubic interpolation; training and offline evaluation used torchvision's default bilinear | Edge server switched to bilinear (Resize 256 → CenterCrop 224, as in training) | 300/300 test images: deployed label = offline label; margin \|Δ\| mean 8e-4, max 7.8e-3 |
| A7 | Escalations uploaded at full camera resolution (up to ~6 MB for a 12-MP photo) | Edge re-encoded the decoded image at its original size, although the cloud immediately resizes to ≤ 336 px | Escalation payload = shorter side 336 px (same size rule and bicubic filter as the cloud's first transform, so the cloud resize becomes a no-op), JPEG q90; configurable (`ESCALATION_SHORT_SIDE`, 0 = original); exposed on `/health` so the always-cloud baseline uses the identical payload | 300 test images, reference cloud: payload vs original → 95.7 % same prediction, accuracy 80.0 % vs 81.0 %; 42.7 KB vs 149 KB (full-res q90) |
| A8 | Failed escalations billed as uploaded bytes | Upload size recorded before the cloud call | `bandwidth_kb` set only after the cloud acknowledges; a failed call bills 0 | Smoke run with cloud offline: 0 KB uploaded |
| A6 | Edge loads weights with `strict=False` (could silently skip layers) | Inherited loader | Guarded by the A5 fidelity check (identical predictions ⇒ weights fully loaded) | 300/300 agreement |
| A4 | Cloud outages would masquerade as low escalation | On cloud failure the edge falls back and reports `routed=False` | Client logs the routing *decision* (`margin < τ`) separately from cloud success (`cloud_ok`) | Escalation rate and cloud failures reported separately |

## B. Cloud node (Jetson Nano 4 GB, JetPack 4)

| # | Problem | Cause | Adaptation | Verification |
|---|---|---|---|---|
| B1 | Reference cloud server cannot run | JetPack 4 = Python 3.6 + CUDA 10.2; the server needs Python ≥3.8 (PyTorch 2, timm ≥1.0.20 for DINOv3, open_clip, pydantic 2) | Decouple model from framework: export backbones to ONNX on the workstation, compile TensorRT engines on the Nano, serve with a torch-free Python 3.6 process | End-to-end parity checks (B9, B10) |
| B2 | TensorRT 8.2 cannot parse modern ONNX | TensorRT 8.2 ONNX parser supports opsets ≤ 13 | Export at opset 13 with the TorchScript-based exporter (`dynamo=False`) | onnxruntime vs PyTorch cosine 1.000000 |
| B3 | Fused attention not exportable at opset 13 | `scaled_dot_product_attention` symbolic requires opset ≥ 14 | During export, replace SDPA with explicit `softmax(QKᵀ·s)V` (mathematically identical) | cosine 1.000000 |
| B4 | CLIP export fails (`_native_multi_head_attention`) | `nn.MultiheadAttention` fast path emits a fused op with no ONNX mapping | `torch.backends.mha.set_fastpath_enabled(False)` during export | CLIP cosine 1.000000 |
| B5 | TensorRT rejects DINOv2 (`cubic` Resize) | DINOv2 bicubically interpolates its 37×37 position grid to the input grid on every forward; TensorRT 8.2 has no cubic Resize | Input size is fixed (336 px), so the interpolated embedding is computed once in PyTorch and baked in as a constant buffer | No Resize node remains; features unchanged |
| B6 | Dynamic shape arithmetic and conditionals in the graphs; onnx-simplifier itself fails on DINOv3 ("nodes not topologically sorted") | timm builds DINOv3 with `dynamic_img_size=True`, so its rotary embedding is recomputed from the input shape every forward (`If`, `Shape`/`Gather` chains); CLIP's reshapes emit `Mod` | (i) Bake DINOv3's rotary embedding for the fixed 21×21 patch grid as a constant buffer; (ii) constant-fold the rest with onnx-simplifier; (iii) reject the export if any TensorRT-8.2-unsafe op (`If`, `Loop`, `Mod`, `NonZero`, cubic `Resize`) survives | No unsafe ops in any graph (1197 / 1629 / 1399 nodes); cosine vs PyTorch 1.000000 |
| B7 | PyTorch/torchvision unavailable for preprocessing and head | Python 3.6 serving process is torch-free by design | NumPy/Pillow re-implementation of Resize(bicubic)+CenterCrop+Normalize and of the QualityGate head | Preprocessing bit-identical (max diff 0); head logits within 2.9e-6 |
| B8 | Engine build OOM-killed (kernel log: python, 1.75 GB anon-rss) | 4 GB shared by CPU and GPU; reading the 1.2 GB ONNX into Python and handing it to the parser holds the weights twice; the stock 1.9 GB swap is zram (compressed RAM) and weights barely compress | (i) one engine per subprocess (memory released between builds); (ii) `OnnxParser.parse_from_file` instead of `parse(bytes)`; (iii) builder workspace 256 MB, fp16; (iv) temporary 4 GB swap file on the SD card for the build only (`swapon`, not in fstab; removed afterwards) | OOM kill gone, but builder then failed with `std::bad_alloc` (GPU memory is carved from RAM and cannot swap) → B17 |
| B18 | Whole fp16 ViT-L still unbuildable: swap thrashing (5.6 GB swapped, GPU load 0, CPU waiting on I/O) | TensorRT 8.2's ONNX parser alone peaks at 2.75 GB RSS for a 607 MB fp16 graph (measured), before the builder starts; loading cuDNN + cuBLASLt for tactic timing adds ~1 GB | (i) `jetson_deploy/cloud/export_chunks.py`: each backbone exported as 4 consecutive chunks of 6 transformer blocks (stem in chunk 0, pooling/projection in chunk 3), each simplified and converted to fp16 (~150 MB), built as its own engine and chained at run time (`TrtChain`); (ii) tactic sources restricted to cuBLAS (`config.set_tactic_sources`); (iii) build script resumable (skips existing engines) | Chain vs unsplit model: cosine 1.000000 in PyTorch and in onnxruntime |
| B19 | Each 6-block chunk compiled into one Myelin `ForeignNode` holding 969 MB device memory (612 MB engine file; ~13 B/param) | TensorRT 8.2 fuses MatMul-with-constant-weight transformer subgraphs into Myelin, which stores several weight copies | Every `nn.Linear` exported as an equivalent 1×1 `Conv2d` (`ConvLinear`); CLIP's `nn.MultiheadAttention` (applies `in_proj_weight` via `F.linear` internally) replaced by an explicit equivalent module (`ExplicitMHA`) | Chain vs unsplit model cosine 1.000000 (torch + onnxruntime) for all 3 backbones; engine 157 MB, 0 ForeignNodes, 635 native layers |
| B20 | Chunk build ~10 min each (12 chunks) | Kernel timing repeated for identical layer shapes | Shared TensorRT timing cache (`timing.cache`) across builds | Later chunks of a backbone reuse timings |
| B21 | 12 chunk engines cannot be co-resident: measured max 9 even with shared scratch (each loaded engine ≈ 157 MB device weights + host-side copy + scratch ≈ 350 MB) | 4 GB shared CPU/GPU memory; TensorRT 8.2 keeps a host-side copy per engine | (i) one activation scratch buffer shared by all execution contexts (`create_execution_context_without_device_memory`, pointer passed as PyCapsule — TRT 8.2 Python binding); (ii) **batched, backbone-major cloud mode**: edge escalations are queued (`POST /enqueue`, edge `ESCALATION_MODE=deferred`), the cloud processes the queue one backbone at a time (`POST /flush`), freeing each backbone before loading the next (`TrtModel.close()`); flushes after the 12:00 and 18:00 captures in the replay | One chunk engine resident at a time (`run_backbone_batched`: load chunk, run every queued image, free, next chunk); peak RAM and board power during a flush in `results/nano_flush_profile/`. Engines per backbone total 0.59 GB (DINOv2, fp16), 1.16 GB (DINOv3, fp32), 1.19 GB (CLIP, fp32) — `results/trt_engine_sizes.json` |
| B22 | Parity gate failed: DINOv3 features NaN, CLIP min cosine 0.947, top-1 agreement 1/24 | First hypothesis (LayerNorm run in fp16) was **wrong**: pinning LayerNorm (`fp16-ln32`), then every non-conv layer (`conv16`), to fp32 changed nothing. Profiling (fp32, desktop): DINOv3 residual stream max 158,174 and linear outputs 52,023 vs fp16 max 65,504 (DINOv2 401/120, CLIP 219/129). CLIP's drift (5/24 images < 0.99) was attributed to its fp16 convolutions on Maxwell (superseded by B23: not precision; re-check 2026-10-12: fp16 CLIP engines match fp32 to cosine >= 0.9998, 1158 vs 1613 ms/img, `jetson_deploy/cloud/rebuild_clip16_check.sh`, `results/clip_fp16_check.json`) | Per-backbone precision: DINOv2 fp16; DINOv3 and CLIP `--precision all32` (FP16 flag kept on because the fp16-weight graph contains fp16-typed casts, every float layer pinned to fp32; a plain fp32 build is rejected: "fp16 precision has been set for a layer … but fp16 is not configured") | DINOv3 fp32: min 0.9874 / mean 0.9991, no NaN; full-pipeline agreement 23/24 |
| B23 | CLIP deviation (min cosine 0.947, 5/24 < 0.99) identical in conv16 and all-fp32 | **Not precision**: Pillow 8.4.0 on the Nano (last release for Python 3.6) vs 12.2.0 on the workstation decode/resample the same JPEG differently (preprocessed tensors differ by up to 0.59 normalised units; 64-px thumbnails up to 6/255) | None possible on JetPack 4 (no newer Pillow for Python 3.6); decomposed and documented: engines fed workstation tensors → cosine DINOv3 1.00000, CLIP 0.99999 (min), DINOv2 0.9965 (fp16), top-1 24/24; complete pipeline 23/24; fleet-level agreement on every escalated image reported in the paper | `results/trt_verify_engine_only.json`, `results/trt_verify_report.json` |
| B17 | Builder `std::bad_alloc` on fp32 ViT-L graphs | 1.2 GB of fp32 weights plus their GPU copies exceed what the Nano can allocate non-swappably | Convert graphs to fp16 weights on the workstation (`jetson_deploy/cloud/onnx_to_fp16.py`, onnxconverter-common), halving them to ~607 MB; LayerNorm arithmetic (`Pow`, `ReduceMean`, `Sqrt`, `Div`) kept in fp32 because ViT activation outliers overflow fp16 when squared | fp16 graph vs fp32 reference: cosine ≥ 0.9999, all outputs finite (onnxruntime, 3 reference images per backbone) |
| B9 | Port fidelity | fp16 + kernel substitutions can drift | 24 reference images: backbone features (cosine) and final top-1 compared against workstation references | `results/trt_verify_report.json`, `results/trt_verify_engine_only.json` (see B23) |
| B10 | Deploy pipeline ≠ paper pipeline? | Separate code paths (deploy server vs offline scripts) | Re-assembled pipeline compared with the paper's offline cloud predictions | 299/300 agree (99.7 %) |
| B11 | TensorRT Python bindings not on PyPI | Shipped by JetPack in system `dist-packages` | venv created with `--system-site-packages` | `import tensorrt` → 8.2.1.8 |
| B12 | Python 3.6 package versions | Most current packages dropped 3.6 | Pinned last 3.6-compatible releases: numpy 1.19.5, Pillow 8.4.0, pycuda 2021.1, fastapi 0.83.0, pydantic 1.9.2, uvicorn 0.16.0, pip < 22 | Clean install |
| B13 | pycuda install compiles numpy from source (~20 min) | pip build isolation pulls a source numpy as a build dependency | Install numpy wheel first, then `pip install --no-build-isolation pycuda` | Install time reduced to pycuda compile only |
| B14 | `Illegal instruction (core dumped)` on `import numpy` | numpy 1.19.5's bundled OpenBLAS mis-detects the Cortex-A57 | `OPENBLAS_CORETYPE=ARMV8` in setup script and systemd unit | Imports succeed |
| B15 | Shared wire protocol module unimportable | `shared/protocol.py` uses `from __future__ import annotations` (Python ≥3.7) | Cloud TRT server mirrors the two message models in pydantic v1; same JSON on the wire, so edge nodes are unchanged | Edge ↔ cloud round trip |
| B16 | CUDA context in a threaded web server | FastAPI runs sync endpoints in a thread pool; a pycuda context is thread-bound | One shared context, pushed/popped around every inference under a lock | Concurrent-safe inference |

## C. Workstation / orchestration

| # | Problem | Cause | Adaptation | Verification |
|---|---|---|---|---|
| C1 | Scripts fail on devices (`set: pipefail: invalid option`) | Windows tools wrote CRLF line endings | `.gitattributes` forces LF for `*.sh`/`*.service`; files normalised to LF before shipping | `file` reports no CRLF |
| C2 | Private deploy repository on devices | Devices hold no GitHub credentials | Code shipped over SSH (tar stream) from the workstation; multi-GB model files over `scp` (tar-over-SSH streams of >1 GB files were rejected by the receiver); checksums compared | SHA-256 match (edge checkpoint, ONNX graphs) |
| C5 | Deployment threshold differs from the reported operating point | Headline point = earliest escalation reaching the target, read off the *test* sweep (τ slightly below 0.20); the deploy default τ = 0.20 gives 23.6 % escalation / 76.94 % offline | τ calibrated on the validation split: the smallest τ whose validation accuracy reaches 95 % of the cloud's (τ = 0.138424, `results/threshold_calibration.json`), deployed on every farm node | Replay escalation 18.6 % = offline prediction at the same τ |
| C3 | Desktop cloud server pin too old | `timm==0.9.16` predates DINOv3 support | To update: timm ≥ 1.0.20, open_clip ≥ 3 (training env: timm 1.0.26, open_clip 3.3.0) | (pending) |
| C4 | Test images must match the paper's split | Batch client expects `<class>/<image>` folders | Exact 3,677-image test split materialised as hard links from the PlantEdge loader (seed-fixed MVPDR protocol) | 89 class folders, names match the model's class list |
| C6 | Second ("improved") configuration had to be deployed without losing the first | Distilled edge model (`outputs/edge_model_v3l_kd/best.pt`, SHA-256 `36e0f03d…ca9bd`) and fp16 CLIP engines replace the replay-1 components | Edge: checkpoint copied to `checkpoints/edge_v3l_kd.pt`; `config/config_kd.yaml` (copy of `config.yaml` with that `ckpt_path`) selected by `DEPLOY_CONFIG` in `config/.env`; `MARGIN_THRESHOLD=0.071893` (validation-calibrated by the same rule, `experiments/scripts/calibrate_threshold.py` → `results/threshold_calibration_kd.json`); original `.env` kept as `config/.env.orig-v3l`. Cloud: systemd drop-in `plant-cloud-trt.service.d/clip-fp16.conf` sets `TRT_DIR=/opt/plant-deploy/trt_clip16` (fp16 CLIP engines, every other file linked to `/opt/plant-deploy/trt`). Revert: restore `.env.orig-v3l` / delete the drop-in, restart the services | Edge node reproduces the distilled model's offline predictions on 300/300 test images (margin diff mean 6×10⁻⁴, `results/edge_fidelity_kd.json`); cloud `/health` OK from `trt_clip16`; CLIP fp16 parity in B22 note |
| C7 | Calibration rule had to be re-run for a new edge model | `results/threshold_calibration.json` was written by an earlier inline script | `experiments/scripts/calibrate_threshold.py` (per-image CSVs from `predict_edge_val.py` / `predict_edge.py`, fp32) | Reproduces the deployed τ: 0.138339 vs 0.138424 (same gap between validation margins, same 19.34 % validation escalation) |

## Manuscript numbering

Appendix A of the manuscript is a curated subset of this log, numbered in
reading order. Map from manuscript item to log item:

| Manuscript | Log | Manuscript | Log | Manuscript | Log |
|---|---|---|---|---|---|
| A1–A5 | A1–A5 | B9 | B17 | B16 | B11 |
| A6 | A7 | B10 | B18 | B17 | B12 |
| A7 | A8 | B11 | B19 | B18 | B14 |
| B1–B8 | B1–B8 | B12 | B20 | B19 | B13 |
| C1, C2 | C1, C2 | B13 | B21 | B20 | B15 |
| C3 | C4 | B14 | B22 | B21 | B16 |
| C4 | C6 | | | | |
| | | B15 | B23 | | |

Not in the manuscript: A6 (guarded by A5), B9–B10 (verification, reported in
the paper's fidelity table), C3 (desktop server pin), C5 (threshold
calibration, described in the paper's Methods), C7 (calibration script).
