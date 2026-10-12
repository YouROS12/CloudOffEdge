# jetson_deploy — the cloud tier (Jetson Nano 4 GB)

Serves the three-backbone foundation ensemble (DINOv2-L, DINOv3-L, CLIP-L + QualityGate
head, 0.9 B parameters) on a **4 GB Jetson Nano with JetPack 4.6.1** (Ubuntu 18.04,
Python 3.6, CUDA 10.2, TensorRT 8.2.1) — a software stack that cannot run PyTorch 2,
`timm` or `open_clip`. Section 5 of the paper describes the port; every adaptation is
logged in [`docs/ADAPTATIONS.md`](../docs/ADAPTATIONS.md) (paper, Appendix A).

```
cloud/server_trt.py        Nano service (Python 3.6, no PyTorch): POST /enqueue, /flush, /classify
cloud/trt_runtime.py       TensorRT engines, chunk chain, chunk-major batched inference
cloud/trt_common.py        NumPy preprocessing + QualityGate head (bit-identical to torchvision)
cloud/build_trt.py         builds the chunk engines on the Nano and gates them (parity check)
cloud/setup_jetson_trt.sh  venv (system TensorRT visible) + engines + systemd service
cloud/rebuild_clip16_check.sh  re-check of CLIP in fp16, in a separate engine directory
cloud/export_onnx.py       [workstation] ONNX export + reference features + pipeline check
cloud/export_chunks.py     [workstation] each backbone as 4 chunks of 6 blocks, 1x1-conv form
cloud/onnx_to_fp16.py      [workstation] fp16 weights, LayerNorm arithmetic kept in fp32
cloud/server.py            reference PyTorch cloud server (workstation; used by the exporters)
cloud/check_payload_fidelity.py  payload-size policy (paper, Table 2)
shared/                    same package as rasp_deploy/shared (protocol, config)
systemd/plant-cloud-trt.service  the Nano service (CLOUD_MODE=batch, OPENBLAS_CORETYPE=ARMV8)
```

## 1. Export the engines' sources (workstation, PyTorch 2)

```bash
pip install -r experiments/requirements.txt
python jetson_deploy/cloud/export_onnx.py --image-dir data/plantwild_test \
    --offline-preds artifacts/cloud_predictions_test.csv --out build/trt_export
python jetson_deploy/cloud/export_chunks.py --out build/trt_export --chunks 4 --image-dir data/plantwild_test
python jetson_deploy/cloud/onnx_to_fp16.py --dir build/trt_export
```

The exporters load the backbones through `timm`/`open_clip` (weights are downloaded
from their hubs) and the trained head `artifacts/cloud_head/qualitygate_fusion_3L.pt`.
Every stage is checked against PyTorch (cosine 1.000000 for the graphs and the chunk
chain, ≥ 0.9999 for the fp16-weight graphs). Copy `build/trt_export/` to the Nano as
`/opt/plant-deploy/trt/`.

## 2. Build and verify on the Nano

```bash
# on the Nano, with jetson_deploy/ copied there
SKIP_ENGINES=1 bash jetson_deploy/cloud/setup_jetson_trt.sh     # venv + service only
PY=/opt/plant-deploy/.venv-trt/bin/python
export OPENBLAS_CORETYPE=ARMV8
for i in 0 1 2 3; do
  $PY /opt/plant-deploy/deploy/cloud/build_trt.py --trt-dir /opt/plant-deploy/trt --only dinov2_vitl14.c$i --precision fp16
  $PY /opt/plant-deploy/deploy/cloud/build_trt.py --trt-dir /opt/plant-deploy/trt --only dinov3_vitl16.c$i --precision all32
  $PY /opt/plant-deploy/deploy/cloud/build_trt.py --trt-dir /opt/plant-deploy/trt --only clip_vitl14.c$i   --precision all32
done
$PY /opt/plant-deploy/deploy/cloud/build_trt.py --trt-dir /opt/plant-deploy/trt --verify-only
sudo systemctl restart plant-cloud-trt && curl -s localhost:8001/health
```

Precision per backbone follows the activation ranges (paper, Fig. 4): DINOv3's
residual stream reaches 1.6×10⁵, above the fp16 maximum, so it is built in fp32
(`all32`: FP16 flag on because the graph has fp16-typed weights, every layer pinned
to fp32). CLIP was built in fp32 for replay 1; the later check
(`cloud/rebuild_clip16_check.sh`, `results/clip_fp16_check.json`) showed fp16 CLIP
equivalent (cosine ≥ 0.9998 to fp32) and 28 % faster, and replay 2 used it. Engines are
built one per process with a shared timing cache; a temporary swap file helps the
first builds (Appendix A, B8–B12).

## 3. Run

`CLOUD_MODE=batch` (the default on the Nano): escalations are queued with
`POST /enqueue`; `POST /flush` processes the queue chunk-major — one of the 12 chunk
engines resident at a time — and returns the ensemble's diagnoses. In the replay the
fleet emulator calls `/flush` after the 12:00 and 18:00 captures. `CLOUD_MODE=resident`
keeps every engine loaded and answers `POST /classify` synchronously; it is meant for
nodes with more memory and was not tested.

Replay 2 used the fp16-CLIP engine set without touching replay 1's engines: a systemd
drop-in sets `TRT_DIR=/opt/plant-deploy/trt_clip16` (docs/ADAPTATIONS.md, item C6):

```ini
# /etc/systemd/system/plant-cloud-trt.service.d/clip-fp16.conf
[Service]
Environment=TRT_DIR=/opt/plant-deploy/trt_clip16
```

Measured on the Nano: 6.9 s (replay 1) / 6.5 s (replay 2) of compute per escalated
image; memory peaks at 2.32 GB (fp32 CLIP) / 2.10 GB (fp16 CLIP) of the 3.86 GB shared
by CPU and GPU; 49 J / 48 J per escalated image (`results/nano_flush_profile/`,
`results/nano_flush_profile_clip16/`).
