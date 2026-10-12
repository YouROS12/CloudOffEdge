#!/bin/bash
# Re-check CLIP in fp16 without touching the deployed engines (/opt/plant-deploy/trt).
# Builds the four CLIP chunk engines in fp16 into a separate directory that
# links every other file from the deployed one, runs the standard on-device
# verification there, and compares the fp16 CLIP features with the deployed
# fp32 CLIP features on the same 24 reference images.
set -euo pipefail
SRC=/opt/plant-deploy/trt
DST=/opt/plant-deploy/trt_clip16
PY=/opt/plant-deploy/.venv-trt/bin/python
export OPENBLAS_CORETYPE=ARMV8
cd /opt/plant-deploy/deploy

mkdir -p "$DST"
for f in "$SRC"/*; do
  b=$(basename "$f")
  case "$b" in clip_vitl14.c?.engine|verify_report.json|trt_*.npy|timing.cache) continue ;; esac
  ln -sfn "$f" "$DST/$b"
done
cp -n "$SRC/timing.cache" "$DST/timing.cache" || true
cp "$SRC/trt_clip_vitl14.npy" "$DST/clip_fp32_reference.npy"   # deployed fp32 CLIP engine outputs

sudo -n systemctl stop plant-cloud-trt   # free memory for the build; restarted below
trap 'sudo -n systemctl start plant-cloud-trt' EXIT
for i in 0 1 2 3; do
  echo "[build] clip_vitl14.c$i (fp16)"
  $PY cloud/build_trt.py --trt-dir "$DST" --only clip_vitl14.c$i --precision fp16 2>&1 | grep -E "built|Error|error|Killed" || true
done
ls -la "$DST"/clip_vitl14.c?.engine
echo "[verify]"
$PY cloud/build_trt.py --trt-dir "$DST" --verify-only 2>&1 | grep -vE "libprotobuf|TRT\] \[W\]"
echo "[fp16 vs fp32 CLIP engines, same inputs]"
$PY - "$DST" <<'EOF'
import sys, numpy as np, json
d = sys.argv[1]
a = np.load(d + "/trt_clip_vitl14.npy").astype(np.float64)
b = np.load(d + "/clip_fp32_reference.npy").astype(np.float64)
a /= np.linalg.norm(a, axis=-1, keepdims=True); b /= np.linalg.norm(b, axis=-1, keepdims=True)
c = (a * b).sum(-1)
print(json.dumps({"n": int(len(c)), "cos_min": float(c.min()), "cos_mean": float(c.mean())}))
EOF
echo "[done]"
