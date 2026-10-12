#!/usr/bin/env bash
# Paper 2 — Jetson Nano (JetPack 4.6, Python 3.6) cloud setup, TensorRT path.
#
# JetPack 4 cannot run the torch/timm/open_clip cloud server (Python 3.6,
# no modern torch). Instead:
#   desktop : python jetson_deploy/cloud/export_onnx.py ...   -> *.onnx + head .npz
#   Nano    : this script -> venv (system TensorRT visible) + engines + service
#
# Expects the export directory (onnx files, qualitygate_head.npz, ref_*.npy,
# ref_images.json) at ${TRT_DIR} (default /opt/plant-deploy/trt).
#
#   bash jetson_deploy/cloud/setup_jetson_trt.sh            # venv + engines + service
#   SKIP_ENGINES=1 bash jetson_deploy/cloud/setup_jetson_trt.sh
set -euo pipefail

INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/plant-deploy}"
VENV="${INSTALL_PREFIX}/.venv-trt"
TRT_DIR="${TRT_DIR:-${INSTALL_PREFIX}/trt}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="/usr/local/cuda/bin:${PATH}"
# numpy 1.19.5's bundled OpenBLAS mis-detects the Nano's Cortex-A57 and
# dies with "Illegal instruction" unless the core type is forced.
export OPENBLAS_CORETYPE=ARMV8

echo "[trt] system packages ..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev python3-pip \
  libjpeg-dev zlib1g-dev build-essential rsync >/dev/null

sudo mkdir -p "${INSTALL_PREFIX}" "${TRT_DIR}"
sudo chown "${USER}:${USER}" "${INSTALL_PREFIX}" "${TRT_DIR}"
mkdir -p "${INSTALL_PREFIX}/deploy"
if [[ "$(realpath "${HERE}/..")" != "$(realpath "${INSTALL_PREFIX}/deploy")" ]]; then
  rsync -a --exclude '.venv*' --exclude '__pycache__' --exclude 'trt' "${HERE}/../" "${INSTALL_PREFIX}/deploy/"
fi

# --system-site-packages: TensorRT's Python bindings live in the system
# python3.6 dist-packages (from JetPack), not on PyPI.
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv --system-site-packages "${VENV}"
fi
"${VENV}/bin/pip" install -q --upgrade "pip<22" "setuptools<60" wheel
# pycuda last, without build isolation: otherwise pip compiles a throwaway
# numpy from source (~20 min on the Nano) just to build pycuda.
REQ="${INSTALL_PREFIX}/deploy/cloud/requirements-jetson-trt.txt"
"${VENV}/bin/pip" install -q $(grep -vE '^(#|pycuda|$)' "${REQ}")
"${VENV}/bin/pip" install -q --no-build-isolation $(grep '^pycuda' "${REQ}")
"${VENV}/bin/python" -c "import tensorrt, pycuda.driver, numpy, PIL, fastapi; \
print('[trt] env ok: tensorrt', tensorrt.__version__, '| numpy', numpy.__version__)"

if [[ "${SKIP_ENGINES:-0}" != "1" ]]; then
  "${VENV}/bin/python" "${INSTALL_PREFIX}/deploy/cloud/build_trt.py" --trt-dir "${TRT_DIR}"
fi

# Systemd service, once every backbone has an engine (whole, or chunked c0..c3).
have_all=1
for b in dinov2_vitl14 dinov3_vitl16 clip_vitl14; do
  [[ -f "${TRT_DIR}/${b}.engine" || -f "${TRT_DIR}/${b}.c0.engine" ]] || have_all=0
done
if [[ "${have_all}" == "1" ]]; then
  sudo sed "s/__USER__/${USER}/" "${INSTALL_PREFIX}/deploy/systemd/plant-cloud-trt.service" \
    | sudo tee /etc/systemd/system/plant-cloud-trt.service >/dev/null
  sudo touch /var/log/plant-cloud-trt.log
  sudo chown "${USER}:${USER}" /var/log/plant-cloud-trt.log
  sudo systemctl daemon-reload
  sudo systemctl enable plant-cloud-trt >/dev/null 2>&1
  sudo systemctl restart plant-cloud-trt
  echo "[trt] plant-cloud-trt started on :8001 (engines load in ~1 min); log: /var/log/plant-cloud-trt.log"
else
  echo "[trt] engines missing in ${TRT_DIR} -- service not installed yet"
fi
