#!/usr/bin/env bash
# Paper 2 — Jetson cloud setup.
#
# Installs system deps, creates a virtualenv, installs Python deps (with
# Jetson-specific torch wheel handling), fetches the cloud head + class
# names, populates .env, and installs a systemd unit.
#
# Works on:
#   - Jetson Nano legacy 4GB (JetPack 4.6 / CUDA 10.2): uses NVIDIA's
#     PyTorch 1.10/1.11 wheel; we DO NOT pip-install torch from PyPI.
#   - Jetson Orin (JetPack 5.x/6.x): uses NVIDIA's torch 2.x wheel.
#   - Generic x86_64 host with CUDA: uses the cu121 PyPI wheel.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ROOT="$(cd "${HERE}/.." && pwd)"
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/plant-deploy}"
VENV="${INSTALL_PREFIX}/.venv"
PY="${PY:-python3}"

is_jetson() { [[ -f /etc/nv_tegra_release ]]; }

echo "[cloud] install_prefix=${INSTALL_PREFIX}"

# ---------------------------------------------------------------------------
# 1) System packages
# ---------------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip python3-dev \
  libjpeg-dev zlib1g-dev libopenblas-dev libomp-dev \
  curl ca-certificates rsync

if is_jetson; then
  # libs needed by NVIDIA's torch wheel (Jetson)
  sudo apt-get install -y \
    libopenmpi-dev libomp-dev libopenblas-dev libblas-dev libeigen3-dev \
    libcurand-dev || true
fi

# ---------------------------------------------------------------------------
# 2) Stage repo
# ---------------------------------------------------------------------------
sudo mkdir -p "${INSTALL_PREFIX}"
sudo chown "${USER}:${USER}" "${INSTALL_PREFIX}"
if [[ "$(realpath "${DEPLOY_ROOT}/..")" != "$(realpath "${INSTALL_PREFIX}")" ]]; then
  echo "[cloud] syncing repo -> ${INSTALL_PREFIX}"
  rsync -a --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    "${DEPLOY_ROOT}/../" "${INSTALL_PREFIX}/"
fi

# ---------------------------------------------------------------------------
# 3) Virtualenv
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV}" ]]; then
  ${PY} -m venv "${VENV}" --system-site-packages
fi
"${VENV}/bin/pip" install --upgrade pip wheel

# ---------------------------------------------------------------------------
# 4) Torch
#
# On Jetson we DO NOT install torch from PyPI — it has no working CUDA wheels
# for Tegra. Instead, NVIDIA publishes wheels at https://developer.download.nvidia.com.
#
# The exact wheel filename depends on JetPack version. We default to:
#   - JetPack 4.6.x (Nano 4GB legacy):  torch-1.11.0  + torchvision-0.12.0
#   - JetPack 5.x (Orin):               torch-2.1.0   + torchvision-0.16.1
#
# Override with TORCH_WHEEL_URL / TORCHVISION_WHEEL_URL env vars if needed.
# Nvidia's wheel repository for Jetson has moved a few times; check
# https://forums.developer.nvidia.com if a URL 404s.
# ---------------------------------------------------------------------------
if is_jetson; then
  if "${VENV}/bin/python" -c "import torch" 2>/dev/null; then
    echo "[cloud] torch already importable — skipping wheel install."
  else
    JETPACK="$(head -n1 /etc/nv_tegra_release | awk -F'R' '{print $2}' | awk '{print $1}')"
    echo "[cloud] detected Tegra release R${JETPACK}"
    case "${JETPACK}" in
      32*|"")
        TW_DEFAULT="https://nvidia.box.com/shared/static/ssf2v7pf5i245fk4i0q926hy4imzs2ph.whl"  # torch-1.11
        TV_DEFAULT=""  # torchvision must be built from source on JetPack 4
        echo "[cloud] WARNING: JetPack 4.x detected. torchvision needs to be built from source:"
        echo "        git clone --branch v0.12.0 https://github.com/pytorch/vision torchvision"
        echo "        cd torchvision && python setup.py install"
        echo "        Doing automated install of torch only."
        ;;
      35*)
        TW_DEFAULT="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
        TV_DEFAULT=""
        ;;
      36*)
        TW_DEFAULT="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl"
        TV_DEFAULT=""
        ;;
      *)
        TW_DEFAULT=""
        TV_DEFAULT=""
        ;;
    esac
    TW="${TORCH_WHEEL_URL:-${TW_DEFAULT}}"
    if [[ -z "${TW}" ]]; then
      echo "[cloud] ERROR: no torch wheel URL for JetPack R${JETPACK}." >&2
      echo "       Set TORCH_WHEEL_URL to the wheel for your JetPack version" >&2
      echo "       (see https://forums.developer.nvidia.com/t/pytorch-for-jetson)." >&2
      exit 1
    fi
    "${VENV}/bin/pip" install "${TW}"
    if [[ -n "${TORCHVISION_WHEEL_URL:-${TV_DEFAULT}}" ]]; then
      "${VENV}/bin/pip" install "${TORCHVISION_WHEEL_URL:-${TV_DEFAULT}}"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# 5) Remaining Python deps
# ---------------------------------------------------------------------------
# Filter out torch/torchvision lines on Jetson (already installed by step 4).
REQ="${INSTALL_PREFIX}/deploy/cloud/requirements.txt"
if is_jetson; then
  TMP_REQ="$(mktemp)"
  grep -vE '^(torch|torchvision)([=<>~!]|$)' "${REQ}" \
    | grep -v 'extra-index-url' \
    > "${TMP_REQ}"
  "${VENV}/bin/pip" install -r "${TMP_REQ}"
  rm -f "${TMP_REQ}"
else
  "${VENV}/bin/pip" install -r "${REQ}"
fi

# ---------------------------------------------------------------------------
# 6) Checkpoints
# ---------------------------------------------------------------------------
CKPT_DIR="${INSTALL_PREFIX}/deploy/checkpoints"
mkdir -p "${CKPT_DIR}"
if [[ ! -f "${CKPT_DIR}/qualitygate_fusion_3L.pt" ]]; then
  if [[ -n "${CHECKPOINTS_URL:-}" ]]; then
    echo "[cloud] downloading checkpoints from ${CHECKPOINTS_URL}"
    curl -fL -o /tmp/ckpts.tgz "${CHECKPOINTS_URL}"
    tar -xzf /tmp/ckpts.tgz -C "${CKPT_DIR}"
  else
    echo "[cloud] WARNING: missing qualitygate_fusion_3L.pt and CHECKPOINTS_URL unset."
    echo "        Place qualitygate_fusion_3L.pt and class_names.txt under ${CKPT_DIR}/"
  fi
fi

# ---------------------------------------------------------------------------
# 7) Config
# ---------------------------------------------------------------------------
CONF="${INSTALL_PREFIX}/deploy/config/config.yaml"
ENVF="${INSTALL_PREFIX}/deploy/config/.env"
if [[ ! -f "${CONF}" ]]; then
  cp "${INSTALL_PREFIX}/deploy/config/config.example.yaml" "${CONF}"
fi
if [[ ! -f "${ENVF}" ]]; then
  cp "${INSTALL_PREFIX}/deploy/config/.env.example" "${ENVF}"
fi
# Force cloud-side env values
{
  echo "DEPLOY_ROLE=cloud"
  echo "SERVER_HOST=0.0.0.0"
  echo "SERVER_PORT=8001"
  if is_jetson; then
    echo "CLOUD_DEVICE=cuda"
    echo "CLOUD_SEQUENTIAL=1"
  else
    echo "CLOUD_DEVICE=${CLOUD_DEVICE:-cuda}"
    echo "CLOUD_SEQUENTIAL=${CLOUD_SEQUENTIAL:-0}"
  fi
} > "${ENVF}"
echo "[cloud] wrote ${ENVF}"

# ---------------------------------------------------------------------------
# 8) Systemd unit
# ---------------------------------------------------------------------------
UNIT_SRC="${INSTALL_PREFIX}/deploy/systemd/plant-cloud.service"
UNIT_DST="/etc/systemd/system/plant-cloud.service"
sudo sed "s/User=%i/User=${USER}/" "${UNIT_SRC}" | sudo tee "${UNIT_DST}" >/dev/null
sudo touch /var/log/plant-cloud.log
sudo chown "${USER}:${USER}" /var/log/plant-cloud.log
sudo systemctl daemon-reload
sudo systemctl enable plant-cloud
sudo systemctl restart plant-cloud

echo "[cloud] up. tail logs with: journalctl -u plant-cloud -f"
