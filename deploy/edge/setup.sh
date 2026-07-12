#!/usr/bin/env bash
# Paper 2 — Raspberry Pi edge setup.
#
# Installs system deps, creates a virtualenv, installs Python deps, fetches
# the edge model checkpoint + class names, populates .env, and installs a
# systemd unit that runs the edge server on boot.
#
# Idempotent: safe to re-run after editing config.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_ROOT="$(cd "${HERE}/.." && pwd)"
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/plant-deploy}"
VENV="${INSTALL_PREFIX}/.venv"
PY="${PY:-python3}"

echo "[edge] install_prefix=${INSTALL_PREFIX}"
echo "[edge] deploy_root=${DEPLOY_ROOT}"

# ---------------------------------------------------------------------------
# 1) System packages
# ---------------------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  libjpeg-dev zlib1g-dev libtiff-dev libopenblas-dev \
  curl ca-certificates

# ---------------------------------------------------------------------------
# 2) Stage repo at INSTALL_PREFIX (so systemd has a stable path)
# ---------------------------------------------------------------------------
sudo mkdir -p "${INSTALL_PREFIX}"
sudo chown "${USER}:${USER}" "${INSTALL_PREFIX}"
if [[ "$(realpath "${DEPLOY_ROOT}/..")" != "$(realpath "${INSTALL_PREFIX}")" ]]; then
  echo "[edge] syncing repo -> ${INSTALL_PREFIX}"
  rsync -a --delete \
    --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    "${DEPLOY_ROOT}/../" "${INSTALL_PREFIX}/"
fi

# ---------------------------------------------------------------------------
# 3) Virtualenv + Python deps
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV}" ]]; then
  ${PY} -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --upgrade pip wheel
"${VENV}/bin/pip" install -r "${INSTALL_PREFIX}/deploy/edge/requirements.txt"

# ---------------------------------------------------------------------------
# 4) Checkpoints
#
# Two ways to provide the edge model + class names:
#   (a) Pre-place them at ${INSTALL_PREFIX}/deploy/checkpoints/{edge_v3l.pt,class_names.txt}
#   (b) Set CHECKPOINTS_URL to a public URL serving a tarball.
# ---------------------------------------------------------------------------
CKPT_DIR="${INSTALL_PREFIX}/deploy/checkpoints"
mkdir -p "${CKPT_DIR}"

if [[ ! -f "${CKPT_DIR}/edge_v3l.pt" ]]; then
  if [[ -n "${CHECKPOINTS_URL:-}" ]]; then
    echo "[edge] downloading checkpoints from ${CHECKPOINTS_URL}"
    curl -fL -o /tmp/ckpts.tgz "${CHECKPOINTS_URL}"
    tar -xzf /tmp/ckpts.tgz -C "${CKPT_DIR}"
  else
    echo "[edge] WARNING: no edge_v3l.pt found and CHECKPOINTS_URL not set."
    echo "       Place edge_v3l.pt and class_names.txt under ${CKPT_DIR}/"
    echo "       then re-run this script (or just \`systemctl start plant-edge\`)."
  fi
fi

# ---------------------------------------------------------------------------
# 5) Config
# ---------------------------------------------------------------------------
CONF="${INSTALL_PREFIX}/deploy/config/config.yaml"
ENVF="${INSTALL_PREFIX}/deploy/config/.env"
if [[ ! -f "${CONF}" ]]; then
  cp "${INSTALL_PREFIX}/deploy/config/config.example.yaml" "${CONF}"
  echo "[edge] created ${CONF} from example. Edit cloud_url before starting."
fi
if [[ ! -f "${ENVF}" ]]; then
  cp "${INSTALL_PREFIX}/deploy/config/.env.example" "${ENVF}"
  # Ensure role + port are set for the edge.
  sed -i 's/^DEPLOY_ROLE=.*/DEPLOY_ROLE=edge/'  "${ENVF}" || true
  sed -i 's/^SERVER_PORT=.*/SERVER_PORT=8000/'  "${ENVF}" || true
  echo "[edge] created ${ENVF}. Edit CLOUD_URL to your Jetson Tailscale name."
fi

# ---------------------------------------------------------------------------
# 6) Systemd unit
# ---------------------------------------------------------------------------
UNIT_SRC="${INSTALL_PREFIX}/deploy/systemd/plant-edge.service"
UNIT_DST="/etc/systemd/system/plant-edge.service"
# Substitute %i (User=) with the current user; the template uses %i so it
# works under systemd's instance template, but we install a plain unit here.
sudo sed "s/User=%i/User=${USER}/" "${UNIT_SRC}" | sudo tee "${UNIT_DST}" >/dev/null
sudo touch /var/log/plant-edge.log
sudo chown "${USER}:${USER}" /var/log/plant-edge.log
sudo systemctl daemon-reload
sudo systemctl enable plant-edge
sudo systemctl restart plant-edge

echo "[edge] up. tail logs with: journalctl -u plant-edge -f"
