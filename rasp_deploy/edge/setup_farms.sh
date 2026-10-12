#!/usr/bin/env bash
# Paper 2 — run N virtual farm edge nodes on ONE physical Raspberry Pi.
#
# Each farm is an independent edge server process (own NODE_ID, own port,
# own routing decisions, own connection to the cloud), so the cloud sees N
# distinct edge clients. They share one physical CPU: report them as
# "N virtual edge nodes emulated on one Raspberry Pi", never as N devices.
#
# Requires edge/setup.sh to have run first (venv + checkpoints + .env).
#
#   bash rasp_deploy/edge/setup_farms.sh            # 4 farms: A..D on ports 8101..8104
#   N_FARMS=3 bash rasp_deploy/edge/setup_farms.sh  # 3 farms
#   bash rasp_deploy/edge/setup_farms.sh --down     # stop farms, restore plant-edge
set -euo pipefail

INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/plant-deploy}"
CONF_DIR="${INSTALL_PREFIX}/deploy/config"
UNIT_SRC="${INSTALL_PREFIX}/deploy/systemd/plant-edge-farm@.service"
UNIT_DST="/etc/systemd/system/plant-edge-farm@.service"
N_FARMS="${N_FARMS:-4}"
BASE_PORT="${BASE_PORT:-8101}"
LETTERS=(A B C D E F G H)

if [[ "${1:-}" == "--down" ]]; then
  for f in "${LETTERS[@]}"; do
    sudo systemctl disable --now "plant-edge-farm@${f}" 2>/dev/null || true
  done
  sudo systemctl enable --now plant-edge
  echo "[farms] stopped all farms; plant-edge restored on :8000"
  exit 0
fi

if [[ ! -f "${CONF_DIR}/.env" ]]; then
  echo "ERROR: ${CONF_DIR}/.env missing -- run rasp_deploy/install_pi.sh first." >&2
  exit 1
fi

# The farms replace the single edge server while the experiment runs.
sudo systemctl disable --now plant-edge 2>/dev/null || true

sudo sed "s/__USER__/${USER}/" "${UNIT_SRC}" | sudo tee "${UNIT_DST}" >/dev/null
sudo systemctl daemon-reload

for ((i = 0; i < N_FARMS; i++)); do
  f="${LETTERS[$i]}"
  port=$((BASE_PORT + i))
  printf 'NODE_ID=farm-%s\nSERVER_PORT=%d\n' "${f}" "${port}" > "${CONF_DIR}/farm-${f}.env"
  sudo touch "/var/log/plant-edge-farm-${f}.log"
  sudo chown "${USER}:${USER}" "/var/log/plant-edge-farm-${f}.log"
  sudo systemctl enable "plant-edge-farm@${f}" >/dev/null 2>&1
  sudo systemctl restart "plant-edge-farm@${f}"
  echo "[farms] farm-${f} -> :${port}"
done

# Wait for every farm to answer /health (model load takes a few seconds).
for ((i = 0; i < N_FARMS; i++)); do
  port=$((BASE_PORT + i))
  for _ in $(seq 1 60); do
    curl -sf "http://localhost:${port}/health" >/dev/null && break
    sleep 1
  done
  curl -s "http://localhost:${port}/health"; echo
done
