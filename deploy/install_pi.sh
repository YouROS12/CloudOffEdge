#!/usr/bin/env bash
# CloudOffEdge -- one-shot Raspberry Pi (edge) installer.
#
# Assumes a fresh Raspberry Pi OS (64-bit) with Python already present and
# nothing else project-specific. Installs system deps, a virtualenv with
# CPU torch, the bundled edge checkpoint, and a systemd service that runs the
# edge server on boot -- then starts it.
#
# ---------------------------------------------------------------------------
# USAGE
#   You must already have the repo on the Pi. If you don't:
#
#     sudo apt-get update && sudo apt-get install -y git
#     git clone https://github.com/YouROS12/CloudOffEdge.git ~/CloudOffEdge
#     cd ~/CloudOffEdge
#
#   Then run this script. Point it at your Jetson in one line:
#
#     CLOUD_URL=http://<jetson-tailscale-name>:8001 bash deploy/install_pi.sh
#
#   Optional environment variables:
#     CLOUD_URL          Jetson cloud endpoint written into .env (recommended)
#     MARGIN_THRESHOLD   routing threshold (default 0.20)
#     INSTALL_TAILSCALE  set to 1 to also install + bring up Tailscale
#
# Idempotent: safe to re-run after editing config or pulling new code.
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../deploy
REPO_ROOT="$(cd "${HERE}/.." && pwd)"          # repo checkout root

CLOUD_URL="${CLOUD_URL:-}"
MARGIN_THRESHOLD="${MARGIN_THRESHOLD:-0.20}"
INSTALL_TAILSCALE="${INSTALL_TAILSCALE:-0}"

echo "=============================================================="
echo " CloudOffEdge edge installer (Raspberry Pi)"
echo "   repo        : ${REPO_ROOT}"
echo "   cloud_url   : ${CLOUD_URL:-<unset -- edit .env later>}"
echo "   margin      : ${MARGIN_THRESHOLD}"
echo "   tailscale   : $([[ "${INSTALL_TAILSCALE}" == "1" ]] && echo install || echo skip)"
echo "=============================================================="

# ---------------------------------------------------------------------------
# 0) Sanity: Python present, and this really is the repo.
# ---------------------------------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi
echo "[0/4] python3: $(python3 --version)"
if [[ ! -f "${HERE}/edge/setup.sh" ]]; then
  echo "ERROR: ${HERE}/edge/setup.sh missing -- run this from inside the repo." >&2
  echo "       git clone https://github.com/YouROS12/CloudOffEdge.git ~/CloudOffEdge" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1) System packages (git/curl + build libs the Python wheels need).
# ---------------------------------------------------------------------------
echo "[1/4] installing system packages (needs sudo) ..."
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates \
  python3 python3-venv python3-pip \
  libjpeg-dev zlib1g-dev libtiff-dev libopenblas-dev

# ---------------------------------------------------------------------------
# 2) Tailscale (optional).
# ---------------------------------------------------------------------------
if [[ "${INSTALL_TAILSCALE}" == "1" ]]; then
  if ! command -v tailscale >/dev/null 2>&1; then
    echo "[2/4] installing Tailscale ..."
    curl -fsSL https://tailscale.com/install.sh | sh
  fi
  echo "[2/4] bringing Tailscale up (may open a login URL) ..."
  sudo tailscale up || true
else
  echo "[2/4] skipping Tailscale (set INSTALL_TAILSCALE=1 to enable)."
fi

# ---------------------------------------------------------------------------
# 3) Delegate venv + torch + checkpoints + systemd to the edge setup.
# ---------------------------------------------------------------------------
echo "[3/4] running edge setup (venv, torch, checkpoints, systemd) ..."
bash "${HERE}/edge/setup.sh"

# ---------------------------------------------------------------------------
# 4) Config: point the edge at the Jetson + set routing threshold.
# ---------------------------------------------------------------------------
ENVF="/opt/plant-deploy/deploy/config/.env"
if [[ -f "${ENVF}" ]]; then
  if [[ -n "${CLOUD_URL}" ]]; then
    if grep -q '^CLOUD_URL=' "${ENVF}"; then
      sudo sed -i "s|^CLOUD_URL=.*|CLOUD_URL=${CLOUD_URL}|" "${ENVF}"
    else
      echo "CLOUD_URL=${CLOUD_URL}" | sudo tee -a "${ENVF}" >/dev/null
    fi
    echo "[4/4] set CLOUD_URL=${CLOUD_URL}"
  else
    echo "[4/4] CLOUD_URL not provided -- edit ${ENVF} then restart."
  fi
  if grep -q '^MARGIN_THRESHOLD=' "${ENVF}"; then
    sudo sed -i "s|^MARGIN_THRESHOLD=.*|MARGIN_THRESHOLD=${MARGIN_THRESHOLD}|" "${ENVF}"
  else
    echo "MARGIN_THRESHOLD=${MARGIN_THRESHOLD}" | sudo tee -a "${ENVF}" >/dev/null
  fi
  sudo systemctl restart plant-edge || true
fi

echo "=============================================================="
echo " Done. The edge server is installed and running."
echo "   health : curl http://localhost:8000/health"
echo "   logs   : journalctl -u plant-edge -f"
echo "   config : ${ENVF}"
if [[ -z "${CLOUD_URL}" ]]; then
  echo ""
  echo " NEXT: set CLOUD_URL in the config above to your Jetson, then:"
  echo "       sudo systemctl restart plant-edge"
fi
echo "=============================================================="
