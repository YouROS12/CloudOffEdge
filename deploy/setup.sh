#!/usr/bin/env bash
# Paper 2 — top-level deploy entry point.
#
# Auto-detects whether this host is the edge (Raspberry Pi) or the cloud
# (Jetson) and dispatches to the appropriate setup script. You can also
# force the role with: ./setup.sh edge   or   ./setup.sh cloud
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROLE="${1:-}"

detect_role() {
  if [[ -f /etc/nv_tegra_release ]]; then
    echo "cloud"; return
  fi
  if [[ -f /sys/firmware/devicetree/base/model ]]; then
    if grep -qi "raspberry pi" /sys/firmware/devicetree/base/model; then
      echo "edge"; return
    fi
  fi
  if [[ "$(uname -m)" == "aarch64" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "cloud"; return
  fi
  echo "unknown"
}

if [[ -z "${ROLE}" ]]; then
  ROLE="$(detect_role)"
  if [[ "${ROLE}" == "unknown" ]]; then
    echo "ERROR: could not detect role." >&2
    echo "Run: $0 edge    (on Raspberry Pi)" >&2
    echo "Run: $0 cloud   (on Jetson Nano)" >&2
    exit 2
  fi
fi

case "${ROLE}" in
  edge)
    echo "[setup] role=edge -> running setup_edge.sh"
    bash "${HERE}/edge/setup.sh"
    ;;
  cloud)
    echo "[setup] role=cloud -> running setup_cloud.sh"
    bash "${HERE}/cloud/setup.sh"
    ;;
  *)
    echo "ERROR: unknown role '${ROLE}' (expected 'edge' or 'cloud')" >&2
    exit 2
    ;;
esac

echo "[setup] done. service status:"
sudo systemctl status "plant-${ROLE}" --no-pager || true
