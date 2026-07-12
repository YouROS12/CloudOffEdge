#!/usr/bin/env bash
# CloudOffEdge -- one-command device update+run.
#
# Run this ON the Raspberry Pi (edge) or Jetson Nano (cloud). It pulls the
# latest code+checkpoints from GitHub, (re)installs, and (re)starts the
# systemd service. Role is autodetected; override with an argument.
#
#   bash deploy/pull_and_run.sh          # autodetect edge/cloud
#   bash deploy/pull_and_run.sh edge     # force edge  (Pi)
#   bash deploy/pull_and_run.sh cloud    # force cloud (Jetson)
#
# Idempotent: safe to re-run any time you push new code from your dev machine.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
ROLE="${1:-}"

echo "[pull_and_run] repo=${REPO}"
echo "[pull_and_run] fetching latest from GitHub ..."
git -C "${REPO}" pull --ff-only

echo "[pull_and_run] running setup (role='${ROLE:-autodetect}') ..."
bash "${HERE}/setup.sh" ${ROLE}

echo "[pull_and_run] done."
