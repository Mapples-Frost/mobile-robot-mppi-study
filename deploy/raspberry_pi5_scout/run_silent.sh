#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRIDGE_ROOT="/home/pi/rlmppi_pi5/livox_bridge"
SDK_CONFIG="/home/pi/Livox-SDK2-master/mid360_pi.json"
OUTPUT="${1:?usage: run_silent.sh OUTPUT_DIR [DURATION_S]}"
DURATION="${2:-15}"
BRIDGE_PID=""

cleanup() {
  if [[ -n "${BRIDGE_PID}" ]]; then
    kill -TERM "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if pgrep -f "rlmppi_livox_udp_bridge" >/dev/null; then
  echo "Refusing to start: another RL-MPPI Livox bridge is active" >&2
  exit 1
fi

"${BRIDGE_ROOT}/build/rlmppi_livox_udp_bridge" "${SDK_CONFIG}" \
  >"${OUTPUT}.bridge.stdout.log" 2>"${OUTPUT}.bridge.stderr.log" &
BRIDGE_PID=$!
sleep 2

"/home/pi/rlmppi_pi5/.venv/bin/python" \
  "${ROOT}/deploy/raspberry_pi5_scout/run_silent_full.py" \
  --duration-s "${DURATION}" \
  --zero-can \
  --lidar-config "${ROOT}/deploy/raspberry_pi5_scout/livox_adapter_pi5.yaml" \
  --output "${OUTPUT}"
