#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pi/rlmppi_pi5/current
BRIDGE=/home/pi/rlmppi_pi5/livox_bridge/build/rlmppi_livox_udp_bridge
SDK_CONFIG=/home/pi/Livox-SDK2-master/mid360_pi.json
PYTHON=/home/pi/rlmppi_pi5/.venv/bin/python
TOKEN=${TOKEN:?set TOKEN to the one-time PC/Pi session token}
OUT=${OUT:?set OUT to a new experiment directory}
DURATION_S=${DURATION_S:-30}
ALLOW_ARM=${ALLOW_ARM:-0}
MAX_V_MPS=${MAX_V_MPS:-0.05}
MAX_REVERSE_V_MPS=${MAX_REVERSE_V_MPS:-0.05}
MAX_OMEGA_RADPS=${MAX_OMEGA_RADPS:-0.10}
SCAN_STRIDE=${SCAN_STRIDE:-8}
CAN_BITRATE=${CAN_BITRATE:-500000}
MAX_LINEAR_ACCELERATION_MPS2=${MAX_LINEAR_ACCELERATION_MPS2:-1.0}
MAX_LINEAR_DECELERATION_MPS2=${MAX_LINEAR_DECELERATION_MPS2:-2.0}
MAX_ANGULAR_ACCELERATION_RADPS2=${MAX_ANGULAR_ACCELERATION_RADPS2:-4.0}
BRIDGE_PID=

cleanup() {
  if [[ -n "${BRIDGE_PID}" ]]; then
    kill -TERM "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -e "${OUT}" ]]; then
  echo "Refusing to overwrite ${OUT}" >&2
  exit 2
fi
if pgrep -f "rlmppi_livox_udp_bridge" >/dev/null; then
  echo "Refusing to start: another RL-MPPI Livox bridge is active" >&2
  exit 3
fi

# Raspberry Pi reboots leave the gs_usb SocketCAN interface present but DOWN.
# Bring it up before the zero-only gateway starts; this transmits no chassis
# motion and preserves the Python gateway's arm/watchdog interlocks.
if ! ip link show can0 >/dev/null 2>&1; then
  echo "Required SocketCAN interface can0 is missing" >&2
  exit 5
fi
if ! ip -details link show can0 | grep -q "bitrate ${CAN_BITRATE}"; then
  sudo ip link set can0 down 2>/dev/null || true
  sudo ip link set can0 type can bitrate "${CAN_BITRATE}"
fi
if ! ip link show can0 | grep -q "UP"; then
  sudo ip link set can0 up
fi

mkdir -p "${OUT}"
"${BRIDGE}" "${SDK_CONFIG}" \
  >"${OUT}/bridge.stdout.log" 2>"${OUT}/bridge.stderr.log" &
BRIDGE_PID=$!
sleep 2
if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
  echo "Livox bridge exited during startup" >&2
  exit 4
fi

gateway_args=(
  --token "${TOKEN}"
  --output "${OUT}/gateway"
  --duration-s "${DURATION_S}"
  --max-v-mps "${MAX_V_MPS}"
  --max-reverse-v-mps "${MAX_REVERSE_V_MPS}"
  --max-omega-radps "${MAX_OMEGA_RADPS}"
  --scan-stride "${SCAN_STRIDE}"
  --watchdog-timeout-s 0.35
  --max-linear-acceleration-mps2 "${MAX_LINEAR_ACCELERATION_MPS2}"
  --max-linear-deceleration-mps2 "${MAX_LINEAR_DECELERATION_MPS2}"
  --max-angular-acceleration-radps2 "${MAX_ANGULAR_ACCELERATION_RADPS2}"
)
if [[ "${ALLOW_ARM}" == "1" ]]; then
  gateway_args+=(--allow-arm)
fi

cd "${ROOT}"
PYTHONPATH="${ROOT}/src" "${PYTHON}" \
  "${ROOT}/deploy/raspberry_pi5_scout/run_remote_pi_gateway.py" \
  "${gateway_args[@]}" \
  >"${OUT}/gateway.stdout.log" 2>"${OUT}/gateway.stderr.log"
