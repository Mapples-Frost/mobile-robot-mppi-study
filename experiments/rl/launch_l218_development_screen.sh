#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
SEED="${1:-91001}"
ARMS="${2:-simple_combination,full_proposed}"
LABEL="${3:-coupled}"
MANIFEST="${4:-configs/research/expanded_navigation_screen_l218.yaml}"
PHYSICS_DOMAINS="${5:-nominal_seen}"
SCENE_FILTER="${6:-all}"
LOG_ROOT="/mnt/c/Users/lenovo"

if [[ ! -x "${PYTHON}" ]]; then
  echo "project Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

scenes=(
  "serpentine:mujoco_l218_serpentine_polyline.yaml"
  "giant_u:mujoco_l218_giant_u_polyline.yaml"
  "opposed_u:mujoco_l218_opposed_u_polyline.yaml"
  "nested_u:mujoco_l218_nested_u_polyline.yaml"
  "cylinder_forest:mujoco_l218_cylinder_forest_polyline.yaml"
  "cylinder_spiral:mujoco_l218_cylinder_spiral_polyline.yaml"
)

pid_file="${LOG_ROOT}/l218_dev_screen_${LABEL}_pids.csv"
printf 'scene,pid,seed\n' >"${pid_file}"
declare -a child_pids=()
cd "${ROOT}"
for item in "${scenes[@]}"; do
  slug="${item%%:*}"
  scene_file="${item#*:}"
  if [[ "${SCENE_FILTER}" != "all" ]] && ! grep -qE "(^|,)${slug}(,|$)" <<<"${SCENE_FILTER}"; then
    continue
  fi
  stdout="${LOG_ROOT}/l218_dev_screen_${LABEL}_${slug}.stdout.log"
  stderr="${LOG_ROOT}/l218_dev_screen_${LABEL}_${slug}.stderr.log"
  output="results/research_platform/rl/l218_dev_screen_${LABEL}_${slug}_seed${SEED}"
  env \
    PYTHONPATH=.:src \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    nohup "${PYTHON}" experiments/rl/run_final_paper_benchmark.py \
      --manifest "${MANIFEST}" \
      --output-dir "${output}" \
      --qualification \
      --seeds "${SEED}" \
      --arms "${ARMS}" \
      --scene-configs "configs/research/${scene_file}" \
      --physics-domains "${PHYSICS_DOMAINS}" \
      --bootstrap-samples 100 \
      >"${stdout}" 2>"${stderr}" </dev/null &
  printf '%s,%d,%s\n' "${slug}" "$!" "${SEED}" >>"${pid_file}"
  child_pids+=("$!")
done

cat "${pid_file}"

status=0
for pid in "${child_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
