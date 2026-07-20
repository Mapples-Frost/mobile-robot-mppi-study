#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
CONFIG="configs/rl/residual_conditioned_expanded_maps_l219.yaml"
INITIAL_ACTOR="results/research_platform/rl/residual_conditioned_l204_seed20262041_60k_v1/checkpoints/best.pt"
LOG_ROOT="/mnt/c/Users/lenovo"

if [[ ! -x "${PYTHON}" ]]; then
  echo "project Python is unavailable: ${PYTHON}" >&2
  exit 2
fi

seeds=(20262191 20262192 20262193)
pid_file="${LOG_ROOT}/l219_residual_conditioned_training_pids.csv"
printf 'seed,pid,output_dir\n' >"${pid_file}"
declare -a child_pids=()

cd "${ROOT}"
for seed in "${seeds[@]}"; do
  output="results/research_platform/rl/l219_expanded_actor_seed${seed}_30k_v1"
  stdout="${LOG_ROOT}/l219_expanded_actor_seed${seed}.stdout.log"
  stderr="${LOG_ROOT}/l219_expanded_actor_seed${seed}.stderr.log"
  env \
    PYTHONPATH=.:src \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    nohup "${PYTHON}" experiments/rl/train_rl_sampling_prior.py \
      --config "${CONFIG}" \
      --output-dir "${output}" \
      --initialize-actor-from "${INITIAL_ACTOR}" \
      --seed "${seed}" \
      --device cpu \
      >"${stdout}" 2>"${stderr}" </dev/null &
  printf '%d,%d,%s\n' "${seed}" "$!" "${output}" >>"${pid_file}"
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
