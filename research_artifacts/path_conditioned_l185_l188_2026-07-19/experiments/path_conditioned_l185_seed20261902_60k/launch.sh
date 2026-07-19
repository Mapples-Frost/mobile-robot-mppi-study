#!/usr/bin/env bash
set -euo pipefail
cd /home/mapples/projects/mobile-robot-mppi-study
exec .venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/path_conditioned_direct_control_sac_l185.yaml \
  --output-dir results/research_platform/rl/path_conditioned_l185_seed20261902_60k \
  --seed 20261902 \
  > results/research_platform/rl/path_conditioned_l185_seed20261902_60k/console.log \
  2>&1
