#!/usr/bin/env bash
# Finish the existing registered experiment without duplicate concurrent work.
set -euo pipefail
cd /home/mapples/projects/mobile-robot-mppi-study
CAL_PY=/home/mapples/.local/share/bohn2021-python37/bin/python
CAL_OUT=research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
exec 9>"$CAL_OUT/continuation.lock"
flock -n 9
# Original suite holds this lock; release the barrier before evaluation needs it.
timeout 14400 flock "$CAL_OUT/suite.lock" true
date -u
for CAL_TASK in vehicle pendulum; do
  for CAL_SEED in 0 1 2; do
    timeout 7200 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_resume.py train \
      --task "$CAL_TASK" --seed "$CAL_SEED" \
      > "$CAL_OUT/resume_${CAL_TASK}_s${CAL_SEED}.log" 2>&1
  done
done
timeout 1800 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_resume.py audit --phase training \
  > "$CAL_OUT/training_audit.log" 2>&1
timeout 7200 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_run.py evaluate --split validation \
  > "$CAL_OUT/validation_suite.log" 2>&1
timeout 1800 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_audit.py validation \
  > "$CAL_OUT/validation_audit.log" 2>&1
timeout 600 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_diagnosis.py \
  > "$CAL_OUT/ranking_diagnosis.log" 2>&1
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_delivery.py
# A failed efficacy gate is a result, not permission to inspect sealed test.
if "$CAL_PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$CAL_OUT/validation_gate.json"; then
  timeout 7200 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_run.py evaluate --split test \
    > "$CAL_OUT/test_suite.log" 2>&1
  timeout 1800 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_audit.py test \
    > "$CAL_OUT/test_audit.log" 2>&1
fi
timeout 7200 "$CAL_PY" -u experiments/bohn2021_reproduction/branch_calibration_timing.py \
  > "$CAL_OUT/serial_timing/run.log" 2>&1
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_timing_report.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_delivery.py
date -u
