#!/usr/bin/env bash
# Run offline diagnostics only after the active serial timing pipeline releases its lock.
set -euo pipefail
cd /home/mapples/projects/mobile-robot-mppi-study
CAL_OUT=research_artifacts/bohn2021_reproduction_2026-09-17/results/branch_calibration_2026-09-24
exec 9>"$CAL_OUT/postprocess.lock"
flock -n 9
timeout 14400 flock "$CAL_OUT/continuation.lock" true
timeout 14400 flock "$CAL_OUT/serial_timing/run.lock" true
test -f "$CAL_OUT/serial_timing/timing_audit.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_failure_diagnosis.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_claim_audit.py
.venv/bin/python experiments/bohn2021_reproduction/branch_calibration_delivery.py
date -u
