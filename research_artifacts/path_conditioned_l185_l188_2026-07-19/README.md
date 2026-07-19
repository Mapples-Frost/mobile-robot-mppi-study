# Path-conditioned ICODE--RL--MPPI development data

This package contains the versioned L185--L188 development evidence.
It includes per-step/per-episode CSV data, trajectories, configuration
snapshots, logs, selected SAC and ICODE checkpoints, calibration
summaries, and SHA256 provenance for every copied artifact.

The data are development-only. L186 sealed geometries and seeds
561--565 are deliberately absent. `experiment_index.csv` is the
compact success/metric index. Files ending in `.csv.gz` are lossless
gzip copies; their uncompressed hashes are in `package_manifest.json`.
