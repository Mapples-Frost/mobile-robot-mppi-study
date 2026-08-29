# Student Scout Mini contribution snapshot

This directory preserves the reviewed, non-production parts of the student
work supplied on 2026-08-29.

Included material:

- `dynamic_mppi_sim_v9.py`: experimental SLAM-map/MPPI simulation;
- `deployment/inference/`: BC policy inference and simulated controller;
- `deployment/model/model_metadata.json`: model architecture metadata;
- `results/`: reported simulation/evaluation summaries;
- `pc_pi_protocol.md`: the student's PC/Pi progress note.

The trained `.pth` file is intentionally not copied into this branch. Its
reported 88% success rate is an offline result and the policy does not include
the paper system's change-aware forecasting, risk-aware MPPI or physical
perception/control contracts.

Nothing under `contrib/` is imported by the production package or deployment
entry points. See
`docs/real_robot/scout_pi5_student_sync_2026-08-29.md` for the audit and the
current physical qualification boundary.
