# Configuration and Reproducibility

Experiment YAML owns task, spaces, plant, scene, sensors, perception, planner,
Memory, and RL ports. Runtime overrides are applied before the resolved
configuration is written.

Every run stores:

- resolved configuration;
- seed and Git SHA;
- configuration hash;
- plant backend and prediction mode;
- MuJoCo version and MJCF hash when applicable;
- checkpoint identity for trained models;
- trajectory and metric records.

Large outputs remain ignored under `results/`.

The initial refactor pins MuJoCo 3.2.3 so physics changes are not conflated
with a simulator-version upgrade.
