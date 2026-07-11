# Git and Release Strategy

The pre-refactor ICODE state is preserved by commit and by branch:

```text
codex/backup-icode-pre-refactor-20260711
```

The complete refactor is developed on:

```text
codex/mujoco-research-platform-refactor
```

The old research branch is not force-updated. Integration uses a normal merge
only after compilation, legacy tests, platform tests, smoke data collection,
MLP/ICODE smoke training, checkpoint evaluation, and MuJoCo benchmark pass.

No generated dataset, checkpoint, video, or benchmark result is committed.
