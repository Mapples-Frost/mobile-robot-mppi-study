import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    return value


class LegacyMemoryAdapter:
    """Preserves the validated Memory field behind a typed runtime port."""

    def __init__(self, project_root, config):
        path = Path(project_root) / "mppi_hardware_bridge" / "scripts" / "mppi_memory_field.py"
        spec = importlib.util.spec_from_file_location("mppi_memory_compat", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("cannot load Memory field")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.field = module.MppiMemoryField(_namespace({"memory": dict(config)}))

    @property
    def enabled(self):
        return bool(self.field.enabled)

    def reset(self):
        self.field.reset()

    def trajectory_cost(self, trajectory, controls):
        return float(self.field.memory_cost_breakdown_for_trajectory(
            trajectory, controls=controls, stride=3
        )["total"])

    def update(self, observation, reference, executed_control, guard):
        state = observation.pose.as_array()
        target = reference.target_at(observation.timestamp, state).pose
        distance = math.hypot(target.x - observation.pose.x, target.y - observation.pose.y)
        return self.field.update(
            state=tuple(state),
            goal_distance=distance,
            control=tuple(executed_control.values[:2]),
            min_front_range=guard.get("min_front_range"),
            avoidance_state=guard.get("front_stop_mode", "CLEAR"),
            now=observation.timestamp,
        )

    def snapshot(self, state=None):
        return self.field.debug_snapshot(state)
