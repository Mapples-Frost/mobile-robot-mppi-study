from typing import Mapping

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import ControlCommand, SafetyDecision


class ScanGuardArbiter:
    """Final control boundary; neither Memory nor RL can bypass it."""

    def __init__(self, action_spec: ActionSpec, config=None):
        self.action_spec = action_spec
        self.config = dict(config or {})
        self.front_soft_block_max_speed = float(
            self.config.get("front_soft_block_max_speed", 0.0)
        )
        if self.front_soft_block_max_speed < 0.0:
            raise ValueError("front_soft_block_max_speed must be non-negative")

    def arbitrate(self, proposed: ControlCommand, guard_result: Mapping[str, object]):
        values = self.action_spec.clip(proposed.values)
        reason = str(guard_result.get("reason", "front_clear"))
        if bool(guard_result.get("emergency_stop", False)):
            if "v_cmd" in self.action_spec.names:
                values[self.action_spec.index("v_cmd")] = 0.0
        elif reason == "front_soft_block":
            # The legacy ROS bridge has a stateful, separately tested creep
            # recovery.  The Python-3 research runtime does not.  Treating a
            # soft block as scale=1 would therefore command full translation
            # inside the body stopping envelope.  Stop translation while
            # retaining omega, so MPPI can turn away without bypassing safety.
            if "v_cmd" in self.action_spec.names:
                index = self.action_spec.index("v_cmd")
                values[index] = min(
                    max(0.0, values[index]), self.front_soft_block_max_speed
                )
        elif bool(guard_result.get("should_slow_down", False)):
            if "v_cmd" in self.action_spec.names:
                index = self.action_spec.index("v_cmd")
                values[index] = max(0.0, values[index]) * float(guard_result.get("slow_scale", 1.0))
        executed = ControlCommand(values, proposed.timestamp, "safety_arbitration")
        overridden = not np.allclose(executed.values, proposed.values, rtol=0.0, atol=1e-12)
        return SafetyDecision(proposed, executed, overridden, reason, dict(guard_result))
