from typing import Mapping

import numpy as np

from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.core.types import ControlCommand, SafetyDecision


class ScanGuardArbiter:
    """Final control boundary; neither Memory nor RL can bypass it."""

    def __init__(self, action_spec: ActionSpec):
        self.action_spec = action_spec

    def arbitrate(self, proposed: ControlCommand, guard_result: Mapping[str, object]):
        values = self.action_spec.clip(proposed.values)
        reason = str(guard_result.get("reason", "front_clear"))
        if bool(guard_result.get("emergency_stop", False)):
            if "v_cmd" in self.action_spec.names:
                values[self.action_spec.index("v_cmd")] = 0.0
        elif bool(guard_result.get("should_slow_down", False)):
            if "v_cmd" in self.action_spec.names:
                index = self.action_spec.index("v_cmd")
                values[index] = max(0.0, values[index]) * float(guard_result.get("slow_scale", 1.0))
        executed = ControlCommand(values, proposed.timestamp, "safety_arbitration")
        overridden = not np.allclose(executed.values, proposed.values, rtol=0.0, atol=1e-12)
        return SafetyDecision(proposed, executed, overridden, reason, dict(guard_result))
