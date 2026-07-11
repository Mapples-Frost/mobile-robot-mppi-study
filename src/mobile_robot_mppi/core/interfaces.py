from typing import Any, Mapping, Protocol, Tuple

import numpy as np

from .spaces import ActionSpec
from .types import ControlCommand, GroundTruth, PlanResult, PlantStep, RobotObservation, SafetyDecision


class PredictionDynamics(Protocol):
    state_dim: int
    control_dim: int

    def derivative(self, state: np.ndarray, control: np.ndarray, time=None) -> np.ndarray:
        ...


class PlantBackend(Protocol):
    def reset(self, seed: int, initial_state: np.ndarray) -> GroundTruth:
        ...

    def step(self, command: ControlCommand, dt: float) -> PlantStep:
        ...

    def ground_truth(self) -> GroundTruth:
        ...

    def close(self) -> None:
        ...


class SamplingPrior(Protocol):
    def propose(self, observation: RobotObservation, reference: Any, horizon: int, action_spec: ActionSpec):
        ...


class Controller(Protocol):
    def reset(self) -> None:
        ...

    def plan(self, observation: RobotObservation, reference: Any) -> PlanResult:
        ...


class SafetyArbiter(Protocol):
    def arbitrate(self, proposed: ControlCommand, guard_result: Mapping[str, Any]) -> SafetyDecision:
        ...
