from types import SimpleNamespace

import numpy as np
import pytest

from mobile_robot_mppi.core.references import PointGoal
from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.core.types import (
    LaserScan,
    Pose2D,
    RobotObservation,
    Twist2D,
)
from mobile_robot_mppi.policies.priors import GoalWarmStartPrior
from mobile_robot_mppi.rl.observation import ObservationEncoder, RunningNormalizer
from mobile_robot_mppi.rl.parameterization import (
    PriorParameterization,
    PriorParameterizationConfig,
)
from mobile_robot_mppi.rl.prior import GateConfig, TorchSACPrior
from mobile_robot_mppi.rl.scene_complexity import (
    SceneComplexityConfig,
    score_scene_complexity,
)


def _scan(ranges, obstacle_ranges=None, timestamp=0.0):
    values = np.asarray(ranges, dtype=np.float64)
    return LaserScan(
        ranges=values,
        obstacle_ranges=obstacle_ranges,
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / float(values.size - 1),
        range_min=0.05,
        range_max=4.0,
        timestamp=float(timestamp),
    )


def _observation(scan):
    return RobotObservation(
        float(scan.timestamp),
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        scan=scan,
    )


def test_missing_and_clear_scans_fail_closed_to_zero_complexity():
    missing = score_scene_complexity(None)
    clear = score_scene_complexity(_scan(np.full(37, 4.0)))
    assert missing.scan_valid is False
    assert missing.score == 0.0
    assert clear.scan_valid is True
    assert clear.score == 0.0
    assert np.isfinite(tuple(clear.to_dict().values())[:-1]).all()


def test_close_frontal_obstacle_monotonically_increases_score():
    far = np.full(37, 4.0)
    medium = far.copy()
    close = far.copy()
    medium[18] = 1.0
    close[18] = 0.40
    far_score = score_scene_complexity(_scan(far))
    medium_score = score_scene_complexity(_scan(medium))
    close_score = score_scene_complexity(_scan(close))
    assert far_score.score < medium_score.score < close_score.score
    assert close_score.front_proximity == 1.0


def test_bilateral_side_obstacles_create_constriction_not_one_wall():
    one_side = np.full(73, 4.0)
    bilateral = one_side.copy()
    one_side[45] = 0.50
    bilateral[45] = 0.50
    bilateral[27] = 0.50
    single_score = score_scene_complexity(_scan(one_side))
    bilateral_score = score_scene_complexity(_scan(bilateral))
    assert single_score.constriction == 0.0
    assert bilateral_score.constriction > 0.9
    assert bilateral_score.score > single_score.score


def test_obstacle_density_is_normalized_by_all_scan_beams():
    sparse = np.full(101, 4.0)
    dense = sparse.copy()
    sparse[:5] = 1.0
    dense[:31] = 1.0
    sparse_score = score_scene_complexity(_scan(sparse))
    dense_score = score_scene_complexity(_scan(dense))
    assert 0.0 < sparse_score.density < dense_score.density
    assert dense_score.density == 1.0


@pytest.mark.parametrize(
    "mapping,match",
    [
        ({"front_half_angle_deg": 130, "side_outer_angle_deg": 120}, "angles"),
        ({"near_distance_m": 1.0, "far_distance_m": 0.5}, "distances"),
        ({"density_full_fraction": 0.0}, "fraction"),
        ({"soft_threshold": 0.8, "hard_threshold": 0.7}, "thresholds"),
    ],
)
def test_complexity_configuration_rejects_invalid_contract(mapping, match):
    with pytest.raises(ValueError, match=match):
        SceneComplexityConfig.from_mapping(mapping).validate()


def test_complexity_gate_blends_from_traditional_to_learned_prior():
    class DummyAgent:
        is_correction_policy = False

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            return np.ones(4, dtype=np.float32), {"alpha": 0.2}

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder(
        {
            "lidar_sectors": 4,
            "include_previous_action": False,
            "include_safety_state": False,
        },
        action_spec,
    )
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    fallback = GoalWarmStartPrior()
    prior = TorchSACPrior(
        DummyAgent(),
        encoder,
        normalizer,
        decoder,
        gate_config={"mode": "complexity"},
        fallback_prior=fallback,
    )
    clear_observation = _observation(_scan(np.full(37, 4.0)))
    close_ranges = np.full(37, 4.0)
    close_ranges[18] = 0.35
    close_observation = _observation(_scan(close_ranges))
    reference = PointGoal(2.0, 0.0)

    clear = prior.propose(clear_observation, reference, 5, action_spec)
    expected = fallback.propose(
        clear_observation, reference, 5, action_spec
    )
    close = prior.propose(close_observation, reference, 5, action_spec)
    assert clear.metadata["gate_alpha"] == 0.0
    assert clear.metadata["scene_complexity_score"] == 0.0
    np.testing.assert_allclose(clear.mean, expected.mean)
    assert close.metadata["gate_alpha"] == 1.0
    assert close.metadata["scene_complexity_score"] == 1.0
    assert not np.allclose(close.mean, expected.mean)


def test_gate_config_accepts_nested_complexity_and_rejects_unknown_mode():
    config = GateConfig.from_mapping({
        "mode": "complexity",
        "complexity": {"soft_threshold": 0.2, "hard_threshold": 0.6},
    })
    config.validate()
    assert config.complexity.soft_threshold == 0.2
    with pytest.raises(ValueError, match="mode"):
        GateConfig(mode="oracle").validate()


def test_correction_support_gate_attenuates_only_the_sac_increment():
    class MutableNormalizer:
        def __init__(self):
            self.score = 2.0

        def normalize(self, observation):
            return np.asarray(observation, dtype=np.float64)

        def ood_score(self, observation):
            del observation
            return self.score

    class CorrectionAgent:
        is_correction_policy = True

        def eval(self):
            return None

        def select_action(
            self, observation, deterministic=False, include_internal=False
        ):
            del observation, deterministic
            diagnostics = {
                "policy_mode": "frozen_bc_correction",
                "base_action_abs_mean": 0.0,
                "unit_correction_abs_mean": 1.0,
                "applied_correction_abs_mean": 1.0,
                "applied_correction_abs_max": 1.0,
            }
            if include_internal:
                diagnostics["_base_action"] = np.zeros(4, dtype=np.float32)
            return np.ones(4, dtype=np.float32), diagnostics

        def filter_correction_by_advantage(
            self, observation, candidate_action, **kwargs
        ):
            del observation, kwargs
            return np.asarray(candidate_action), {
                "correction_advantage_gate_alpha": 1.0,
                "correction_advantage_gate_mode": "lcb",
                "correction_advantage_critic_source": "target",
                "correction_advantage_threshold": 0.0,
                "correction_advantage_uncertainty_multiplier": 2.0,
                "raw_applied_correction_abs_mean": 1.0,
                "applied_correction_abs_mean": 1.0,
                "applied_correction_abs_max": 1.0,
            }

        def frozen_base_action(self, observation):
            del observation
            return np.zeros(4, dtype=np.float32)

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder({
        "lidar_sectors": 4,
        "include_previous_action": False,
        "include_safety_state": False,
    }, action_spec)
    normalizer = MutableNormalizer()
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    prior = TorchSACPrior(
        CorrectionAgent(),
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "complexity",
            "reuse_policy_base_action": True,
            "correction_support_gate_enabled": True,
            "correction_support_soft_threshold": 3.0,
            "correction_support_hard_threshold": 4.5,
        },
        fallback_prior=GoalWarmStartPrior(),
    )
    ranges = np.full(37, 4.0)
    ranges[18] = 0.35
    observation = _observation(_scan(ranges))
    reference = PointGoal(2.0, 0.0)
    full = prior.propose(observation, reference, 5, action_spec)
    assert full.metadata["correction_support_confidence"] == 1.0
    assert full.metadata["correction_effective_gate_alpha"] == 1.0
    assert full.metadata["applied_correction_abs_mean"] == 1.0

    normalizer.score = 4.5
    suppressed = prior.propose(observation, reference, 5, action_spec)
    assert suppressed.metadata["gate_alpha"] == 1.0
    assert suppressed.metadata["correction_support_confidence"] == 0.0
    assert suppressed.metadata["correction_effective_gate_alpha"] == 0.0
    assert suppressed.metadata["applied_correction_abs_mean"] == 0.0
    assert not np.allclose(full.mean, suppressed.mean)


@pytest.mark.parametrize(
    "mapping",
    [
        {
            "correction_support_soft_threshold": 2.0,
            "correction_support_hard_threshold": 1.0,
        },
        {"correction_support_soft_threshold": -1.0},
    ],
)
def test_correction_support_gate_rejects_invalid_thresholds(mapping):
    with pytest.raises(ValueError, match="support hard threshold"):
        GateConfig.from_mapping(mapping).validate()


def test_zero_complexity_fastpath_skips_agent_without_changing_prior():
    class CountingAgent:
        is_correction_policy = False

        def __init__(self):
            self.calls = 0

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            self.calls += 1
            return np.ones(4, dtype=np.float32), {"alpha": 0.2}

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder(
        {
            "lidar_sectors": 4,
            "include_previous_action": False,
            "include_safety_state": False,
        },
        action_spec,
    )
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    agent = CountingAgent()
    fallback = GoalWarmStartPrior()
    prior = TorchSACPrior(
        agent,
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "complexity",
            "skip_zero_complexity_inference": True,
        },
        fallback_prior=fallback,
    )
    observation = _observation(_scan(np.full(37, 4.0)))
    reference = PointGoal(2.0, 0.0)
    result = prior.propose(observation, reference, 5, action_spec)
    expected = fallback.propose(observation, reference, 5, action_spec)
    assert agent.calls == 0
    assert result.metadata["learned_inference_skipped"] is True
    assert result.metadata["gate_alpha"] == 0.0
    np.testing.assert_array_equal(result.mean, expected.mean)
    assert result.covariance is expected.covariance

    close = np.full(37, 4.0)
    close[18] = 0.35
    active = prior.propose(_observation(_scan(close)), reference, 5, action_spec)
    assert agent.calls == 1
    assert active.metadata["learned_inference_skipped"] is False


def test_progress_complexity_gate_only_runs_actor_after_stagnation():
    class CountingAgent:
        is_correction_policy = False

        def __init__(self):
            self.calls = 0

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            self.calls += 1
            return np.ones(4, dtype=np.float32), {"alpha": 0.2}

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder({
        "lidar_sectors": 4,
        "include_previous_action": False,
        "include_safety_state": False,
    }, action_spec)
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    agent = CountingAgent()
    prior = TorchSACPrior(
        agent,
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "progress_complexity",
            "skip_zero_complexity_inference": True,
            "progress": {
                "observation_window_s": 1.0,
                "minimum_window_coverage_s": 1.0,
                "full_activation_progress_m": 0.04,
                "zero_activation_progress_m": 0.20,
                "activation_hold_s": 0.0,
            },
        },
        fallback_prior=GoalWarmStartPrior(),
    )
    close = np.full(37, 4.0)
    close[18] = 0.35
    reference = PointGoal(2.0, 0.0)

    def observation(timestamp, x):
        return RobotObservation(
            timestamp,
            Pose2D(x, 0.0, 0.0),
            Twist2D(0.0, 0.0),
            scan=_scan(close, timestamp=timestamp),
        )

    first = prior.propose(observation(0.0, 0.0), reference, 5, action_spec)
    assert first.metadata["gate_alpha"] == 0.0
    assert first.metadata["baseline_progress_gate_ready"] is False
    assert agent.calls == 0

    stagnant = prior.propose(
        observation(1.0, 0.01), reference, 5, action_spec
    )
    assert stagnant.metadata["baseline_progress_gate_ready"] is True
    assert stagnant.metadata["baseline_stagnation_activation"] == 1.0
    assert stagnant.metadata["gate_alpha"] == 1.0
    assert agent.calls == 1

    prior.reset()
    prior.propose(observation(2.0, 0.0), reference, 5, action_spec)
    progressing = prior.propose(
        observation(3.0, 0.30), reference, 5, action_spec
    )
    assert progressing.metadata["baseline_stagnation_activation"] == 0.0
    assert progressing.metadata["gate_alpha"] == 0.0
    assert progressing.metadata["learned_inference_skipped"] is True
    assert agent.calls == 1


def test_temporal_closing_risk_activates_on_scan_only_side_approach():
    class CountingAgent:
        is_correction_policy = False

        def __init__(self):
            self.calls = 0

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            self.calls += 1
            return np.ones(4, dtype=np.float32), {"alpha": 0.2}

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder({
        "lidar_sectors": 4,
        "include_previous_action": False,
        "include_safety_state": False,
    }, action_spec)
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    agent = CountingAgent()
    prior = TorchSACPrior(
        agent,
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "complexity",
            "skip_zero_complexity_inference": True,
            "temporal_closing_enabled": True,
            "temporal_closing_soft_mps": 0.05,
            "temporal_closing_hard_mps": 0.20,
        },
        fallback_prior=GoalWarmStartPrior(),
    )
    reference = PointGoal(2.0, 0.0)
    clear = np.full(73, 4.0)
    first = prior.propose(
        _observation(_scan(clear, timestamp=0.0)),
        reference,
        5,
        action_spec,
    )
    assert first.metadata["gate_alpha"] == 0.0
    assert agent.calls == 0

    side = clear.copy()
    side[45] = 1.0
    second_observation = _observation(_scan(side, timestamp=0.1))
    second = prior.propose(second_observation, reference, 5, action_spec)
    assert score_scene_complexity(second_observation.scan).constriction == 0.0
    assert second.metadata["temporal_closing_gate_alpha"] == 1.0
    assert second.metadata["gate_alpha"] == 1.0
    assert second.metadata["learned_inference_skipped"] is False
    assert agent.calls == 1

    prior.reset()
    reset = prior.propose(second_observation, reference, 5, action_spec)
    assert reset.metadata["temporal_closing_gate_alpha"] == 0.0


@pytest.mark.parametrize(
    "mapping,match",
    [
        ({"temporal_closing_soft_mps": 0.2,
          "temporal_closing_hard_mps": 0.1}, "hard rate"),
        ({"temporal_closing_max_clearance_m": 0.0}, "clearance"),
        ({"temporal_closing_hold_s": -0.1}, "hold"),
    ],
)
def test_temporal_closing_configuration_rejects_invalid_values(mapping, match):
    with pytest.raises(ValueError, match=match):
        GateConfig.from_mapping(mapping).validate()


def test_complexity_confidence_separates_hazard_from_policy_competence():
    class DummyAgent:
        is_correction_policy = False

        def eval(self):
            return None

        def select_action(self, observation, deterministic=False):
            del observation, deterministic
            return np.ones(4, dtype=np.float32), {"alpha": 0.2}

        def critic_disagreement(self, observation, action):
            del observation, action
            return 0.0

    action_spec = body_velocity_action((0.0, 0.4), 1.0)
    encoder = ObservationEncoder({
        "lidar_sectors": 4,
        "include_previous_action": False,
        "include_safety_state": False,
    }, action_spec)
    normalizer = RunningNormalizer(encoder.dimension)
    normalizer.update(np.zeros((2, encoder.dimension)))
    decoder = PriorParameterization(
        PriorParameterizationConfig(num_knots=2, mode="delta"),
        action_spec,
        (0.1, 0.2),
    )
    prior = TorchSACPrior(
        DummyAgent(),
        encoder,
        normalizer,
        decoder,
        gate_config={
            "mode": "complexity_confidence",
            "ood_soft_threshold": 1.0,
            "ood_hard_threshold": 2.0,
            "temporal_closing_enabled": True,
            "temporal_closing_source": "perception_scan_flow",
        },
        fallback_prior=GoalWarmStartPrior(),
    )
    scan = _scan(np.full(37, 4.0))
    observation = RobotObservation(
        scan.timestamp,
        Pose2D(0.0, 0.0, 0.0),
        Twist2D(0.0, 0.0),
        scan=scan,
        auxiliary={
            "temporal_scan_flow": {
                "risk_alpha": 1.0,
                "closing_rate_mps": 0.4,
                "held": False,
            }
        },
    )
    result = prior.propose(observation, PointGoal(2.0, 0.0), 5, action_spec)

    assert result.metadata["hazard_activation"] == 1.0
    assert result.metadata["ood_score"] >= 2.0
    assert result.metadata["competence_confidence"] == 0.0
    assert result.metadata["gate_alpha"] == 0.0
    assert result.metadata["temporal_closing_rate_mps"] == 0.4
