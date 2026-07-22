import numpy as np
import torch

from experiments.rl.run_l272_critic_architecture_causal_probe import (
    _augment,
    _conditioned_bundle,
    _conditioned_update,
    _copy_conditioned_critic,
    _filter_scene,
    _gate,
)
from experiments.rl.run_l268_critic_only_intervention import _frozen_hashes
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.sac import QNetwork, SACAgent, SACConfig


def test_filter_scene_preserves_exact_1000_rows():
    arrays = {
        "observations": np.zeros((6000, 69), dtype=np.float32),
        "actions": np.zeros((6000, 2), dtype=np.float32),
        "groups": np.repeat(np.arange(6), 1000).astype(np.int32),
    }
    filtered = _filter_scene(arrays, 3)
    assert filtered["observations"].shape == (1000, 69)
    assert filtered["actions"].shape == (1000, 2)
    assert np.all(filtered["groups"] == 3)


def test_conditioned_transplant_preserves_old_columns_and_zeros_context():
    config = SACConfig(
        hidden_sizes=(16, 16), critic_distribution="quantile", critic_num_quantiles=5,
    )
    source = QNetwork(69, 2, config)
    target = _copy_conditioned_critic(source, 69, 6, 2, config, torch.device("cpu"))
    source_weight = source.network[0].weight.detach()
    target_weight = target.network[0].weight.detach()
    assert torch.equal(target_weight[:, :69], source_weight[:, :69])
    assert torch.equal(target_weight[:, 75:], source_weight[:, 69:])
    assert torch.count_nonzero(target_weight[:, 69:75]) == 0
    for source_layer, target_layer in zip(source.network[2:], target.network[2:]):
        if hasattr(source_layer, "weight"):
            assert torch.equal(source_layer.weight, target_layer.weight)


def test_augment_appends_one_hot_without_changing_observation():
    observation = torch.randn(6, 69)
    groups = torch.arange(6)
    augmented = _augment(observation, groups, 6)
    assert augmented.shape == (6, 75)
    assert torch.equal(augmented[:, :69], observation)
    assert torch.equal(augmented[:, 69:], torch.eye(6))


def test_conditioned_update_changes_only_critic_and_stays_finite():
    config = SACConfig(
        hidden_sizes=(16, 16), critic_distribution="quantile", critic_num_quantiles=5,
    )
    agent = SACAgent(69, 2, config=config, device="cpu", seed=17)
    normalizer = RunningNormalizer(69)
    rng = np.random.RandomState(19)
    observations = rng.normal(size=(24, 69)).astype(np.float32)
    normalizer.update(observations)
    batch = {
        "observations": observations,
        "actions": np.tanh(rng.normal(size=(24, 2))).astype(np.float32),
        "rewards": rng.normal(size=(24, 1)).astype(np.float32),
        "next_observations": (
            observations + 0.01 * rng.normal(size=(24, 69))
        ).astype(np.float32),
        "dones": np.zeros((24, 1), dtype=np.float32),
        "groups": np.tile(np.arange(6), 4).astype(np.int32),
    }
    frozen = _frozen_hashes(agent)
    bundle = _conditioned_bundle(agent, 6)
    before = {
        name: value.detach().clone()
        for name, value in bundle["critic1"].state_dict().items()
    }
    metrics = _conditioned_update(bundle, agent, normalizer, batch, 6)
    assert np.isfinite(np.asarray(list(metrics.values()))).all()
    assert frozen == _frozen_hashes(agent)
    assert any(
        not torch.equal(before[name], value)
        for name, value in bundle["critic1"].state_dict().items()
    )


def test_gate_selects_conditioned_in_fixed_order_when_both_pass():
    results = []
    scenes = {f"s{i}": 0.5 for i in range(6)}
    for seed in (1, 2, 3):
        results.extend([
            {
                "seed": seed, "arm": "shared_69d",
                "recovery_forward_pair_accuracy": 0.50,
                "mean_three_action_spearman": 0.00,
                "top1_action_agreement": 0.30,
                "scene_pair_accuracy": scenes,
                "finite": True, "actor_alpha_unchanged": True,
                "maximum_absolute_mean_q": 1.0, "mean_quantile_spread": 1.0,
            },
            {
                "seed": seed, "arm": "per_scene_equal_compute",
                "recovery_forward_pair_accuracy": 0.75,
                "mean_three_action_spearman": 0.30,
                "top1_action_agreement": 0.60,
                "scene_pair_accuracy": {f"s{i}": 0.75 for i in range(6)},
                "finite": True, "actor_alpha_unchanged": True,
                "maximum_absolute_mean_q": 1.0, "mean_quantile_spread": 1.0,
            },
            {
                "seed": seed, "arm": "scene_conditioned_equal_compute",
                "recovery_forward_pair_accuracy": 0.80,
                "mean_three_action_spearman": 0.35,
                "top1_action_agreement": 0.65,
                "scene_pair_accuracy": {f"s{i}": 0.80 for i in range(6)},
                "finite": True, "actor_alpha_unchanged": True,
                "maximum_absolute_mean_q": 1.0, "mean_quantile_spread": 1.0,
            },
        ])
    gate = {
        "minimum_aggregate_pair_accuracy": 0.65,
        "minimum_paired_median_pair_accuracy_improvement": 0.10,
        "minimum_aggregate_three_action_spearman": 0.15,
        "minimum_paired_median_spearman_improvement": 0.10,
        "minimum_aggregate_top1_agreement": 0.45,
        "minimum_paired_median_top1_improvement": 0.05,
        "minimum_jointly_improving_seed_blocks": 2,
        "minimum_scenes_with_pair_accuracy_improvement": 4,
        "maximum_scene_pair_accuracy_decrease": 0.166667,
        "maximum_absolute_mean_q": 500.0,
        "maximum_mean_quantile_spread": 100.0,
    }
    rows, selected = _gate(
        results, gate,
        ["scene_conditioned_equal_compute", "per_scene_equal_compute"],
    )
    assert all(row["gate_pass"] for row in rows)
    assert selected == "scene_conditioned_equal_compute"
