import torch

from mobile_robot_mppi.rl.maneuver_actor import (
    FrozenManeuverProposalPolicy,
    ManeuverProposalActor,
    best_of_m_loss,
)
from mobile_robot_mppi.core.spaces import body_velocity_action


def test_maneuver_actor_outputs_bounded_multimodal_sequences():
    actor = ManeuverProposalActor(
        10, heads=3, horizon=36, action_dim=2, hidden_sizes=(16,)
    )
    values = actor(torch.zeros(4, 10))
    assert values.shape == (4, 3, 36, 2)
    assert torch.all(values <= 1.0)
    assert torch.all(values >= -1.0)


def test_best_of_m_selects_closest_head_and_penalizes_collapse():
    targets = torch.zeros(2, 4, 2)
    predictions = torch.zeros(2, 3, 4, 2)
    predictions[:, 0] = 0.8
    predictions[:, 1] = 0.1
    predictions[:, 2] = -0.7
    loss, details = best_of_m_loss(
        predictions, targets, diversity_margin=0.08, diversity_weight=0.02
    )
    assert details["assignments"].tolist() == [1, 1]
    assert float(details["best_mse"]) < 0.02
    assert torch.isfinite(loss)


def test_frozen_policy_loads_causal_proposal_only_checkpoint(tmp_path):
    model = ManeuverProposalActor(
        4, heads=3, horizon=5, action_dim=2, hidden_sizes=(8,)
    )
    path = tmp_path / "actor.pt"
    torch.save({
        "schema_version": 1,
        "model": model.state_dict(),
        "model_config": {
            "heads": 3,
            "horizon": 5,
            "action_dim": 2,
            "hidden_sizes": [8],
        },
        "observation_dim": 4,
        "observation_mean": torch.zeros(4).numpy(),
        "observation_std": torch.ones(4).numpy(),
        "student_input_causal_only": True,
        "execution_authority": "proposal_only",
    }, path)

    policy = FrozenManeuverProposalPolicy.from_checkpoint(path)
    proposals = policy.propose(
        [0.0, 0.0, 0.0, 0.0],
        body_velocity_action((0.0, 0.5), 1.0),
    )

    assert proposals.shape == (3, 5, 2)
    assert (proposals[..., 0] >= 0.0).all()
    assert (proposals[..., 0] <= 0.5).all()
    assert (proposals[..., 1] >= -1.0).all()
    assert (proposals[..., 1] <= 1.0).all()
