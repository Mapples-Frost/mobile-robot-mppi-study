import torch

# Keep the optional PyTorch model out of residual/__init__.py and import its
# implementation explicitly.
from src.dynamics.residual.icode_residual import ICODEResidual


def _expected_network_parameters(input_dim, hidden_sizes, output_dim):
    widths = (input_dim,) + tuple(hidden_sizes) + (output_dim,)
    return sum((source + 1) * target for source, target in zip(widths, widths[1:]))


def test_icode_batch_and_unbatched_component_shapes_are_dimension_driven():
    feature_dim, state_dim, control_dim = 7, 5, 3
    model = ICODEResidual(
        state_feature_dim=feature_dim,
        state_dim=state_dim,
        control_dim=control_dim,
        hidden_sizes=(11, 13),
        activation="tanh",
    )

    state = torch.randn(feature_dim)
    control = torch.randn(control_dim)
    single = model(state, control, return_components=True)

    assert single["residual"].shape == (state_dim,)
    assert single["drift"].shape == (state_dim,)
    assert single["gain"].shape == (state_dim, control_dim)
    assert single["control_contribution"].shape == (state_dim,)
    assert model.drift(state).shape == (state_dim,)
    assert model.gain(state).shape == (state_dim, control_dim)

    batch_size = 4
    states = torch.randn(batch_size, feature_dim)
    controls = torch.randn(batch_size, control_dim)
    batch = model(states, controls, return_components=True)

    assert batch["residual"].shape == (batch_size, state_dim)
    assert batch["drift"].shape == (batch_size, state_dim)
    assert batch["gain"].shape == (batch_size, state_dim, control_dim)
    assert batch["control_contribution"].shape == (batch_size, state_dim)
    assert model(states, controls).shape == (batch_size, state_dim)


def test_icode_control_affine_bmm_identity_and_outputs_are_finite():
    torch.manual_seed(17)
    model = ICODEResidual(4, 3, 2, hidden_sizes=(9,), activation="softplus")
    states = torch.randn(6, 4, requires_grad=True)
    controls = torch.randn(6, 2, requires_grad=True)

    components = model(states, controls, return_components=True)
    expected_control = torch.bmm(
        components["gain"], controls.unsqueeze(-1)
    ).squeeze(-1)

    torch.testing.assert_close(components["control_contribution"], expected_control)
    torch.testing.assert_close(
        components["residual"], components["drift"] + expected_control
    )
    assert all(torch.isfinite(value).all() for value in components.values())

    components["residual"].square().mean().backward()
    assert torch.isfinite(states.grad).all()
    assert torch.isfinite(controls.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_icode_parameter_count_matches_configured_networks_without_3x2_hardcode():
    feature_dim, state_dim, control_dim = 6, 4, 5
    hidden_sizes = (8, 10)
    model = ICODEResidual(
        feature_dim,
        state_dim,
        control_dim,
        hidden_sizes=hidden_sizes,
        activation="relu",
    )

    expected = _expected_network_parameters(
        feature_dim, hidden_sizes, state_dim
    ) + _expected_network_parameters(
        feature_dim, hidden_sizes, state_dim * control_dim
    )

    assert model.parameter_count() == expected
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    output = model(torch.zeros(2, feature_dim), torch.zeros(2, control_dim))
    assert output.shape == (2, state_dim)
    assert torch.isfinite(output).all()
