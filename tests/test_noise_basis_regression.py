"""Regression tests for the ``noise_basis`` sampling option.

The controlling requirement is that the default (``noise_basis == "iid"``)
reproduces the historical sampler BIT-EXACTLY. Every frozen protocol, every
sealed result, and the 1,860-episode v4 matrix depend on that. These tests
assert it against an explicit reference implementation of the pre-change code
rather than trusting inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobile_robot_mppi.core.spaces import (
    ActionSpec,
    body_velocity_action,
    unicycle_state,
)
from mobile_robot_mppi.planning.dynamics import LegacyUnicyclePrediction
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController
from mobile_robot_mppi.planning.rl_driven_mppi import (
    PaperRLDrivenMppiController,
)
from mobile_robot_mppi.policies.priors import PriorOutput
from mobile_robot_mppi.sampling.bases import (
    Ar1Basis,
    IidBasis,
    PiecewiseConstantBasis,
    build_basis,
    verify_marginal_variance,
)

HORIZON = 36
SAMPLES = 600
SIGMA = np.array([0.12, 0.35])


def _legacy_noise(rng, num_samples, horizon, sigma):
    """Verbatim pre-change expression from planning/mppi.py:1653."""
    shape = (num_samples, horizon, len(sigma))
    return rng.normal(size=shape) * np.asarray(sigma)[None, None, :]


class TestDefaultIsBitExact:
    def test_iid_basis_matches_legacy_expression_exactly(self):
        for seed in (0, 1, 7, 12345):
            legacy = _legacy_noise(
                np.random.default_rng(seed), SAMPLES, HORIZON, SIGMA
            )
            current = IidBasis().sample(
                np.random.default_rng(seed), SAMPLES, HORIZON, SIGMA
            )
            assert legacy.shape == current.shape
            # Bit-exact, not approximate.
            assert np.array_equal(legacy, current), (
                f"iid basis diverged from the legacy sampler at seed {seed}"
            )

    def test_iid_consumes_the_same_rng_stream(self):
        """A later draw must be identical, proving RNG consumption matches."""
        legacy_rng = np.random.default_rng(99)
        current_rng = np.random.default_rng(99)
        _legacy_noise(legacy_rng, SAMPLES, HORIZON, SIGMA)
        IidBasis().sample(current_rng, SAMPLES, HORIZON, SIGMA)
        assert np.array_equal(
            legacy_rng.normal(size=(4, 3)), current_rng.normal(size=(4, 3))
        )

    def test_build_basis_default_spec_is_iid(self):
        assert isinstance(build_basis("iid"), IidBasis)

    @pytest.mark.parametrize("seed", [0, 1, 42, 777])
    def test_full_mppi_sample_matches_legacy_body_exactly(self, seed):
        horizon = 7
        count = 48
        sigma = np.array([0.20, 0.45])
        action = body_velocity_action((-0.15, 0.45), 1.0)
        config = MppiConfig(
            horizon=horizon,
            num_samples=count,
            dt=0.1,
            noise_sigma=tuple(sigma),
            noise_basis="iid",
            seed=seed,
        )
        controller = MppiController(
            LegacyUnicyclePrediction(), unicycle_state(), action, config
        )
        prior_mean = np.column_stack(
            (
                np.linspace(-0.05, 0.35, horizon),
                np.linspace(-0.75, 0.75, horizon),
            )
        )
        prior = PriorOutput(prior_mean, None, {})

        legacy_rng = np.random.RandomState(seed)
        shape = (count, horizon, action.dimension)
        legacy_noise = (
            legacy_rng.normal(size=shape)
            * np.asarray(config.noise_sigma)[None, None, :]
        )
        expected = np.clip(
            prior.mean[None, :, :] + legacy_noise,
            action.lower,
            action.upper,
        )
        expected[0] = prior.mean

        current_rng = np.random.RandomState(seed)
        actual = controller._sample(prior, rng=current_rng)

        assert np.array_equal(actual, expected)
        assert np.array_equal(actual[0], prior.mean)
        assert np.all(actual >= action.lower[None, None, :])
        assert np.all(actual <= action.upper[None, None, :])
        assert np.array_equal(
            current_rng.normal(size=(4, 3)),
            legacy_rng.normal(size=(4, 3)),
        )

    @staticmethod
    def _paper_gaussian_harness(noise_basis):
        controller = object.__new__(PaperRLDrivenMppiController)
        controller.config = MppiConfig(
            horizon=6,
            num_samples=20,
            dt=0.1,
            noise_sigma=(0.08, 0.20),
            noise_basis=noise_basis,
        )
        controller.action_spec = ActionSpec(
            ("v_cmd", "omega_cmd"),
            lower=np.array([-0.15, -1.0]),
            upper=np.array([0.45, 1.0]),
            rate_limits=np.array([0.6, 2.0]),
        )
        controller.previous_action = np.array([0.10, -0.10])
        return controller

    def test_paper_gaussian_iid_path_is_bit_exact_and_rng_exact(self):
        controller = self._paper_gaussian_harness("iid")
        mean = np.column_stack(
            (
                np.linspace(0.10, 0.30, controller.config.horizon),
                np.linspace(-0.20, 0.20, controller.config.horizon),
            )
        )
        variance = np.broadcast_to(
            np.array([0.08, 0.20])[None, :] ** 2,
            mean.shape,
        ).copy()
        count = 17
        seed = 991

        legacy_rng = np.random.RandomState(seed)
        expected = mean[None, :, :] + legacy_rng.normal(
            size=(count,) + mean.shape
        ) * np.sqrt(variance)[None, :, :]
        expected = np.clip(
            expected, controller.action_spec.lower, controller.action_spec.upper
        )
        expected[0] = mean
        previous = np.repeat(
            controller.previous_action.reshape(1, -1), count, axis=0
        )
        for step in range(controller.config.horizon):
            expected[:, step, :] = controller.action_spec.clip(
                expected[:, step, :],
                previous=previous,
                dt=controller.config.dt,
            )
            previous = expected[:, step, :]

        current_rng = np.random.RandomState(seed)
        actual = controller._gaussian_samples(
            mean, variance, count, current_rng
        )

        assert np.array_equal(actual, expected)
        assert np.array_equal(
            current_rng.normal(size=(4, 3)),
            legacy_rng.normal(size=(4, 3)),
        )
        assert (
            controller._last_gaussian_rate_limit_audit[
                "post_limit_violation_fraction"
            ]
            == 0.0
        )

    def test_paper_gaussian_ar1_changes_only_temporal_basis(self):
        mean = np.zeros((36, 2), dtype=np.float64)
        variance = np.broadcast_to(SIGMA[None, :] ** 2, mean.shape).copy()
        iid = self._paper_gaussian_harness("iid")
        ar1 = self._paper_gaussian_harness("ar1:2.0")
        iid.config = MppiConfig(
            horizon=36,
            num_samples=400,
            dt=0.1,
            noise_sigma=tuple(SIGMA),
            noise_basis="iid",
        )
        ar1.config = MppiConfig(
            horizon=36,
            num_samples=400,
            dt=0.1,
            noise_sigma=tuple(SIGMA),
            noise_basis="ar1:2.0",
        )
        iid.action_spec = body_velocity_action((-10.0, 10.0), 10.0)
        ar1.action_spec = iid.action_spec
        iid.previous_action = np.zeros(2)
        ar1.previous_action = np.zeros(2)
        iid_values = iid._gaussian_samples(
            mean, variance, 400, np.random.RandomState(5)
        )
        ar1_values = ar1._gaussian_samples(
            mean, variance, 400, np.random.RandomState(5)
        )

        def lag1(values):
            return float(np.corrcoef(
                values[:, :-1, :].ravel(),
                values[:, 1:, :].ravel(),
            )[0, 1])

        assert abs(lag1(iid_values[1:])) < 0.05
        assert lag1(ar1_values[1:]) > 0.9


class TestMarginalVariancePreserved:
    """Correlated bases must not smuggle in extra magnitude.

    If they did, any reach or coverage comparison against i.i.d. would be
    confounded with amplitude rather than measuring temporal structure.
    """

    @pytest.mark.parametrize(
        "spec",
        ["iid", "piecewise:3", "piecewise:4", "piecewise:8", "ar1:0.5", "ar1:1.0", "ar1:2.0"],
    )
    def test_marginal_variance_within_tolerance(self, spec):
        report = verify_marginal_variance(
            build_basis(spec), SIGMA, HORIZON, count=40_000, seed=3
        )
        assert report["passes"], (
            f"{spec} deviates {report['max_relative_deviation']:.4f} "
            f"from the production sigma"
        )


class TestCorrelationStructure:
    def test_piecewise_is_constant_within_segments(self):
        noise = PiecewiseConstantBasis(segments=4).sample(
            np.random.default_rng(0), 16, 36, SIGMA
        )
        edges = np.rint(np.linspace(0, 36, 5)).astype(int)
        for start, stop in zip(edges[:-1], edges[1:]):
            block = noise[:, start:stop, :]
            assert np.allclose(block, block[:, :1, :]), (
                "piecewise basis varies inside a segment"
            )

    def test_ar1_is_more_correlated_than_iid(self):
        def lag1(noise):
            a = noise[:, :-1, :].ravel()
            b = noise[:, 1:, :].ravel()
            return float(np.corrcoef(a, b)[0, 1])

        rng = np.random.default_rng(5)
        iid = lag1(IidBasis().sample(rng, 400, HORIZON, SIGMA))
        ar1 = lag1(
            Ar1Basis(correlation_time_s=2.0, dt=0.1).sample(
                np.random.default_rng(5), 400, HORIZON, SIGMA
            )
        )
        assert abs(iid) < 0.05, f"i.i.d. lag-1 correlation should be ~0, got {iid}"
        assert ar1 > 0.9, f"ar1 tau=2.0s lag-1 correlation should be high, got {ar1}"

    def test_ar1_rho_matches_correlation_time(self):
        basis = Ar1Basis(correlation_time_s=2.0, dt=0.1)
        assert basis.rho == pytest.approx(np.exp(-0.05), rel=1e-12)


class TestSpecParsing:
    @pytest.mark.parametrize(
        "spec", ["", "gaussian", "ar1", "piecewise", "ar1:0", "ar1:-1", "piecewise:0", "pw4"]
    )
    def test_invalid_specs_raise(self, spec):
        with pytest.raises(ValueError):
            build_basis(spec)

    def test_shape_and_finiteness(self):
        for spec in ("iid", "piecewise:4", "ar1:1.0"):
            noise = build_basis(spec).sample(
                np.random.default_rng(2), 32, HORIZON, SIGMA
            )
            assert noise.shape == (32, HORIZON, 2)
            assert np.isfinite(noise).all()


class TestMppiConfigIntegration:
    def test_config_defaults_to_iid(self):
        assert MppiConfig().noise_basis == "iid"

    def test_from_mapping_defaults_to_iid(self):
        assert MppiConfig.from_mapping({}, 2).noise_basis == "iid"

    def test_from_mapping_accepts_override(self):
        config = MppiConfig.from_mapping({"noise_basis": "ar1:2.0"}, 2)
        assert config.noise_basis == "ar1:2.0"

    def test_validate_rejects_malformed_basis(self):
        with pytest.raises(ValueError):
            MppiConfig(noise_basis="nonsense").validate(2)

    def test_validate_accepts_supported_bases(self):
        for spec in ("iid", "piecewise:4", "ar1:2.0"):
            MppiConfig(noise_basis=spec).validate(2)
