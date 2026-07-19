"""Self-contained Soft Actor-Critic for continuous MPPI-prior parameters."""

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _activation(name):
    choices = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "softplus": nn.Softplus,
        "silu": nn.SiLU,
    }
    key = str(name).lower()
    if key not in choices:
        raise ValueError("unsupported SAC activation: %s" % name)
    return choices[key]


def _mlp(input_dim, output_dim, hidden_sizes, activation):
    layers = []
    previous = int(input_dim)
    layer_type = _activation(activation)
    for width in hidden_sizes:
        width = int(width)
        if width <= 0:
            raise ValueError("SAC hidden sizes must be positive")
        linear = nn.Linear(previous, width)
        nn.init.orthogonal_(linear.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, layer_type()))
        previous = width
    output = nn.Linear(previous, int(output_dim))
    nn.init.uniform_(output.weight, -3e-3, 3e-3)
    nn.init.uniform_(output.bias, -3e-3, 3e-3)
    layers.append(output)
    return nn.Sequential(*layers)


def group_robust_mean(per_sample_loss, groups, temperature):
    """Return a smooth worst-scene aggregation of per-sample actor losses."""

    if per_sample_loss.ndim == 2 and per_sample_loss.shape[1] == 1:
        per_sample_loss = per_sample_loss[:, 0]
    if per_sample_loss.ndim != 1:
        raise ValueError("group-robust loss must contain one scalar per sample")
    groups = groups.reshape(-1)
    if groups.shape[0] != per_sample_loss.shape[0]:
        raise ValueError("group labels must match the actor-loss batch")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError(
            "group-robust temperature must be finite and non-negative"
        )
    unique_groups = torch.unique(groups, sorted=True)
    if unique_groups.numel() == 0:
        raise ValueError("group-robust actor loss requires a nonempty batch")
    group_losses = torch.stack([
        per_sample_loss[groups == group].mean()
        for group in unique_groups
    ])
    if temperature == 0.0:
        robust = torch.max(group_losses)
    else:
        robust = temperature * (
            torch.logsumexp(group_losses / temperature, dim=0)
            - np.log(float(group_losses.numel()))
        )
    return robust, group_losses


def quantile_huber_loss(predictions, targets, kappa=1.0):
    """Pairwise quantile-regression Huber loss for distributional critics."""

    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("quantile predictions and targets must be [batch, N]")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("quantile predictions and targets must share a batch")
    if min(predictions.shape[1], targets.shape[1]) <= 0:
        raise ValueError("quantile tensors must be nonempty")
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("quantile Huber kappa must be finite and positive")
    delta = targets.unsqueeze(1) - predictions.unsqueeze(2)
    absolute = torch.abs(delta)
    huber = torch.where(
        absolute <= kappa,
        0.5 * delta ** 2,
        kappa * (absolute - 0.5 * kappa),
    )
    count = predictions.shape[1]
    taus = (
        (torch.arange(count, device=predictions.device, dtype=predictions.dtype) + 0.5)
        / float(count)
    ).view(1, count, 1)
    weights = torch.abs(
        taus - (delta.detach() < 0.0).to(predictions.dtype)
    )
    return torch.mean(weights * huber / kappa)


def lower_tail_cvar(quantiles, fraction):
    """Average the lowest requested fraction of a quantile distribution."""

    if quantiles.ndim < 1 or quantiles.shape[-1] <= 0:
        raise ValueError("CVaR requires a nonempty quantile dimension")
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must lie in (0, 1]")
    count = max(1, int(np.ceil(quantiles.shape[-1] * fraction)))
    ordered = torch.sort(quantiles, dim=-1).values
    return ordered[..., :count].mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class SACConfig:
    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "relu"
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    minimum_alpha: float = 0.0
    automatic_entropy_tuning: bool = True
    target_entropy: Optional[float] = None
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    gradient_clip_norm: float = 10.0
    policy_mode: str = "direct"
    correction_scale: Tuple[float, ...] = (0.20,)
    correction_gate_alpha: float = 1.0
    correction_initial_log_std: float = -2.5
    correction_penalty_weight: float = 0.0
    actor_group_robust_enabled: bool = False
    actor_group_robust_temperature: float = 0.10
    critic_distribution: str = "scalar"
    critic_num_quantiles: int = 25
    critic_quantile_huber_kappa: float = 1.0
    actor_cvar_fraction: float = 1.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        raw_correction_scale = values.get("correction_scale", (0.20,))
        if isinstance(raw_correction_scale, (int, float)):
            correction_scale = (float(raw_correction_scale),)
        else:
            correction_scale = tuple(float(value) for value in raw_correction_scale)
        return cls(
            hidden_sizes=tuple(int(value) for value in values.get("hidden_sizes", (256, 256))),
            activation=str(values.get("activation", "relu")),
            gamma=float(values.get("gamma", 0.99)),
            tau=float(values.get("tau", 0.005)),
            actor_lr=float(values.get("actor_lr", 3e-4)),
            critic_lr=float(values.get("critic_lr", 3e-4)),
            alpha_lr=float(values.get("alpha_lr", 3e-4)),
            initial_alpha=float(values.get("initial_alpha", 0.2)),
            minimum_alpha=float(values.get("minimum_alpha", 0.0)),
            automatic_entropy_tuning=bool(values.get("automatic_entropy_tuning", True)),
            target_entropy=(
                None if values.get("target_entropy") is None else float(values["target_entropy"])
            ),
            log_std_min=float(values.get("log_std_min", -5.0)),
            log_std_max=float(values.get("log_std_max", 1.0)),
            gradient_clip_norm=float(values.get("gradient_clip_norm", 10.0)),
            policy_mode=str(values.get("policy_mode", "direct")),
            correction_scale=correction_scale,
            correction_gate_alpha=float(values.get("correction_gate_alpha", 1.0)),
            correction_initial_log_std=float(
                values.get("correction_initial_log_std", -2.5)
            ),
            correction_penalty_weight=float(
                values.get("correction_penalty_weight", 0.0)
            ),
            actor_group_robust_enabled=bool(
                values.get("actor_group_robust_enabled", False)
            ),
            actor_group_robust_temperature=float(
                values.get("actor_group_robust_temperature", 0.10)
            ),
            critic_distribution=str(
                values.get("critic_distribution", "scalar")
            ),
            critic_num_quantiles=int(
                values.get("critic_num_quantiles", 25)
            ),
            critic_quantile_huber_kappa=float(
                values.get("critic_quantile_huber_kappa", 1.0)
            ),
            actor_cvar_fraction=float(
                values.get("actor_cvar_fraction", 1.0)
            ),
        )

    def validate(self, action_dim=None):
        if not self.hidden_sizes:
            raise ValueError("SAC requires at least one hidden layer")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("SAC gamma/tau are out of range")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.initial_alpha) <= 0.0:
            raise ValueError("SAC learning rates and alpha must be positive")
        if not 0.0 <= self.minimum_alpha <= self.initial_alpha:
            raise ValueError("SAC minimum_alpha must be in [0, initial_alpha]")
        if self.log_std_min >= self.log_std_max or self.gradient_clip_norm <= 0.0:
            raise ValueError("invalid SAC log-std or gradient clipping limits")
        if self.policy_mode not in ("direct", "frozen_bc_correction"):
            raise ValueError(
                "SAC policy_mode must be direct or frozen_bc_correction"
            )
        correction_scale = np.asarray(self.correction_scale, dtype=np.float64)
        if (
            correction_scale.ndim != 1
            or correction_scale.size == 0
            or not np.isfinite(correction_scale).all()
            or np.any(correction_scale <= 0.0)
            or np.any(correction_scale > 1.0)
        ):
            raise ValueError(
                "SAC correction_scale must contain finite values in (0, 1]"
            )
        if action_dim is not None and correction_scale.size not in (1, int(action_dim)):
            raise ValueError(
                "SAC correction_scale must be scalar or match action_dim"
            )
        if not 0.0 < self.correction_gate_alpha <= 1.0:
            raise ValueError("SAC correction_gate_alpha must be in (0, 1]")
        if not (
            np.isfinite(self.correction_initial_log_std)
            and self.log_std_min
            <= self.correction_initial_log_std
            <= self.log_std_max
        ):
            raise ValueError(
                "SAC correction_initial_log_std must lie within log-std bounds"
            )
        if (
            not np.isfinite(self.correction_penalty_weight)
            or self.correction_penalty_weight < 0.0
        ):
            raise ValueError(
                "SAC correction_penalty_weight must be finite and non-negative"
            )
        if self.policy_mode == "direct" and self.correction_penalty_weight > 0.0:
            raise ValueError(
                "SAC correction penalty requires frozen_bc_correction mode"
            )
        if (
            not np.isfinite(self.actor_group_robust_temperature)
            or self.actor_group_robust_temperature < 0.0
        ):
            raise ValueError(
                "SAC actor group-robust temperature must be finite and "
                "non-negative"
            )
        if self.critic_distribution not in ("scalar", "quantile"):
            raise ValueError(
                "SAC critic_distribution must be scalar or quantile"
            )
        if (
            self.critic_distribution == "quantile"
            and self.critic_num_quantiles < 2
        ):
            raise ValueError("SAC quantile critic requires at least 2 quantiles")
        if self.critic_num_quantiles <= 0:
            raise ValueError("SAC critic_num_quantiles must be positive")
        if (
            not np.isfinite(self.critic_quantile_huber_kappa)
            or self.critic_quantile_huber_kappa <= 0.0
        ):
            raise ValueError(
                "SAC critic quantile Huber kappa must be finite and positive"
            )
        if (
            not np.isfinite(self.actor_cvar_fraction)
            or not 0.0 < self.actor_cvar_fraction <= 1.0
        ):
            raise ValueError("SAC actor CVaR fraction must lie in (0, 1]")
        _activation(self.activation)

    def to_dict(self):
        return asdict(self)


class SquashedGaussianActor(nn.Module):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(config.log_std_min)
        self.log_std_max = float(config.log_std_max)
        self.network = _mlp(
            observation_dim,
            2 * self.action_dim,
            config.hidden_sizes,
            config.activation,
        )

    def distribution(self, observation):
        output = self.network(observation)
        mean, log_std = torch.chunk(output, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def mean_action(self, observation):
        """Return the deterministic bounded action used by BC and inference."""

        mean, _ = self.distribution(observation)
        return torch.tanh(mean)

    def sample(self, observation, deterministic=False):
        mean, log_std = self.distribution(observation)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + torch.exp(log_std) * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if deterministic:
            log_probability = None
        else:
            variance = torch.exp(2.0 * log_std)
            gaussian_log_probability = -0.5 * (
                ((pre_tanh - mean) ** 2) / variance
                + 2.0 * log_std
                + np.log(2.0 * np.pi)
            )
            gaussian_log_probability = gaussian_log_probability.sum(dim=-1, keepdim=True)
            correction = 2.0 * (
                np.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
            )
            log_probability = gaussian_log_probability - correction.sum(dim=-1, keepdim=True)
        return action, log_probability, torch.tanh(mean), log_std


class QNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, config):
        super().__init__()
        self.network = _mlp(
            int(observation_dim) + int(action_dim),
            (
                int(config.critic_num_quantiles)
                if config.critic_distribution == "quantile"
                else 1
            ),
            config.hidden_sizes,
            config.activation,
        )

    def forward(self, observation, action):
        return self.network(torch.cat((observation, action), dim=-1))


class SACAgent:
    def __init__(self, observation_dim, action_dim, config=None, device="cpu", seed=0):
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        if min(self.observation_dim, self.action_dim) <= 0:
            raise ValueError("SAC observation/action dimensions must be positive")
        self.config = config if isinstance(config, SACConfig) else SACConfig.from_mapping(config)
        self.config.validate(action_dim=self.action_dim)
        requested_device = str(device)
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for SAC but is unavailable")
        self.device = torch.device(requested_device)
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        self.actor = SquashedGaussianActor(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.base_actor = None
        self.base_actor_initialized = False
        if self.is_correction_policy:
            self.base_actor = SquashedGaussianActor(
                self.observation_dim, self.action_dim, self.config
            ).to(self.device)
            self.base_actor.requires_grad_(False)
            self.base_actor.eval()
            self._zero_initialize_correction_actor()
        self.critic1 = QNetwork(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.critic2 = QNetwork(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.target_critic1 = QNetwork(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.target_critic2 = QNetwork(self.observation_dim, self.action_dim, self.config).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.critic_lr)
        initial_log_alpha = float(np.log(self.config.initial_alpha))
        self.log_alpha = torch.tensor(initial_log_alpha, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.config.alpha_lr)
        self.target_entropy = float(
            -self.action_dim if self.config.target_entropy is None else self.config.target_entropy
        )
        self.update_steps = 0
        self.bc_update_steps = 0
        self.bc_anchor_update_steps = 0

    @property
    def is_correction_policy(self):
        return self.config.policy_mode == "frozen_bc_correction"

    @property
    def is_quantile_critic(self):
        return self.config.critic_distribution == "quantile"

    def _critic_expectation(self, values):
        return values.mean(dim=-1, keepdim=True)

    def _critic_risk_value(self, values):
        fraction = (
            self.config.actor_cvar_fraction
            if self.is_quantile_critic
            else 1.0
        )
        return lower_tail_cvar(values, fraction)

    def _expanded_correction_scale(self):
        values = tuple(float(value) for value in self.config.correction_scale)
        if len(values) == 1:
            values = values * self.action_dim
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _zero_initialize_correction_actor(self):
        """Make the deterministic residual exactly zero before RL updates."""

        output = self.actor.network[-1]
        if not isinstance(output, nn.Linear) or output.out_features != 2 * self.action_dim:
            raise RuntimeError("unexpected correction actor output layer")
        with torch.no_grad():
            output.weight.zero_()
            output.bias.zero_()
            output.bias[self.action_dim:].fill_(
                float(self.config.correction_initial_log_std)
            )

    def initialize_frozen_base_actor(self, actor_state):
        """Load an immutable BC policy and reset the trainable residual head.

        Only the deterministic BC mean is used as the base action.  The source
        actor's stochastic log-standard-deviation is retained in the audit
        checkpoint but does not inject noise into the frozen base policy.
        """

        if not self.is_correction_policy:
            raise ValueError(
                "frozen base actor initialization requires correction policy mode"
            )
        self.base_actor.load_state_dict(actor_state)
        self.base_actor.requires_grad_(False)
        self.base_actor.eval()
        self.base_actor_initialized = True
        self._zero_initialize_correction_actor()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

    def _require_correction_base(self):
        if self.is_correction_policy and not self.base_actor_initialized:
            raise RuntimeError(
                "frozen BC correction policy requires a BC initialization "
                "checkpoint before inference or training"
            )

    def frozen_base_actor_sha256(self):
        if not self.is_correction_policy:
            return None
        self._require_correction_base()
        digest = hashlib.sha256()
        for name, value in sorted(self.base_actor.state_dict().items()):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _compose_correction(self, base_action, correction_action):
        """Apply a state-dependent, exactly bounded normalized correction.

        Positive and negative corrections use only the available room between
        the frozen BC action and the normalized action-space boundary.  This
        avoids an unreported post-hoc clip and provides the exact Jacobian used
        by the transformed SAC log probability.
        """

        scale = self._expanded_correction_scale().view(
            *((1,) * (base_action.ndim - 1)), self.action_dim
        )
        positive_room = torch.minimum(
            scale, torch.clamp(1.0 - base_action, min=0.0)
        )
        negative_room = torch.minimum(
            scale, torch.clamp(1.0 + base_action, min=0.0)
        )
        room = torch.where(correction_action >= 0.0, positive_room, negative_room)
        jacobian = float(self.config.correction_gate_alpha) * room
        applied = jacobian * correction_action
        final_action = base_action + applied
        # Numerical roundoff is the only possible source of an out-of-range
        # value because ``room`` already respects both action-space boundaries.
        final_action = torch.clamp(final_action, -1.0, 1.0)
        return final_action, applied, jacobian

    def _sample_policy(self, observation, deterministic=False):
        if not self.is_correction_policy:
            action, log_probability, mean_action, log_std = self.actor.sample(
                observation, deterministic=deterministic
            )
            zeros = torch.zeros_like(action)
            diagnostics = {
                "base_action": zeros,
                "unit_correction": zeros,
                "applied_correction": zeros,
            }
            return action, log_probability, mean_action, log_std, diagnostics

        self._require_correction_base()
        with torch.no_grad():
            base_action = self.base_actor.mean_action(observation)
        (
            unit_correction,
            correction_log_probability,
            mean_unit_correction,
            log_std,
        ) = self.actor.sample(observation, deterministic=deterministic)
        action, applied, jacobian = self._compose_correction(
            base_action, unit_correction
        )
        mean_action, mean_applied, _ = self._compose_correction(
            base_action, mean_unit_correction
        )
        log_probability = correction_log_probability
        if log_probability is not None:
            # The residual action has a state-dependent support.  Account for
            # the piecewise-linear residual-to-final-action transform rather
            # than treating a clipped action as if it came from plain SAC.
            epsilon = torch.finfo(jacobian.dtype).eps
            log_probability = log_probability - torch.log(
                torch.clamp(jacobian, min=epsilon)
            ).sum(dim=-1, keepdim=True)
        diagnostics = {
            "base_action": base_action,
            "unit_correction": unit_correction,
            "applied_correction": applied,
            "mean_applied_correction": mean_applied,
        }
        return action, log_probability, mean_action, log_std, diagnostics

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(
        self, observation, deterministic=False, include_internal=False
    ):
        data = np.asarray(observation, dtype=np.float32)
        if data.shape != (self.observation_dim,) or not np.isfinite(data).all():
            raise ValueError("SAC observation must be a finite vector")
        with torch.no_grad():
            tensor = torch.as_tensor(data, device=self.device).unsqueeze(0)
            action, _, mean_action, log_std, policy = self._sample_policy(
                tensor, deterministic=deterministic
            )
        selected = action if not deterministic else mean_action
        diagnostics = {
            "policy_log_std_mean": float(log_std.mean().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "policy_mode": self.config.policy_mode,
            "correction_gate_alpha": (
                float(self.config.correction_gate_alpha)
                if self.is_correction_policy else 0.0
            ),
            "base_action_abs_mean": float(
                policy["base_action"].abs().mean().cpu()
            ),
            "unit_correction_abs_mean": float(
                policy["unit_correction"].abs().mean().cpu()
            ),
            "applied_correction_abs_mean": float(
                policy["applied_correction"].abs().mean().cpu()
            ),
            "applied_correction_abs_max": float(
                policy["applied_correction"].abs().max().cpu()
            ),
        }
        if include_internal:
            diagnostics["_base_action"] = (
                policy["base_action"][0].cpu().numpy().astype(np.float32)
            )
        return selected[0].cpu().numpy().astype(np.float32), diagnostics

    def select_action_batch(self, observations, deterministic=True):
        """Evaluate the policy for a finite observation batch.

        This inference-only API is used by RL-Driven MPPI to evaluate terminal
        states without a Python loop.  It preserves correction-policy
        composition and never updates actor, critic, or optimizer state.
        """

        data = np.asarray(observations, dtype=np.float32)
        expected_tail = (self.observation_dim,)
        if (
            data.ndim != 2
            or data.shape[1:] != expected_tail
            or data.shape[0] <= 0
            or not np.isfinite(data).all()
        ):
            raise ValueError(
                "SAC observation batch must be finite with shape [B,%d]"
                % self.observation_dim
            )
        with torch.no_grad():
            tensor = torch.as_tensor(data, device=self.device)
            action, _, mean_action, _, _ = self._sample_policy(
                tensor, deterministic=deterministic
            )
            selected = mean_action if deterministic else action
        if not bool(torch.isfinite(selected).all()):
            raise FloatingPointError("SAC batched policy produced NaN or Inf")
        return selected.cpu().numpy().astype(np.float32)

    def policy_gaussian_parameters_batch(self, observations):
        """Return an auditable Gaussian approximation of the physical Actor.

        RL-Driven MPPI needs both the policy mean and its stochastic spread to
        initialize a control-sequence distribution.  Returning the exact
        pre-tanh parameters lets the caller use its own seeded generator while
        preserving the SAC actor's bounded-action transform. For a frozen-base
        correction policy, the deterministic composed physical-control mean is
        exact. Deployment deliberately retains the frozen base policy's
        first-order post-tanh spread, so residual conditioning changes only the
        MPPI proposal mean. This approximation is used only as proposal
        covariance, never as a plant or value-model assumption.
        """
        data = np.asarray(observations, dtype=np.float32)
        if (
            data.ndim != 2
            or data.shape[1:] != (self.observation_dim,)
            or data.shape[0] <= 0
            or not np.isfinite(data).all()
        ):
            raise ValueError(
                "SAC observation batch must be finite with shape [B,%d]"
                % self.observation_dim
            )
        with torch.no_grad():
            tensor = torch.as_tensor(data, device=self.device)
            if self.is_correction_policy:
                self._require_correction_base()
                base_pre_tanh, base_log_std = self.base_actor.distribution(
                    tensor
                )
                base_mean = torch.tanh(base_pre_tanh)
                correction_mean, correction_log_std = self.actor.distribution(
                    tensor
                )
                unit_mean = torch.tanh(correction_mean)
                final_mean, _, jacobian = self._compose_correction(
                    base_mean, unit_mean
                )
                # The correction changes only the proposal mean. Preserve the
                # frozen L185 Actor's post-tanh spread exactly so the MPPI
                # ablation does not confound residual conditioning with a
                # covariance change. The correction log-std remains part of
                # SAC training but does not silently alter deployment sampling.
                del correction_log_std, jacobian
                final_std = torch.clamp(
                    (1.0 - base_mean ** 2) * torch.exp(base_log_std),
                    min=1e-6,
                )
                bounded_mean = torch.clamp(
                    final_mean, -1.0 + 1e-6, 1.0 - 1e-6
                )
                mean = torch.atanh(bounded_mean)
                log_std = torch.log(
                    final_std / torch.clamp(
                        1.0 - bounded_mean ** 2, min=1e-6
                    )
                )
            else:
                mean, log_std = self.actor.distribution(tensor)
        if not bool(torch.isfinite(mean).all() and torch.isfinite(log_std).all()):
            raise FloatingPointError(
                "SAC policy Gaussian parameters contain NaN or Inf"
            )
        return (
            mean.cpu().numpy().astype(np.float64),
            log_std.cpu().numpy().astype(np.float64),
        )

    def expected_twin_q(self, observations, actions, critic_source="online"):
        """Return expected twin-Q values for scalar or quantile critics.

        Quantile critics are averaged over their return distribution here.
        Risk-sensitive lower-tail aggregation is intentionally a separate
        ablation rather than being silently mixed into the simple baseline.
        """

        observation = np.asarray(observations, dtype=np.float32)
        action = np.asarray(actions, dtype=np.float32)
        if (
            observation.ndim != 2
            or observation.shape[1:] != (self.observation_dim,)
            or observation.shape[0] <= 0
            or not np.isfinite(observation).all()
        ):
            raise ValueError("critic observations must have shape [B, observation_dim]")
        if (
            action.ndim != 2
            or action.shape != (observation.shape[0], self.action_dim)
            or not np.isfinite(action).all()
        ):
            raise ValueError("critic actions must have shape [B, action_dim]")
        source = str(critic_source)
        if source == "online":
            critic1, critic2 = self.critic1, self.critic2
        elif source == "target":
            critic1, critic2 = self.target_critic1, self.target_critic2
        else:
            raise ValueError("critic_source must be 'online' or 'target'")
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            )
            action_tensor = torch.as_tensor(
                action, dtype=torch.float32, device=self.device
            )
            q1 = self._critic_expectation(
                critic1(observation_tensor, action_tensor)
            ).reshape(-1)
            q2 = self._critic_expectation(
                critic2(observation_tensor, action_tensor)
            ).reshape(-1)
        if not bool(torch.isfinite(q1).all() and torch.isfinite(q2).all()):
            raise FloatingPointError("SAC critic produced NaN or Inf")
        q1_array = q1.cpu().numpy().astype(np.float64)
        q2_array = q2.cpu().numpy().astype(np.float64)
        return {
            "q1": q1_array,
            "q2": q2_array,
            "minimum": np.minimum(q1_array, q2_array),
            "mean": 0.5 * (q1_array + q2_array),
            "disagreement": np.abs(q1_array - q2_array),
            "critic_source": source,
        }

    def critic_disagreement(self, observation, action):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        with torch.no_grad():
            q1 = self._critic_risk_value(
                self.critic1(observation, action)
            )
            q2 = self._critic_risk_value(
                self.critic2(observation, action)
            )
        return float(torch.abs(q1 - q2).mean().cpu())

    def frozen_base_action(self, observation):
        """Return the deterministic frozen-BC latent action for one state."""

        if not self.is_correction_policy:
            raise ValueError(
                "frozen base action requires frozen_bc_correction mode"
            )
        self._require_correction_base()
        observation_array = np.asarray(observation, dtype=np.float32)
        if (
            observation_array.shape != (self.observation_dim,)
            or not np.isfinite(observation_array).all()
        ):
            raise ValueError("frozen base observation is invalid")
        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation_array, device=self.device
            ).unsqueeze(0)
            action = self.base_actor.mean_action(observation_tensor)
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError("frozen base action produced NaN or Inf")
        return action[0].cpu().numpy().astype(np.float32)

    def filter_correction_by_advantage(
        self,
        observation,
        candidate_action,
        gate_mode="none",
        critic_source="online",
        threshold=0.0,
        uncertainty_multiplier=1.0,
        compute_unselected_diagnostics=True,
        base_action=None,
    ):
        """Diagnose and optionally reject a deterministic correction.

        The comparison is made in the normalized latent action space consumed
        by the MPPI-prior decoder.  For each twin critic we subtract the value
        of the frozen BC action from the value of the candidate correction and
        then use the smaller difference as the conservative advantage.  A hard
        rejection recovers the *frozen BC action*, not the unrelated
        GoalWarmStart fallback used by the outer deployment gate.

        ``lcb`` mode additionally requires agreement between the two action
        advantages.  Its score is the twin mean minus ``beta`` times their
        half-disagreement.  It is scale equivariant and beta=1 is exactly the
        previous conservative minimum.  This is an engineering lower-bound
        score, not a calibrated probability or statistical confidence bound.
        """

        if not self.is_correction_policy:
            raise ValueError(
                "correction advantage requires frozen_bc_correction mode"
            )
        self._require_correction_base()
        gate_mode = str(gate_mode)
        critic_source = str(critic_source)
        threshold = float(threshold)
        uncertainty_multiplier = float(uncertainty_multiplier)
        compute_unselected_diagnostics = bool(
            compute_unselected_diagnostics
        )
        if gate_mode not in ("none", "hard", "lcb", "base"):
            raise ValueError(
                "correction advantage gate mode must be none, hard, lcb or base"
            )
        if critic_source not in ("online", "target"):
            raise ValueError(
                "correction advantage critic source must be online or target"
            )
        if not np.isfinite(threshold):
            raise ValueError("correction advantage threshold must be finite")
        if (
            not np.isfinite(uncertainty_multiplier)
            or uncertainty_multiplier < 1.0
        ):
            raise ValueError(
                "correction advantage uncertainty multiplier must be finite "
                "and at least one"
            )

        observation_array = np.asarray(observation, dtype=np.float32)
        candidate_array = np.asarray(candidate_action, dtype=np.float32)
        if (
            observation_array.shape != (self.observation_dim,)
            or not np.isfinite(observation_array).all()
        ):
            raise ValueError("correction advantage observation is invalid")
        if (
            candidate_array.shape != (self.action_dim,)
            or not np.isfinite(candidate_array).all()
            or np.any(candidate_array < -1.000001)
            or np.any(candidate_array > 1.000001)
        ):
            raise ValueError("correction advantage candidate action is invalid")
        if base_action is not None:
            base_array = np.asarray(base_action, dtype=np.float32)
            if (
                base_array.shape != (self.action_dim,)
                or not np.isfinite(base_array).all()
                or np.any(base_array < -1.000001)
                or np.any(base_array > 1.000001)
            ):
                raise ValueError("correction advantage base action is invalid")
        else:
            base_array = None

        with torch.no_grad():
            observation_tensor = torch.as_tensor(
                observation_array, device=self.device
            ).unsqueeze(0)
            candidate_tensor = torch.as_tensor(
                candidate_array, device=self.device
            ).unsqueeze(0)
            base_tensor = (
                self.base_actor.mean_action(observation_tensor)
                if base_array is None
                else torch.as_tensor(
                    base_array, device=self.device
                ).unsqueeze(0)
            )

            def evaluate_pair(first, second):
                q1_base = self._critic_risk_value(
                    first(observation_tensor, base_tensor)
                )
                q2_base = self._critic_risk_value(
                    second(observation_tensor, base_tensor)
                )
                q1_candidate = self._critic_risk_value(
                    first(observation_tensor, candidate_tensor)
                )
                q2_candidate = self._critic_risk_value(
                    second(observation_tensor, candidate_tensor)
                )
                advantages = torch.cat((
                    q1_candidate - q1_base,
                    q2_candidate - q2_base,
                ), dim=-1)
                conservative = torch.min(
                    advantages, dim=-1
                ).values
                mean = torch.mean(advantages, dim=-1)
                half_disagreement = 0.5 * torch.abs(
                    advantages[:, 0] - advantages[:, 1]
                )
                consensus_lcb = mean - (
                    uncertainty_multiplier * half_disagreement
                )
                return {
                    "q1_base": q1_base,
                    "q2_base": q2_base,
                    "q1_candidate": q1_candidate,
                    "q2_candidate": q2_candidate,
                    "advantages": advantages,
                    "conservative": conservative,
                    "mean": mean,
                    "half_disagreement": half_disagreement,
                    "consensus_lcb": consensus_lcb,
                }

            online = None
            target = None
            if compute_unselected_diagnostics or critic_source == "online":
                online = evaluate_pair(self.critic1, self.critic2)
            if compute_unselected_diagnostics or critic_source == "target":
                target = evaluate_pair(
                    self.target_critic1, self.target_critic2
                )
            selected = online if critic_source == "online" else target
            selected_advantage = selected["conservative"]
            selected_consensus_lcb = selected["consensus_lcb"]
            if gate_mode == "none":
                gate_alpha = 1.0
            elif gate_mode == "base":
                # Exact frozen-base ablation: no arbitrary rejection threshold
                # and no authority from either critic.
                gate_alpha = 0.0
            elif gate_mode == "hard":
                gate_alpha = float(selected_advantage.item() >= threshold)
            else:
                gate_alpha = float(selected_consensus_lcb.item() >= 0.0)
            gated_tensor = base_tensor + gate_alpha * (
                candidate_tensor - base_tensor
            )
            raw_correction = candidate_tensor - base_tensor
            gated_correction = gated_tensor - base_tensor

        scalar_tensors = [gated_tensor]
        for values in (online, target):
            if values is not None:
                scalar_tensors.extend(values.values())
        if not all(bool(torch.isfinite(value).all()) for value in scalar_tensors):
            raise FloatingPointError(
                "correction advantage diagnostic produced NaN or Inf"
            )
        diagnostics = {
            "correction_advantage_gate_mode": gate_mode,
            "correction_advantage_critic_source": critic_source,
            "correction_advantage_threshold": threshold,
            "correction_advantage_uncertainty_multiplier": (
                uncertainty_multiplier
            ),
            "critic_quantile_count": float(
                self.config.critic_num_quantiles
                if self.is_quantile_critic else 1
            ),
            "critic_cvar_fraction": float(
                self.config.actor_cvar_fraction
                if self.is_quantile_critic else 1.0
            ),
            "correction_advantage_gate_alpha": gate_alpha,
            "unselected_critic_diagnostics_computed": (
                compute_unselected_diagnostics
            ),
            "selected_consensus_lcb": float(
                selected_consensus_lcb.item()
            ),
            "raw_applied_correction_abs_mean": float(
                raw_correction.abs().mean().item()
            ),
            "raw_applied_correction_abs_max": float(
                raw_correction.abs().max().item()
            ),
            # Overwrite the existing policy diagnostics with the correction
            # that is actually sent to the prior decoder after the gate.
            "applied_correction_abs_mean": float(
                gated_correction.abs().mean().item()
            ),
            "applied_correction_abs_max": float(
                gated_correction.abs().max().item()
            ),
        }
        for prefix, values in (("online", online), ("target", target)):
            if values is None:
                continue
            diagnostics.update({
                "%s_q1_base" % prefix: float(
                    values["q1_base"].item()
                ),
                "%s_q2_base" % prefix: float(
                    values["q2_base"].item()
                ),
                "%s_q1_candidate" % prefix: float(
                    values["q1_candidate"].item()
                ),
                "%s_q2_candidate" % prefix: float(
                    values["q2_candidate"].item()
                ),
                "%s_advantage_q1" % prefix: float(
                    values["advantages"][0, 0].item()
                ),
                "%s_advantage_q2" % prefix: float(
                    values["advantages"][0, 1].item()
                ),
                "%s_conservative_advantage" % prefix: float(
                    values["conservative"].item()
                ),
                "%s_advantage_mean" % prefix: float(
                    values["mean"].item()
                ),
                "%s_advantage_half_disagreement" % prefix: float(
                    values["half_disagreement"].item()
                ),
                "%s_consensus_lcb" % prefix: float(
                    values["consensus_lcb"].item()
                ),
            })
        return (
            gated_tensor[0].cpu().numpy().astype(np.float32),
            diagnostics,
        )

    def _clip(self, module):
        return float(nn.utils.clip_grad_norm_(module.parameters(), self.config.gradient_clip_norm))

    def _behavior_cloning_losses(self, observations, expert_actions, target_log_std):
        observation = np.asarray(observations, dtype=np.float32)
        target = np.asarray(expert_actions, dtype=np.float32)
        if (
            observation.ndim != 2
            or observation.shape[0] == 0
            or observation.shape[1] != self.observation_dim
            or not np.isfinite(observation).all()
        ):
            raise ValueError(
                "BC observations must be finite [batch, observation_dim]"
            )
        if (
            target.ndim != 2
            or target.shape != (observation.shape[0], self.action_dim)
            or not np.isfinite(target).all()
        ):
            raise ValueError("BC expert actions must be finite [batch, action_dim]")
        if np.any(target < -1.000001) or np.any(target > 1.000001):
            raise ValueError("BC expert actions must lie in normalized [-1, 1]")
        target_log_std = float(target_log_std)
        if not np.isfinite(target_log_std):
            raise ValueError("BC target_log_std must be finite")
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        target_tensor = torch.as_tensor(
            target, dtype=torch.float32, device=self.device
        )
        mean, log_std = self.actor.distribution(observation_tensor)
        predicted = torch.tanh(mean)
        mean_loss = F.mse_loss(predicted, target_tensor)
        log_std_loss = F.mse_loss(
            log_std, torch.full_like(log_std, target_log_std)
        )
        return mean_loss, log_std_loss, log_std

    def behavior_cloning_update(
        self,
        observations,
        expert_actions,
        log_std_weight=0.0,
        target_log_std=-2.0,
    ):
        """Fit only the actor mean to normalized expert prior parameters.

        The expert target is the actor's own bounded parameter space, not an
        executed ``(v, omega)`` command.  Critics, target critics and entropy
        temperature are deliberately untouched so a BC checkpoint can later
        initialize a fresh, causally isolated SAC run.
        """

        if self.is_correction_policy:
            raise ValueError(
                "behavior cloning trains a direct actor; correction policies "
                "must initialize their frozen base from a BC checkpoint"
            )

        log_std_weight = float(log_std_weight)
        if not np.isfinite(log_std_weight) or log_std_weight < 0.0:
            raise ValueError("BC log-std loss settings are invalid")
        mean_loss, log_std_loss, log_std = self._behavior_cloning_losses(
            observations, expert_actions, target_log_std
        )
        loss = mean_loss + log_std_weight * log_std_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = self._clip(self.actor)
        self.actor_optimizer.step()
        self.bc_update_steps += 1
        metrics = {
            "bc_loss": float(loss.detach().cpu()),
            "bc_mean_mse": float(mean_loss.detach().cpu()),
            "bc_mean_rmse": float(torch.sqrt(mean_loss).detach().cpu()),
            "bc_log_std_loss": float(log_std_loss.detach().cpu()),
            "bc_log_std_mean": float(log_std.mean().detach().cpu()),
            "bc_gradient_norm": gradient,
        }
        if not np.isfinite(tuple(metrics.values())).all():
            raise FloatingPointError("BC actor update produced NaN or Inf")
        return metrics

    def update(
        self,
        batch,
        behavior_batch=None,
        behavior_cloning_weight=0.0,
        behavior_log_std_weight=0.0,
        behavior_target_log_std=-2.0,
        update_actor=True,
    ):
        self._require_correction_base()
        if self.is_correction_policy and (
            behavior_batch is not None
            or float(behavior_cloning_weight) > 0.0
            or float(behavior_log_std_weight) > 0.0
        ):
            raise ValueError(
                "a frozen BC correction policy cannot also use a BC actor anchor"
            )
        observation = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_observation = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_action, next_log_probability, _, _, _ = self._sample_policy(
                next_observation
            )
            next_distribution = torch.minimum(
                self.target_critic1(next_observation, next_action),
                self.target_critic2(next_observation, next_action),
            )
            target = reward + (1.0 - done) * self.config.gamma * (
                next_distribution
                - self.alpha.detach() * next_log_probability
            )
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
        if self.is_quantile_critic:
            critic1_loss = quantile_huber_loss(
                q1,
                target,
                self.config.critic_quantile_huber_kappa,
            )
            critic2_loss = quantile_huber_loss(
                q2,
                target,
                self.config.critic_quantile_huber_kappa,
            )
        else:
            critic1_loss = F.mse_loss(q1, target)
            critic2_loss = F.mse_loss(q2, target)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic1_gradient = self._clip(self.critic1)
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        critic2_gradient = self._clip(self.critic2)
        self.critic2_optimizer.step()
        (
            policy_action,
            log_probability,
            _,
            _,
            policy_diagnostics,
        ) = self._sample_policy(observation)
        policy_distribution1 = self.critic1(observation, policy_action)
        policy_distribution2 = self.critic2(observation, policy_action)
        policy_q = torch.minimum(
            self._critic_risk_value(policy_distribution1),
            self._critic_risk_value(policy_distribution2),
        )
        per_sample_sac_actor_loss = (
            self.alpha.detach() * log_probability - policy_q
        )
        sac_actor_loss = per_sample_sac_actor_loss.mean()
        correction_penalty = torch.zeros((), device=self.device)
        per_sample_correction_penalty = torch.zeros_like(
            per_sample_sac_actor_loss
        )
        if self.is_correction_policy:
            maximum = (
                self._expanded_correction_scale().view(1, -1)
                * float(self.config.correction_gate_alpha)
            )
            relative_correction = (
                policy_diagnostics["applied_correction"] / maximum
            )
            per_sample_correction_penalty = torch.mean(
                relative_correction ** 2, dim=-1, keepdim=True
            )
            correction_penalty = per_sample_correction_penalty.mean()
        per_sample_actor_loss = (
            per_sample_sac_actor_loss
            + float(self.config.correction_penalty_weight)
            * per_sample_correction_penalty
        )
        group_losses = None
        if self.config.actor_group_robust_enabled:
            if "groups" not in batch:
                raise ValueError(
                    "group-robust actor training requires replay group labels"
                )
            groups = torch.as_tensor(
                batch["groups"], dtype=torch.long, device=self.device
            )
            actor_loss, group_losses = group_robust_mean(
                per_sample_actor_loss,
                groups,
                self.config.actor_group_robust_temperature,
            )
        else:
            actor_loss = per_sample_actor_loss.mean()
        anchor_mean_loss = torch.zeros((), device=self.device)
        anchor_log_std_loss = torch.zeros((), device=self.device)
        anchor_log_std = torch.zeros((), device=self.device)
        behavior_cloning_weight = float(behavior_cloning_weight)
        behavior_log_std_weight = float(behavior_log_std_weight)
        if (
            not np.isfinite(
                (behavior_cloning_weight, behavior_log_std_weight)
            ).all()
            or behavior_cloning_weight < 0.0
            or behavior_log_std_weight < 0.0
        ):
            raise ValueError(
                "SAC behavior-anchor weights must be finite and non-negative"
            )
        if behavior_batch is not None:
            (
                anchor_mean_loss,
                anchor_log_std_loss,
                anchor_log_std_values,
            ) = self._behavior_cloning_losses(
                behavior_batch["observations"],
                behavior_batch["actions"],
                behavior_target_log_std,
            )
            anchor_log_std = anchor_log_std_values.mean()
            actor_loss = (
                actor_loss
                + behavior_cloning_weight * anchor_mean_loss
                + behavior_log_std_weight * anchor_log_std_loss
            )
            if update_actor:
                self.bc_anchor_update_steps += 1
        elif behavior_cloning_weight > 0.0 or behavior_log_std_weight > 0.0:
            raise ValueError("SAC behavior-anchor weights require behavior_batch")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_gradient = 0.0
        if update_actor:
            actor_loss.backward()
            actor_gradient = self._clip(self.actor)
            self.actor_optimizer.step()
        alpha_loss = torch.zeros((), device=self.device)
        if self.config.automatic_entropy_tuning and update_actor:
            alpha_loss = -(
                self.log_alpha * (log_probability + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.config.minimum_alpha > 0.0:
                with torch.no_grad():
                    self.log_alpha.clamp_(
                        min=float(np.log(self.config.minimum_alpha))
                    )
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic1.parameters(), self.critic1.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.tau).add_(parameter, alpha=self.config.tau)
            for target_parameter, parameter in zip(
                self.target_critic2.parameters(), self.critic2.parameters()
            ):
                target_parameter.mul_(1.0 - self.config.tau).add_(parameter, alpha=self.config.tau)
        self.update_steps += 1
        values = {
            "critic1_loss": float(critic1_loss.detach().cpu()),
            "critic2_loss": float(critic2_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "sac_actor_loss": float(sac_actor_loss.detach().cpu()),
            "correction_penalty": float(correction_penalty.detach().cpu()),
            "weighted_correction_penalty": float(
                (
                    float(self.config.correction_penalty_weight)
                    * correction_penalty
                ).detach().cpu()
            ),
            "actor_group_robust_enabled": float(
                bool(self.config.actor_group_robust_enabled)
            ),
            "actor_group_robust_loss": float(actor_loss.detach().cpu()),
            "actor_group_loss_max": float(
                (
                    group_losses.max()
                    if group_losses is not None
                    else per_sample_actor_loss.mean()
                ).detach().cpu()
            ),
            "actor_group_loss_min": float(
                (
                    group_losses.min()
                    if group_losses is not None
                    else per_sample_actor_loss.mean()
                ).detach().cpu()
            ),
            "actor_group_count": float(
                0 if group_losses is None else group_losses.numel()
            ),
            "bc_anchor_mean_mse": float(anchor_mean_loss.detach().cpu()),
            "bc_anchor_mean_rmse": float(
                torch.sqrt(anchor_mean_loss).detach().cpu()
            ),
            "bc_anchor_log_std_loss": float(anchor_log_std_loss.detach().cpu()),
            "bc_anchor_log_std_mean": float(anchor_log_std.detach().cpu()),
            "actor_update_applied": float(bool(update_actor)),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "q_mean": float(
                self._critic_expectation(
                    torch.minimum(q1, q2)
                ).mean().detach().cpu()
            ),
            "q_cvar_mean": float(
                self._critic_risk_value(
                    torch.minimum(q1, q2)
                ).mean().detach().cpu()
            ),
            "target_q_mean": float(target.mean().detach().cpu()),
            "policy_q_cvar_mean": float(policy_q.mean().detach().cpu()),
            "critic_quantile_enabled": float(bool(self.is_quantile_critic)),
            "critic_quantile_count": float(
                self.config.critic_num_quantiles
                if self.is_quantile_critic else 1
            ),
            "critic_cvar_fraction": float(
                self.config.actor_cvar_fraction
                if self.is_quantile_critic else 1.0
            ),
            "critic_quantile_spread_mean": float(
                (
                    torch.minimum(q1, q2).max(dim=-1).values
                    - torch.minimum(q1, q2).min(dim=-1).values
                ).mean().detach().cpu()
            ),
            "entropy": float((-log_probability).mean().detach().cpu()),
            "base_action_abs_mean": float(
                policy_diagnostics["base_action"].abs().mean().detach().cpu()
            ),
            "unit_correction_abs_mean": float(
                policy_diagnostics["unit_correction"].abs().mean().detach().cpu()
            ),
            "applied_correction_abs_mean": float(
                policy_diagnostics["applied_correction"].abs().mean().detach().cpu()
            ),
            "applied_correction_abs_max": float(
                policy_diagnostics["applied_correction"].abs().max().detach().cpu()
            ),
            "actor_gradient_norm": actor_gradient,
            "critic1_gradient_norm": critic1_gradient,
            "critic2_gradient_norm": critic2_gradient,
        }
        if not np.isfinite(tuple(values.values())).all():
            raise FloatingPointError("SAC update produced NaN or Inf")
        return values

    def state_dict(self):
        self._require_correction_base()
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": self.config.to_dict(),
            "actor": self.actor.state_dict(),
            "base_actor": (
                None if self.base_actor is None else self.base_actor.state_dict()
            ),
            "base_actor_initialized": bool(self.base_actor_initialized),
            "base_actor_sha256": self.frozen_base_actor_sha256(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "target_entropy": self.target_entropy,
            "update_steps": self.update_steps,
            "bc_update_steps": self.bc_update_steps,
            "bc_anchor_update_steps": self.bc_anchor_update_steps,
        }

    def load_state_dict(self, state, load_optimizers=True):
        if int(state["observation_dim"]) != self.observation_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("SAC checkpoint dimensions do not match agent")
        self.actor.load_state_dict(state["actor"])
        if self.is_correction_policy:
            base_actor_state = state.get("base_actor")
            if base_actor_state is None or not bool(
                state.get("base_actor_initialized", False)
            ):
                raise ValueError(
                    "correction policy checkpoint is missing its frozen BC base"
                )
            self.base_actor.load_state_dict(base_actor_state)
            self.base_actor.requires_grad_(False)
            self.base_actor.eval()
            self.base_actor_initialized = True
            saved_digest = state.get("base_actor_sha256")
            actual_digest = self.frozen_base_actor_sha256()
            if saved_digest is not None and str(saved_digest) != actual_digest:
                raise ValueError(
                    "correction policy frozen BC base checksum does not match"
                )
        elif state.get("base_actor") is not None:
            raise ValueError(
                "direct SAC agent cannot load a correction-policy base actor"
            )
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        self.log_alpha.data.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
            self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.target_entropy = float(state.get("target_entropy", self.target_entropy))
        self.update_steps = int(state.get("update_steps", 0))
        self.bc_update_steps = int(state.get("bc_update_steps", 0))
        self.bc_anchor_update_steps = int(state.get("bc_anchor_update_steps", 0))

    def train(self):
        self.actor.train()
        if self.base_actor is not None:
            self.base_actor.eval()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        if self.base_actor is not None:
            self.base_actor.eval()
        self.critic1.eval()
        self.critic2.eval()
