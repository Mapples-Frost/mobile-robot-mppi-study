"""Self-contained Soft Actor-Critic for continuous MPPI-prior parameters."""

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

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
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
        )

    def validate(self):
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
            1,
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
        self.config.validate()
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

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, observation, deterministic=False):
        data = np.asarray(observation, dtype=np.float32)
        if data.shape != (self.observation_dim,) or not np.isfinite(data).all():
            raise ValueError("SAC observation must be a finite vector")
        with torch.no_grad():
            tensor = torch.as_tensor(data, device=self.device).unsqueeze(0)
            action, _, mean_action, log_std = self.actor.sample(tensor, deterministic=deterministic)
        selected = action if not deterministic else mean_action
        return selected[0].cpu().numpy().astype(np.float32), {
            "policy_log_std_mean": float(log_std.mean().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }

    def critic_disagreement(self, observation, action):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        with torch.no_grad():
            q1 = self.critic1(observation, action)
            q2 = self.critic2(observation, action)
        return float(torch.abs(q1 - q2).mean().cpu())

    def _clip(self, module):
        return float(nn.utils.clip_grad_norm_(module.parameters(), self.config.gradient_clip_norm))

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
        log_std_weight = float(log_std_weight)
        target_log_std = float(target_log_std)
        if (
            not np.isfinite(log_std_weight)
            or log_std_weight < 0.0
            or not np.isfinite(target_log_std)
        ):
            raise ValueError("BC log-std loss settings are invalid")
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

    def update(self, batch):
        observation = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_observation = torch.as_tensor(batch["next_observations"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_action, next_log_probability, _, _ = self.actor.sample(next_observation)
            next_q = torch.minimum(
                self.target_critic1(next_observation, next_action),
                self.target_critic2(next_observation, next_action),
            )
            target = reward + (1.0 - done) * self.config.gamma * (
                next_q - self.alpha.detach() * next_log_probability
            )
        q1 = self.critic1(observation, action)
        q2 = self.critic2(observation, action)
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
        policy_action, log_probability, _, _ = self.actor.sample(observation)
        policy_q = torch.minimum(
            self.critic1(observation, policy_action),
            self.critic2(observation, policy_action),
        )
        actor_loss = (self.alpha.detach() * log_probability - policy_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_gradient = self._clip(self.actor)
        self.actor_optimizer.step()
        alpha_loss = torch.zeros((), device=self.device)
        if self.config.automatic_entropy_tuning:
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
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "target_q_mean": float(target.mean().detach().cpu()),
            "entropy": float((-log_probability).mean().detach().cpu()),
            "actor_gradient_norm": actor_gradient,
            "critic1_gradient_norm": critic1_gradient,
            "critic2_gradient_norm": critic2_gradient,
        }
        if not np.isfinite(tuple(values.values())).all():
            raise FloatingPointError("SAC update produced NaN or Inf")
        return values

    def state_dict(self):
        return {
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "config": self.config.to_dict(),
            "actor": self.actor.state_dict(),
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
        }

    def load_state_dict(self, state, load_optimizers=True):
        if int(state["observation_dim"]) != self.observation_dim or int(state["action_dim"]) != self.action_dim:
            raise ValueError("SAC checkpoint dimensions do not match agent")
        self.actor.load_state_dict(state["actor"])
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

    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
