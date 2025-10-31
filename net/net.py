from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import gymnasium as gym
import torch
import torch.nn as nn


ActivationFactory = Callable[[], nn.Module]


def _tanh_factory() -> nn.Module:
    return nn.Tanh()


DEFAULT_ACTIVATIONS: dict[str, ActivationFactory] = {
    "tanh": _tanh_factory,
}


@dataclass(slots=True)
class MLPConfig:
    """Configuration for the shared trunk of the policy/value networks."""

    hidden_sizes: Sequence[int] = field(default_factory=lambda: (64, 64))
    activation: str = "tanh"
    ortho_init: bool = True
    log_std_init: float = -0.5

    def activation_factory(self) -> ActivationFactory:
        if self.activation.lower() not in DEFAULT_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. "
                f"Available: {', '.join(DEFAULT_ACTIVATIONS)}"
            )
        return DEFAULT_ACTIVATIONS[self.activation.lower()]


def _init_layer(layer: nn.Linear, *, gain: float = 1.0, ortho_init: bool = True) -> None:
    if ortho_init:
        nn.init.orthogonal_(layer.weight, gain=gain)
    else:
        nn.init.xavier_uniform_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def _flatten_obs_space(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        if space.dtype.kind not in {"f", "i"}:
            raise ValueError("Unsupported Box dtype for observations; expected float or int.")
        return int(torch.tensor(space.shape).prod().item())
    raise TypeError(f"Unsupported observation space type {type(space).__name__}")


def _infer_action_dim(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        if space.shape is None:
            raise ValueError("Box action space shape is undefined.")
        return int(torch.tensor(space.shape).prod().item())
    raise TypeError(f"Unsupported action space type {type(space).__name__}")


class ActorCriticMLP(nn.Module):
    """Actor-Critic network with a shared trunk and separate policy/value heads."""

    def __init__(self, obs_dim: int, action_dim: int, config: MLPConfig | None = None) -> None:
        super().__init__()
        self.config = config or MLPConfig()

        layers: list[nn.Module] = []
        last_dim = obs_dim
        activation_factory = self.config.activation_factory()
        hidden_gain = nn.init.calculate_gain(self.config.activation)
        for hidden_size in self.config.hidden_sizes:
            layer = nn.Linear(last_dim, hidden_size)
            _init_layer(layer, gain=hidden_gain, ortho_init=self.config.ortho_init)
            layers.append(layer)
            layers.append(activation_factory())
            last_dim = hidden_size

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()

        self.policy_head = nn.Linear(last_dim, action_dim)
        _init_layer(self.policy_head, gain=0.01, ortho_init=self.config.ortho_init)

        self.value_head = nn.Linear(last_dim, 1)
        _init_layer(self.value_head, gain=1.0, ortho_init=self.config.ortho_init)

        self.log_std = nn.Parameter(torch.full((action_dim,), self.config.log_std_init))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() > 2:
            obs = obs.view(obs.size(0), -1)

        features = self.feature_extractor(obs)
        mean_actions = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        log_std = self.log_std.expand_as(mean_actions)
        return mean_actions, log_std, values

    def get_action_distribution(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, log_std, values = self(obs)
        std = log_std.exp().clamp(min=1e-6)
        distribution = torch.distributions.Normal(mean, std)
        return distribution, values


def build_actor_critic_for_env(
    observation_space: gym.Space,
    action_space: gym.Space,
    config: MLPConfig | None = None,
) -> ActorCriticMLP:
    obs_dim = _flatten_obs_space(observation_space)
    action_dim = _infer_action_dim(action_space)
    return ActorCriticMLP(obs_dim, action_dim, config=config)
