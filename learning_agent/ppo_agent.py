from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Generator, NamedTuple, TYPE_CHECKING
from pathlib import Path

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from net import ActorCriticMLP, MLPConfig, build_actor_critic_for_env

if TYPE_CHECKING:
    import gymnasium as gym

__all__ = [
    "PPOConfig",
    "RolloutBatch",
    "RolloutBuffer",
    "PPOAgent",
    "generalized_advantage_estimation",
]


@dataclass(slots=True)
class PPOConfig:
    """Hyperparameters for the PPO clipped objective."""

    total_batch_size: int = 2048
    mini_batch_size: int = 64
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_range: float = 0.2
    value_clip_range: float | None = None
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    adam_eps: float = 1e-5


class RolloutBatch(NamedTuple):
    """Container for a mini-batch sampled from the rollout buffer."""

    observations: Tensor
    actions: Tensor
    old_values: Tensor
    old_log_probs: Tensor
    returns: Tensor
    advantages: Tensor


def generalized_advantage_estimation(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    last_done: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE advantages and bootstrapped returns."""
    rewards = rewards.view(-1).to(values.device, dtype=torch.float32)
    values = values.view(-1).to(values.device, dtype=torch.float32)
    dones = dones.view(-1).to(values.device, dtype=torch.float32)
    last_value = last_value.view(-1).to(values.device, dtype=torch.float32)
    last_done = last_done.view(-1).to(values.device, dtype=torch.float32)

    advantages = torch.zeros_like(values)
    returns = torch.zeros_like(values)

    next_value = last_value[-1]
    next_non_terminal = 1.0 - last_done[-1]
    gae = torch.zeros_like(next_value)

    # Walk backwards so each step can reuse the bootstrap from the next timestep.
    for step in range(rewards.shape[0] - 1, -1, -1):
        # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        # Recursively accumulate the exponentially-weighted advantage estimates.
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae
        returns[step] = gae + values[step]

        next_value = values[step]
        next_non_terminal = 1.0 - dones[step]

    return advantages, returns


class RolloutBuffer:
    """Simple rollout buffer that supports GAE computation and mini-batch sampling."""

    def __init__(self, gamma: float, gae_lambda: float, device: torch.device | str) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)
        self._tensor_ready = False
        self.reset()

    def reset(self) -> None:
        # Until finalised, data is stored as Python lists for cheap appends.
        self.observations: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.dones: list[Tensor] = []
        self.values: list[Tensor] = []
        self.log_probs: list[Tensor] = []
        self.advantages: Tensor | list[Tensor] = []
        self.returns: Tensor | list[Tensor] = []
        self._tensor_ready = False

    def add(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float,
        done: bool,
        value: Tensor,
        log_prob: Tensor,
    ) -> None:
        """Append a single transition to the buffer."""
        if self._tensor_ready:
            raise RuntimeError("RolloutBuffer already finalised—call reset() before adding new samples.")

        # Keep tensors on the target device to avoid later host→device transfers.
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=self.device).reshape(-1)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=self.device).reshape(-1)

        value_tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).reshape(-1).detach()
        log_prob_tensor = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device).reshape(-1).detach()

        self.observations.append(obs_tensor)
        self.actions.append(action_tensor)
        self.rewards.append(reward_tensor)
        self.dones.append(done_tensor)
        self.values.append(value_tensor)
        self.log_probs.append(log_prob_tensor)

    def compute_returns_and_advantages(self, last_value: Tensor, last_done: bool | Tensor) -> None:
        """Finalize the buffer by computing GAE advantages and returns."""
        if isinstance(last_done, bool):
            last_done_tensor = torch.tensor(float(last_done), device=self.device)
        else:
            last_done_tensor = torch.as_tensor(last_done, dtype=torch.float32, device=self.device)
        last_value_tensor = torch.as_tensor(last_value, dtype=torch.float32, device=self.device)

        # Stack per-transition tensors into flat vectors for vectorised GAE.
        rewards = torch.stack(self.rewards).view(-1)
        values = torch.stack(self.values).view(-1)
        dones = torch.stack(self.dones).view(-1)

        advantages, returns = generalized_advantage_estimation(
            rewards,
            values,
            dones,
            last_value_tensor.view(-1),
            last_done_tensor.view(-1),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # From this point on we keep contiguous tensors for fast indexing.
        self.observations = torch.stack(self.observations)
        self.actions = torch.stack(self.actions)
        self.rewards = rewards
        self.dones = dones
        self.values = values
        self.log_probs = torch.stack(self.log_probs).view(-1)
        self.advantages = advantages
        self.returns = returns
        self._tensor_ready = True

    def normalize_advantages(self) -> None:
        """Normalize advantages to have zero mean and unit variance."""
        if not self._tensor_ready:
            raise RuntimeError("Compute advantages before normalising them.")
        advantages = torch.as_tensor(self.advantages, device=self.device)
        # Normalising tends to stabilise optimisation when reward scales drift.
        std = advantages.std(unbiased=False)
        if torch.isfinite(std) and std > 0.0:
            self.advantages = (advantages - advantages.mean()) / (std + 1e-8)

    def get(self, batch_size: int, shuffle: bool = True) -> Generator[RolloutBatch, None, None]:
        """Yield mini-batches of rollout data."""
        if not self._tensor_ready:
            raise RuntimeError("Rollout buffer not ready—call compute_returns_and_advantages first.")

        total = len(self)
        indices = torch.arange(total, device=self.device)
        if shuffle:
            # PPO relies on IID mini-batches, so reshuffle between epochs.
            indices = indices[torch.randperm(total, device=self.device)]

        for start in range(0, total, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield RolloutBatch(
                observations=self.observations[batch_idx],
                actions=self.actions[batch_idx],
                old_values=self.values[batch_idx],
                old_log_probs=self.log_probs[batch_idx],
                returns=self.returns[batch_idx],
                advantages=self.advantages[batch_idx],
            )

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rewards)

    @property
    def ready(self) -> bool:
        return self._tensor_ready


class PPOAgent:
    """Implementation of PPO with the clipped surrogate objective."""

    def __init__(
        self,
        observation_space: "gym.Space",
        action_space: "gym.Space",
        config: PPOConfig | None = None,
        mlp_config: MLPConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(device)

        self.policy: ActorCriticMLP = build_actor_critic_for_env(
            observation_space,
            action_space,
            config=mlp_config,
        )
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate, eps=self.config.adam_eps)
        self.rollout_buffer = RolloutBuffer(self.config.gamma, self.config.gae_lambda, self.device)

    def act(self, observation: Tensor, deterministic: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Sample an action from the current policy along with its log-probability and value estimate."""
        self.policy.eval()
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            # Most gym envs emit 1-D observations; add batch dim expected by network.
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            distribution, values = self.policy.get_action_distribution(obs_tensor)
            if deterministic:
                actions = distribution.mean
            else:
                actions = distribution.sample()
            log_probs = distribution.log_prob(actions).sum(dim=-1)

        return actions.squeeze(0), log_probs.squeeze(0), values.squeeze(0)

    def store_transition(
        self,
        observation: Tensor,
        action: Tensor,
        reward: float,
        done: bool,
        value: Tensor,
        log_prob: Tensor,
    ) -> None:
        """Push a transition into the internal rollout buffer."""
        self.rollout_buffer.add(observation, action, reward, done, value, log_prob)

    def prepare_update(self, last_value: Tensor, last_done: bool | Tensor) -> None:
        """Finalize the rollout buffer in preparation for a policy update."""
        self.rollout_buffer.compute_returns_and_advantages(last_value, last_done)
        if self.config.normalize_advantage:
            # Normalised advantages behave like a centered gradient estimator.
            self.rollout_buffer.normalize_advantages()

    def update(self, rollout_buffer: RolloutBuffer | None = None) -> dict[str, float]:
        """Perform multiple epochs of PPO updates over the collected rollout data."""
        buffer = rollout_buffer or self.rollout_buffer
        if not buffer.ready:
            raise RuntimeError("Rollout buffer is not ready—call prepare_update() before update().")

        self.policy.train()

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        clip_fraction_sum = 0.0
        batches_processed = 0

        for _ in range(self.config.update_epochs):
            for batch in buffer.get(self.config.mini_batch_size):
                distribution, values = self.policy.get_action_distribution(batch.observations)
                values = values.view(-1)

                log_probs = distribution.log_prob(batch.actions).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1).mean()

                # Importance-sampling ratio between new and behaviour policies.
                ratios = torch.exp(log_probs - batch.old_log_probs)
                surr1 = ratios * batch.advantages
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.config.value_clip_range is not None:
                    values_clipped = batch.old_values + (values - batch.old_values).clamp(
                        -self.config.value_clip_range,
                        self.config.value_clip_range,
                    )
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (values_clipped - batch.returns).pow(2)
                    # Take the worst-case (clipped vs unclipped) to discourage large updates.
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (batch.returns - values).pow(2).mean()

                # Standard PPO objective: policy term + scaled value loss − entropy bonus.
                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Monitor KL and clipping statistics for debugging or early stopping.
                    approx_kl = (batch.old_log_probs - log_probs).mean()
                    clip_fraction = (ratios - 1.0).abs() > self.config.clip_range

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.item()
                kl_sum += approx_kl.item()
                clip_fraction_sum += clip_fraction.float().mean().item()
                batches_processed += 1

        metrics = {
            "policy_loss": policy_loss_sum / batches_processed,
            "value_loss": value_loss_sum / batches_processed,
            "entropy": entropy_sum / batches_processed,
            "approx_kl": kl_sum / batches_processed,
            "clip_fraction": clip_fraction_sum / batches_processed,
        }

        if rollout_buffer is None:
            buffer.reset()

        return metrics

    def to(self, device: torch.device | str) -> None:
        """Move the policy network (and future rollouts) to the target device."""
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.rollout_buffer.device = self.device
