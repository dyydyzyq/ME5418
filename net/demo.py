"""Simple sanity check for the ActorCriticMLP network.

This script instantiates the PandaObstacleEnv, builds the actor-critic MLP,
and performs a single forward pass plus an action sample to verify the wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.env import PandaObstacleEnv
from net.net import MLPConfig, build_actor_critic_for_env


def _format_tensor(tensor: torch.Tensor) -> str:
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, mean={tensor.mean():.4f}, std={std:.4f}"


def main() -> None:
    env = PandaObstacleEnv()
    try:
        obs, _ = env.reset(seed=0)

        config = MLPConfig(activation="tanh")
        model = build_actor_critic_for_env(env.observation_space, env.action_space, config)
        model.eval()

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, log_std, value = model(obs_tensor)
            dist, value_again = model.get_action_distribution(obs_tensor)
            action_sample = dist.sample()

        print("Network check complete:")
        print(f"  Observation tensor  : {_format_tensor(obs_tensor)}")
        print(f"  Mean action output  : {_format_tensor(mean)}")
        print(f"  Log-std output      : {_format_tensor(log_std)}")
        print(f"  Value output        : {_format_tensor(value)}")
        print(f"  Resampled value     : {_format_tensor(value_again)}")
        print(f"  Sampled action      : {_format_tensor(action_sample)}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
