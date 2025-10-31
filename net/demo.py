"""Forward/backward demo for the actor-critic network in the Panda obstacle env."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Ensure project root is on sys.path so that local package imports work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.env import PandaObstacleEnv
from net.net import MLPConfig, build_actor_critic_for_env


def _format_tensor(tensor: torch.Tensor) -> str:
    """Return a short string summarizing basic tensor statistics."""
    std = tensor.std(unbiased=False) if tensor.numel() > 1 else torch.tensor(0.0)
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, mean={tensor.mean():.4f}, std={std:.4f}"


def main() -> None:
    """Demonstrate a single forward and backward pass through the actor-critic."""
    env = PandaObstacleEnv()
    try:
        # Reset the environment to obtain a deterministic initial observation.
        obs, _ = env.reset(seed=0)

        # Build the actor-critic MLP with the default config and switch to eval for inference.
        config = MLPConfig(activation="tanh")
        model = build_actor_critic_for_env(env.observation_space, env.action_space, config)
        model.eval()

        # Convert the observation into a batch-1 tensor for the network.
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Forward pass: compute action distribution statistics and value estimate.
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

        # Backward pass: switch to train mode, build a dummy loss, and take gradients.
        model.train()
        model.zero_grad(set_to_none=True)
        mean_train, log_std_train, value_train = model(obs_tensor)
        loss = (
            mean_train.pow(2).mean()
            + log_std_train.pow(2).mean()
            + value_train.pow(2).mean()
        )
        loss.backward()

        print("Backward pass check:")
        print(f"  Dummy loss scalar   : {loss.item():.6f}")
        print(f"  Policy weight grad  : {_format_tensor(model.policy_head.weight.grad)}")
        print(f"  Value weight grad   : {_format_tensor(model.value_head.weight.grad)}")
        print(f"  Log-std grad        : {_format_tensor(model.log_std.grad)}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
