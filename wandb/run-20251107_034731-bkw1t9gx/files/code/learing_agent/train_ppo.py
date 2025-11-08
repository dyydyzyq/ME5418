from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.env import PandaObstacleEnv
from learing_agent.ppo import PPOAgent, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent in the Panda obstacle environment.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Number of environment steps to collect.")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Steps per policy update.")
    parser.add_argument("--mini-batch-size", type=int, default=64, help="Mini-batch size for PPO updates.")
    parser.add_argument("--update-epochs", type=int, default=10, help="How many epochs to iterate over the rollout buffer.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping range.")
    parser.add_argument(
        "--value-clip-range",
        type=float,
        default=None,
        help="Optional value function clip range. Leave unset to disable value clipping.",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--device", type=str, default="auto", help="Training device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--project", type=str, default="panda-ppo", help="Weights & Biases project name.")
    parser.add_argument("--entity", type=str, default=None, help="Weights & Biases entity (team) name.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional custom run name for Weights & Biases.")
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle logging to Weights & Biases.",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="How many updates between console logs.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def init_wandb(args: argparse.Namespace, config: Dict[str, Any]) -> Any | None:
    if not args.track:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError("wandb is not installed. Install it with `pip install wandb` or disable tracking.") from exc

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config,
        sync_tensorboard=False,
        save_code=True,
        mode="online",
    )
    return run


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    env = PandaObstacleEnv()
    obs, _ = env.reset(seed=args.seed)

    ppo_config = PPOConfig(
        total_batch_size=args.rollout_length,
        mini_batch_size=args.mini_batch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_clip_range=args.value_clip_range,
        entropy_coef=args.entropy_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )
    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        config=ppo_config,
        device=device,
    )

    run = init_wandb(args, config={**asdict(ppo_config), "device": str(device), "seed": args.seed})

    total_timesteps = args.total_timesteps
    rollout_length = args.rollout_length
    num_updates = max(1, total_timesteps // rollout_length)
    global_step = 0
    start_time = time.perf_counter()

    episode_return = 0.0
    episode_length = 0

    for update_idx in range(1, num_updates + 1):
        next_done = torch.zeros(1, device=device)
        for step in range(rollout_length):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, log_prob, value = agent.act(obs_tensor)

            action_env = action.detach().cpu().numpy()
            action_env = np.clip(action_env, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = bool(terminated or truncated)

            agent.store_transition(
                obs_tensor.detach(),
                torch.as_tensor(action, dtype=torch.float32, device=device).detach(),
                float(reward),
                done,
                value.detach(),
                log_prob.detach(),
            )

            episode_return += reward
            episode_length += 1
            global_step += 1

            if done:
                if run is not None:
                    run.log(
                        {
                            "train/episode_return": episode_return,
                            "train/episode_length": episode_length,
                        },
                        step=global_step,
                    )
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                obs = next_obs

            next_done = torch.tensor(float(done), device=device)

        if next_done.item() < 0.5:
            with torch.no_grad():
                last_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                _, _, last_value = agent.act(last_obs_tensor)
        else:
            last_value = torch.zeros(1, device=device)

        agent.prepare_update(last_value.detach(), next_done)
        metrics = agent.update()

        if run is not None:
            run.log(
                {
                    "train/policy_loss": metrics["policy_loss"],
                    "train/value_loss": metrics["value_loss"],
                    "train/entropy": metrics["entropy"],
                    "train/approx_kl": metrics["approx_kl"],
                    "train/clip_fraction": metrics["clip_fraction"],
                    "train/learning_rate": args.learning_rate,
                },
                step=global_step,
            )

        if update_idx % args.log_interval == 0 or update_idx == num_updates:
            elapsed = time.perf_counter() - start_time
            sps = global_step / max(elapsed, 1e-8)
            print(
                f"[Update {update_idx:04d}/{num_updates}] "
                f"Step={global_step} "
                f"PolicyLoss={metrics['policy_loss']:.4f} "
                f"ValueLoss={metrics['value_loss']:.4f} "
                f"Entropy={metrics['entropy']:.4f} "
                f"KL={metrics['approx_kl']:.5f} "
                f"SPS={sps:.1f}"
            )

    env.close()
    if run is not None:
        run.finish()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover
    main()
