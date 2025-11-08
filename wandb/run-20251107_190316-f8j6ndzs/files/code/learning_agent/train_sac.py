"""
Soft Actor-Critic (SAC) Training Script

Implements the full SAC training loop, coordinating interactions
between the environment, agent, and replay buffer.

Training pipeline:
1. Initialize environment, agent, and replay buffer
2. Warm-up: collect random experiences
3. Main training loop:
   - Agent selects action using current policy
   - Environment steps and returns transition
   - Store transition into replay buffer
   - Sample batches and update networks
4. Periodic evaluation and checkpoint saving

Author: ME5418
"""

import argparse
import sys
from pathlib import Path
from collections import deque
import numpy as np
import torch
from typing import Optional

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.env import PandaObstacleEnv
from learning_agent.replay_buffer import ReplayBuffer
from learning_agent.sac_agent import SACAgent   


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train SAC agent in the Panda obstacle environment")

    # Training
    parser.add_argument("--total-steps", type=int, default=1_000_000, help="Total environment steps")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for all networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update rate for target networks")
    parser.add_argument("--alpha", type=float, default=0.2, help="Initial entropy coefficient")

    # Replay buffer
    parser.add_argument("--buffer-capacity", type=int, default=1_000_000, help="Replay buffer capacity")
    parser.add_argument("--min-buffer-size", type=int, default=10_000, help="Minimum samples before training starts")

    # Network
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--lstm-hidden-dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for LSTM inputs")

    # Training schedule
    parser.add_argument("--update-frequency", type=int, default=1, help="Network update frequency (steps)")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Gradient steps per update")

    # Evaluation & logging
    parser.add_argument("--eval-frequency", type=int, default=10_000, help="Evaluate every N steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--save-frequency", type=int, default=50_000, help="Save model every N steps")
    parser.add_argument("--log-frequency", type=int, default=2_000, help="Log metrics every N steps")
    parser.add_argument("--project", type=str, default="panda-sac", help="Weights & Biases project name")
    parser.add_argument("--entity", type=str, default=None, help="Weights & Biases entity (team) name")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for Weights & Biases")
    parser.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle logging to Weights & Biases",
    )

    # Paths
    parser.add_argument("--save-dir", type=Path, default=Path("agent/models"), help="Directory to save models")
    parser.add_argument("--load-path", type=Path, default=None, help="Path to pretrained model (optional)")

    # Misc
    parser.add_argument("--device", type=str, default="cpu", help="Training device: 'cpu' or 'cuda'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def init_wandb(args):
    """Initialize a Weights & Biases run if tracking is enabled."""
    if not args.track:
        return None

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb` or disable tracking via `--no-track`."
        ) from exc

    def serialize(value):
        if isinstance(value, Path):
            return str(value)
        return value

    config = {k: serialize(v) for k, v in vars(args).items() if k != "track"}
    config["algorithm"] = "SAC"

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config,
        save_code=True,
        sync_tensorboard=False,
        mode="online",
    )
    return run


def evaluate_agent(agent: SACAgent, env: PandaObstacleEnv, num_episodes: int = 10) -> dict:
    """
    Evaluate agent performance over several episodes.

    Args:
        agent: SACAgent instance
        env: environment to evaluate in
        num_episodes: number of episodes

    Returns:
        Dictionary with metrics:
        {
            'episode_rewards': list of rewards,
            'episode_lengths': list of episode lengths,
            'success_rate': success ratio,
            'mean_reward': average episode reward,
            'mean_length': average episode length,
            'std_reward': reward standard deviation
        }
    """
    episode_rewards, episode_lengths, successes = [], [], []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward, episode_length = 0, 0
        hidden_state = None
        state_sequence = deque(maxlen=agent.seq_len)
        state_sequence.append(obs)

        while not done:
            # Build LSTM sequence
            if len(state_sequence) < agent.seq_len:
                state_seq_array = np.array([obs] * agent.seq_len)
            else:
                state_seq_array = np.array(list(state_sequence))

            # Deterministic policy for evaluation
            action, hidden_state = agent.select_action(
                state_seq_array, deterministic=True, hidden_state=hidden_state
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            state_sequence.append(obs)

            if done:
                hidden_state = None

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        success = info.get("is_success", False)
        successes.append(success)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate": np.mean(successes),
        "mean_reward": np.mean(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_reward": np.std(episode_rewards),
    }


def main():
    """Main training function implementing the full SAC training loop."""
    args = parse_args()
    run = init_wandb(args)

    # Random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create directories
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    print("=" * 60)
    print("Initializing Environment")
    print("=" * 60)
    env = PandaObstacleEnv()
    env.reset(seed=args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action space: {env.action_space}")
    if run is not None:
        run.config.update(
            {"state_dim": state_dim, "action_dim": action_dim},
            allow_val_change=True,
        )

    # Initialize replay buffer
    print("\n" + "=" * 60)
    print("Initializing Replay Buffer")
    print("=" * 60)
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity, state_dim=state_dim, action_dim=action_dim
    )
    print(f"Capacity: {args.buffer_capacity:,}")
    print(f"Min size before training: {args.min_buffer_size:,}")

    # Initialize SAC agent
    print("\n" + "=" * 60)
    print("Initializing SAC Agent")
    print("=" * 60)
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=True,
        seq_len=args.seq_len,
        device=args.device,
    )

    # Load pretrained model if provided
    if args.load_path is not None and args.load_path.exists():
        print(f"\nLoading pretrained model from {args.load_path}")
        agent.load(str(args.load_path))

    # -------------------- Warm-up phase --------------------
    print("\n" + "=" * 60)
    print("Warm-up: Collecting random experience")
    print("=" * 60)
    obs, info = env.reset()
    state_sequence = deque(maxlen=agent.seq_len)
    state_sequence.append(obs)
    warmup_steps = 0

    while len(replay_buffer) < args.min_buffer_size:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.store(obs, action, reward, next_obs, done)
        obs = next_obs if not done else env.reset()[0]
        state_sequence.append(obs)
        warmup_steps += 1
        if warmup_steps % 1000 == 0:
            print(f"Warm-up progress: {len(replay_buffer):,} / {args.min_buffer_size:,}")
            if run is not None:
                run.log(
                    {"warmup/buffer_size": len(replay_buffer), "warmup/steps": warmup_steps},
                    step=warmup_steps,
                )

    print(f"Warm-up complete: {len(replay_buffer):,} transitions collected in {warmup_steps:,} steps")
    if run is not None:
        run.log(
            {"warmup/final_buffer_size": len(replay_buffer), "warmup/steps": warmup_steps},
            step=warmup_steps,
        )

    # -------------------- Training loop --------------------
    print("\n" + "=" * 60)
    print("Start Training")
    print("=" * 60)
    print(f"Total steps: {args.total_steps:,}")

    total_steps = warmup_steps
    episode_reward, episode_length, episode_count = 0, 0, 0
    obs, info = env.reset()
    state_sequence = deque(maxlen=agent.seq_len)
    state_sequence.append(obs)
    hidden_state = None

    while total_steps < args.total_steps:
        # Prepare LSTM input
        if len(state_sequence) < agent.seq_len:
            state_seq_array = np.array([obs] * agent.seq_len)
        else:
            state_seq_array = np.array(list(state_sequence))

        # Select action from policy
        action, hidden_state = agent.select_action(
            state_seq_array, deterministic=False, hidden_state=hidden_state
        )

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.store(obs, action, reward, next_obs, done)
        episode_reward += reward
        episode_length += 1
        total_steps += 1

        # Periodic updates
        if total_steps % args.update_frequency == 0:
            for _ in range(args.gradient_steps):
                losses = agent.update(replay_buffer, batch_size=args.batch_size)
                if total_steps % args.log_frequency == 0 and losses:
                    print(f"\nStep {total_steps:,} - Losses:")
                    for k, v in losses.items():
                        print(f"  {k}: {v:.6f}")
                    if run is not None:
                        wandb_metrics = {f"train/{k}": v for k, v in losses.items()}
                        wandb_metrics["train/buffer_size"] = len(replay_buffer)
                        wandb_metrics["train/total_steps"] = total_steps
                        run.log(wandb_metrics, step=total_steps)

        # Episode end
        if done:
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"\nEpisode {episode_count} - Steps: {total_steps:,}")
                print(f"  Reward: {episode_reward:.2f}")
                print(f"  Length: {episode_length}")
            if run is not None:
                run.log(
                    {
                        "train/episode_reward": episode_reward,
                        "train/episode_length": episode_length,
                        "train/episode_count": episode_count,
                    },
                    step=total_steps,
                )
            obs, info = env.reset()
            state_sequence = deque(maxlen=agent.seq_len)
            state_sequence.append(obs)
            hidden_state = None
            episode_reward, episode_length = 0, 0
        else:
            obs = next_obs
            state_sequence.append(obs)

        # Periodic evaluation
        if total_steps % args.eval_frequency == 0:
            print(f"\n{'=' * 60}")
            print(f"Evaluation at training step {total_steps:,} (after {(total_steps // args.eval_frequency) - 1} evaluations )")
            print("=" * 60)
            eval_results = evaluate_agent(agent, env, num_episodes=args.eval_episodes)
            print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Success rate: {eval_results['success_rate']:.2%}")
            print(f"Avg episode length: {eval_results['mean_length']:.1f}")
            if run is not None:
                run.log(
                    {
                        "eval/mean_reward": eval_results["mean_reward"],
                        "eval/std_reward": eval_results["std_reward"],
                        "eval/success_rate": eval_results["success_rate"],
                        "eval/mean_length": eval_results["mean_length"],
                    },
                    step=total_steps,
                )

        # Periodic checkpoint
        if total_steps % args.save_frequency == 0:
            save_path = args.save_dir / f"sac_agent_step_{total_steps}.pt"
            agent.save(str(save_path))
            print(f"\nModel saved to {save_path}")
            if run is not None:
                run.log({"checkpoint/step": total_steps}, step=total_steps)

    # Final save
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    final_save_path = args.save_dir / "sac_agent_final.pt"
    agent.save(str(final_save_path))
    print(f"Final model saved to {final_save_path}")
    env.close()
    if run is not None:
        run.log({"training/total_steps": total_steps}, step=total_steps)
        run.finish()


if __name__ == "__main__":
    main()
