from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Ensure the repo root is importable so we can load the custom environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.env import PandaObstacleEnv  # noqa: E402

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - matplotlib should be available but fail gracefully
    raise RuntimeError("matplotlib is required for visualisations") from exc

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None

ALGORITHMS: Dict[str, Callable] = {"ppo": PPO, "sac": SAC}


class TrainingVisualizationCallback(BaseCallback):
    """Collect per-episode statistics and generate visualisations after training."""

    def __init__(self, log_dir: Path, success_window: int = 50, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.success_window = success_window

        self.episode_rewards: List[float] = []
        self.episode_success: List[float] = []
        self.episode_avg_distance: List[float] = []
        self._actions: List[np.ndarray] = []

        self._episode_reward: np.ndarray | None = None
        self._episode_distance: np.ndarray | None = None
        self._episode_length: np.ndarray | None = None

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        self._episode_reward = np.zeros(num_envs, dtype=np.float64)
        self._episode_distance = np.zeros(num_envs, dtype=np.float64)
        self._episode_length = np.zeros(num_envs, dtype=np.int32)

    def _on_rollout_end(self) -> None:
        # nothing to aggregate here but keep hook for completeness
        return None

    def _on_step(self) -> bool:
        if self._episode_reward is None or self._episode_distance is None or self._episode_length is None:
            return True

        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones")
        actions = self.locals.get("actions")

        if rewards is not None:
            self._episode_reward += rewards
        if actions is not None:
            self._actions.append(np.array(actions, copy=True))

        if infos:
            for idx, info in enumerate(infos):
                distance_value = info.get("distance_to_goal")
                if distance_value is not None:
                    self._episode_distance[idx] += float(np.asarray(distance_value).squeeze())
                self._episode_length[idx] += 1

        if dones is not None and np.any(dones):
            for idx, done in enumerate(dones):
                if not done:
                    continue
                info: Dict[str, np.ndarray] = infos[idx] if idx < len(infos) else {}

                reward_ep = float(self._episode_reward[idx])
                steps_ep = int(self._episode_length[idx])
                avg_distance = (
                    float(self._episode_distance[idx] / steps_ep) if steps_ep > 0 else float("nan")
                )

                success_flag = self._extract_success(info)

                self.episode_rewards.append(reward_ep)
                self.episode_success.append(success_flag)
                self.episode_avg_distance.append(avg_distance)

                if self.logger is not None:
                    self.logger.record("rollout/episode_reward", reward_ep)
                    self.logger.record("rollout/episode_success", success_flag)
                    self.logger.record("rollout/episode_avg_distance", avg_distance)
                    window_success = float(
                        np.mean(self.episode_success[-self.success_window :])
                    )
                    self.logger.record("rollout/success_rate_window", window_success)

                self._episode_reward[idx] = 0.0
                self._episode_distance[idx] = 0.0
                self._episode_length[idx] = 0

        return True

    def _extract_success(self, info: Dict[str, np.ndarray]) -> float:
        if "is_success" in info:
            return float(np.asarray(info["is_success"]).astype(np.float32).squeeze())

        collided = info.get("collided")
        collided_flag = bool(np.asarray(collided).squeeze()) if collided is not None else False
        time_limit = bool(info.get("TimeLimit.truncated", False))

        return 0.0 if collided_flag or time_limit else 1.0

    def _on_training_end(self) -> None:
        if not self.episode_rewards:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        episodes = np.arange(1, len(self.episode_rewards) + 1)
        self._plot_line(
            episodes,
            self.episode_rewards,
            title="Episode Return",
            ylabel="Return",
            file_path=self.log_dir / "episode_return.png",
        )

        cumulative_success = np.cumsum(self.episode_success) / np.arange(1, len(self.episode_success) + 1)
        self._plot_line(
            episodes,
            cumulative_success,
            title="Success Rate",
            ylabel="Success Rate",
            file_path=self.log_dir / "success_rate.png",
            ylim=(0.0, 1.05),
        )

        self._plot_line(
            episodes,
            self.episode_avg_distance,
            title="Average Distance To Goal",
            ylabel="Distance (m)",
            file_path=self.log_dir / "avg_distance.png",
        )

        self._plot_action_distribution()

    def _plot_line(
        self,
        x_values: np.ndarray,
        y_values: List[float] | np.ndarray,
        *,
        title: str,
        ylabel: str,
        file_path: Path,
        ylim: tuple[float, float] | None = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_values, y_values, linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(file_path)
        plt.close(fig)

    def _plot_action_distribution(self) -> None:
        if not self._actions:
            return

        actions = np.concatenate(self._actions, axis=0)
        if actions.ndim == 1:
            actions = actions[:, None]

        action_dim = actions.shape[1]
        fig, axes = plt.subplots(action_dim, 1, figsize=(6, 2.5 * action_dim), sharex=True)
        if action_dim == 1:
            axes = [axes]

        for idx, ax in enumerate(axes):
            ax.hist(actions[:, idx], bins=40, color="tab:blue", alpha=0.7)
            ax.set_ylabel(f"Action {idx}")
            ax.grid(True, linestyle="--", alpha=0.3)

        axes[-1].set_xlabel("Action Value")
        fig.suptitle("Action Distribution")
        fig.tight_layout()
        fig.savefig(self.log_dir / "action_distribution.png")
        plt.close(fig)


def record_policy_rollout(
    model,
    env_factory: Callable[[], PandaObstacleEnv],
    video_path: Path,
    *,
    episodes: int = 1,
    deterministic: bool = True,
) -> None:
    if imageio is None:
        print("imageio not available, skipping policy rollout video generation.")
        return

    env = env_factory()
    fps = env.metadata.get("render_fps", 30) if hasattr(env, "metadata") else 30
    frames: List[np.ndarray] = []

    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            step_count = 0
            max_steps = getattr(env, "max_episode_steps", 1000)

            while not (terminated or truncated) and step_count < max_steps:
                frame = env.render()
                frames.append(frame)
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _, terminated, truncated, _ = env.step(action)
                step_count += 1

            if terminated or truncated:
                frame = env.render()
                frames.append(frame)

    finally:
        env.close()

    if not frames:
        print("No frames captured for policy rollout video.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Saved policy rollout video to {video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Panda obstacle avoidance policy with SB3.")
    parser.add_argument("--algo", choices=ALGORITHMS.keys(), default="ppo", help="RL algorithm to use")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Number of training steps")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel vectorized environments")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory for SB3 logs")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to save models")
    parser.add_argument("--tensorboard", type=Path, default=Path("tb_logs"), help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Environment steps between checkpoints")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Environment steps between policy evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation run")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy class name for the algorithm")
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=1,
        help="Number of episodes to record for the post-training policy rollout video",
    )
    return parser.parse_args()


def make_env(seed: int) -> Callable[[], PandaObstacleEnv]:
    def _init() -> PandaObstacleEnv:
        env = PandaObstacleEnv()
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    args = parse_args()

    algo_key = args.algo.lower()
    algo_cls = ALGORITHMS[algo_key]

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_env(args.seed + i) for i in range(args.num_envs)])
    vec_env = VecMonitor(vec_env, filename=str(args.log_dir / "monitor.csv"))

    eval_env = DummyVecEnv([make_env(args.seed + 10_000)])
    eval_env = VecMonitor(eval_env)

    visualization_dir = args.log_dir / "visuals"
    visualization_callback = TrainingVisualizationCallback(visualization_dir)

    checkpoint_callback = None
    if args.checkpoint_freq > 0:
        checkpoint_dir = args.model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.num_envs),
            save_path=str(checkpoint_dir),
            name_prefix=f"{algo_key}_panda",
        )

    eval_callback = None
    if args.eval_freq > 0:
        best_model_dir = args.model_dir / "best"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(args.log_dir / "eval"),
            eval_freq=max(1, args.eval_freq // args.num_envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        )

    callbacks = [cb for cb in (checkpoint_callback, eval_callback, visualization_callback) if cb is not None]

    model = algo_cls(
        args.policy,
        vec_env,
        verbose=1,
        tensorboard_log=str(args.tensorboard),
        seed=args.seed,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks if callbacks else None)

    model_path = args.model_dir / f"{algo_key}_panda_final"
    model.save(str(model_path))

    rollout_env_factory = make_env(args.seed + 20_000)
    video_path = visualization_dir / "policy_rollout.gif"
    record_policy_rollout(
        model,
        rollout_env_factory,
        video_path,
        episodes=max(1, args.rollout_episodes),
    )

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
