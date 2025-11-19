from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

try:
    import gym_notices.notices as gym_notices
except Exception:
    gym_notices = None
else:
    # Gymnasium is already used, but SB3 still imports the legacy gym package,
    # which prints a migration notice on every import (once per subprocess).
    # Clearing the notice registry before gym is imported silences that noise.
    gym_notices.notices.clear()

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import KVWriter, configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

# Ensure the repo root is importable so we can load the custom environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.env import PandaObstacleEnv

import numpy as np
import wandb
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - matplotlib should be available but fail gracefully
    raise RuntimeError("matplotlib is required for visualisations") from exc

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None

ALGO_NAME = "sac"


class WandbOutputFormat(KVWriter):
    """SB3 logger output that forwards scalars straight to Weights & Biases."""

    def __init__(self, run: Any | None, step_key: str = "time/total_timesteps") -> None:
        self.run = run
        self.step_key = step_key

    def write(
        self,
        kvs: Dict[str, Any],
        _key_excluded: Dict[str, str] | None = None,
        step: int | None = None,
    ) -> None:
        if self.run is None or not kvs:
            return

        log_data: Dict[str, float] = {}
        step_value: int | None = step

        for key, value in kvs.items():
            if isinstance(value, tuple):
                value = value[0]
            if isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    value = float(value)
                else:
                    continue

            if isinstance(value, (int, float)):
                log_data[key] = float(value)

        if not log_data:
            return

        if step_value is None and self.step_key in log_data:
            step_value = int(log_data[self.step_key])
        elif step_value is None and "time/step" in log_data:
            step_value = int(log_data["time/step"])

        if step_value is not None:
            self.run.log(log_data, step=step_value)
        else:
            self.run.log(log_data)

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return


def parse_args() -> argparse.Namespace:  # set the parameters for training
    parser = argparse.ArgumentParser(description="Train the Panda obstacle avoidance policy with SB3 (SAC).")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (e.g., 'cpu', 'cuda', or 'auto')",
    )
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=10_000,
        help="Number of steps with random actions before training starts.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for each SAC gradient update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for rewards.")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Polyak averaging coefficient for target network updates.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="How often to update the networks (in environment steps or episodes depending on --train-freq-unit).",
    )
    parser.add_argument(
        "--train-freq-unit",
        type=str,
        choices=("step", "episode"),
        default="step",
        help="Unit for --train-freq (step or episode).",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Number of gradient steps to run per training update.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1,
        help="Number of gradient steps between target network updates.",
    )
    parser.add_argument(
        "--ent-coef",
        type=str,
        default="auto",
        help="Entropy coefficient ('auto' enables automatic entropy tuning).",
    )
    parser.add_argument(
        "--target-entropy",
        type=str,
        default="auto",
        help="Target entropy for the policy ('auto' uses the SB3 heuristic).",
    )
    parser.add_argument("--total-timesteps", type=int, default=4000_000, help="Number of training steps.")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel vectorized environments.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory for SB3 logs")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to save models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Environment steps between checkpoints")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Environment steps between policy evaluations")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes per evaluation run")
    parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy class name for the algorithm")
    parser.add_argument(
        "--policy-hidden-sizes",
        type=int,
        nargs="+",
        default=(128, 128),
        help="Hidden layer sizes for the shared feature extractor (e.g., --policy-hidden-sizes 256 128).",
    )
    parser.add_argument(
        "--policy-vf-hidden-sizes",
        "--policy-qf-hidden-sizes",
        dest="policy_qf_hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Optional critic-specific hidden sizes; provide to build independent actor/critic MLPs.",
    )
    parser.add_argument(
        "--policy-activation",
        type=str,
        choices=("tanh", "relu", "elu", "gelu", "leaky_relu"),
        default="tanh",
        help="Activation function used inside the policy network.",
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=5,
        help="Number of episodes to record for the post-training policy rollout video",
    )
    parser.add_argument(
        "--rollout-video",
        type=Path,
        default=None,
        help="Optional path for the policy rollout video (default: logs/visuals/final_policy_rollout.mp4)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume training from the latest saved model and VecNormalize statistics if available.",
    )
    parser.add_argument(
        "--video-interval",
        type=int,
        default=2_000_000,
        help="Environment steps between intermediate rollout recordings (set <=0 to disable).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="panda-obstacle-sac",
        help="Weights & Biases project name",
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional entity/team for Weights & Biases logging")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Optional name to give the Weights & Biases run")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases mode to run in (online, offline, or disabled).",
    )
    return parser.parse_args()


def _parse_float_or_keep_string(value: str) -> float | str:
    """Convert CLI strings to float when possible, otherwise return the original string (e.g., 'auto')."""

    try:
        return float(value)
    except ValueError:
        return value


def build_wandb_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI arguments into a serialisable config for Weights & Biases."""

    config: Dict[str, Any] = {"algorithm": ALGO_NAME}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


class VecNormalizeSyncCallback(BaseCallback):
    """Keep evaluation VecNormalize statistics aligned with the training environment."""

    def __init__(self, eval_env: VecNormalize | None) -> None:
        super().__init__(verbose=0)
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        if not isinstance(self.training_env, VecNormalize):
            return True
        if self.eval_env is None or not isinstance(self.eval_env, VecNormalize):
            return True

        if self.training_env.obs_rms is not None:
            self.eval_env.obs_rms = np.copy(self.training_env.obs_rms)
        if self.training_env.ret_rms is not None:
            self.eval_env.ret_rms = np.copy(self.training_env.ret_rms)
        return True


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


def build_policy_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct policy kwargs so users can easily customise the MLP architecture."""

    activation_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
    }
    activation_fn = activation_map[args.policy_activation]

    net_arch: List[int] | Dict[str, List[int]]
    shared_layers = list(args.policy_hidden_sizes)
    if args.policy_qf_hidden_sizes is not None:
        net_arch = {
            "pi": shared_layers,
            "qf": list(args.policy_qf_hidden_sizes),
        }
    else:
        net_arch = shared_layers

    return {"net_arch": net_arch, "activation_fn": activation_fn}


def record_policy_rollout(           #record the video of the robot performing the task
    model,
    env_factory: Callable[[], PandaObstacleEnv],
    video_path: Path,
    *,
    episodes: int = 10,
    deterministic: bool = True,
    normalization_path: Path | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
    wandb_key: str = "final_policy",
) -> None:
    if imageio is None:
        print("imageio not available, skipping policy rollout video generation.")
        return

    if normalization_path is not None:
        # Recreate the VecNormalize wrapper for deterministic evaluation
        rollout_vec_env = DummyVecEnv([env_factory])
        rollout_vec_env = VecMonitor(rollout_vec_env)
        vec_env = VecNormalize.load(str(normalization_path), rollout_vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        base_env = vec_env.envs[0]
    else:
        vec_env = None
        base_env = env_factory()

    fps = base_env.metadata.get("render_fps", 30) if hasattr(base_env, "metadata") else 30
    frames: List[np.ndarray] = []

    rollout_start = time.perf_counter()
    episode_durations: List[float] = []

    try:
        for _ in range(episodes):
            episode_start = time.perf_counter()
            terminated = False
            truncated = False
            step_count = 0
            max_steps = getattr(base_env, "max_episode_steps", 1000)

            if vec_env is not None:
                obs = vec_env.reset()
            else:
                obs, _ = base_env.reset()

            while not (terminated or truncated) and step_count < max_steps:
                frame = base_env.render()
                frames.append(frame)
                action, _ = model.predict(obs, deterministic=deterministic)
                if vec_env is not None:
                    obs, _, dones, infos = vec_env.step(action)
                    done_flag = bool(dones[0])
                    info = infos[0] if infos else {}
                    truncated = bool(info.get("TimeLimit.truncated", False))
                    terminated = done_flag and not truncated
                else:
                    obs, _, terminated, truncated, _ = base_env.step(action)
                step_count += 1

            episode_durations.append(time.perf_counter() - episode_start)

            if terminated or truncated:
                frame = base_env.render()
                frames.append(frame)

    finally:
        if vec_env is not None:
            vec_env.close()
        else:
            base_env.close()

    total_duration = time.perf_counter() - rollout_start

    if not frames:
        print("No frames captured for policy rollout video.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix.lower()
    if suffix == ".gif":
        imageio.mimsave(video_path, frames, fps=fps)
    else:
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
    if episode_durations:
        avg_duration = float(np.mean(episode_durations))
        print(
            f"Saved policy rollout video to {video_path} "
            f"(total {total_duration:.2f}s, {len(episode_durations)} episodes, "
            f"avg {avg_duration:.2f}s/episode)."
        )
    else:
        print(f"Saved policy rollout video to {video_path} (total {total_duration:.2f}s).")

    if wandb_run is not None:
        try:
            wandb_run.log({wandb_key: wandb.Video(str(video_path), fps=fps, format="mp4")})
            print(f"Uploaded policy rollout video to W&B under key '{wandb_key}'.")
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"Failed to upload video to Weights & Biases: {exc}")





def make_env(seed: int) -> Callable[[], PandaObstacleEnv]:
    def _init() -> PandaObstacleEnv:
        env = PandaObstacleEnv()
        env.reset(seed=seed)
        return env

    return _init


def main() -> None:
    args = parse_args()

    algo_cls = SAC

    args.log_dir.mkdir(parents=True, exist_ok=True)  # create the directories
    args.model_dir.mkdir(parents=True, exist_ok=True)
    sb3_log_dir = args.log_dir / "sb3"
    sb3_log_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = args.log_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        dir=str(wandb_dir),
        config=build_wandb_config(args),
    )

    normalization_path = args.model_dir / f"{ALGO_NAME}_panda_vecnormalize.pkl"

    env_fns = [make_env(args.seed + i) for i in range(args.num_envs)]
    train_env = SubprocVecEnv(env_fns) if args.num_envs > 1 else DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env, filename=str(args.log_dir / "monitor.csv"))

    if args.resume and normalization_path.exists():
        vec_env = VecNormalize.load(str(normalization_path), train_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env_base = DummyVecEnv([make_env(args.seed + 10_000)])
    eval_env_base = VecMonitor(eval_env_base)
    if args.resume and normalization_path.exists():
        eval_env = VecNormalize.load(str(normalization_path), eval_env_base)
    else:
        eval_env = VecNormalize(
            eval_env_base,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
    eval_env.training = False
    eval_env.norm_reward = False

    if isinstance(vec_env, VecNormalize) and isinstance(eval_env, VecNormalize):
        if vec_env.obs_rms is not None:
            eval_env.obs_rms = np.copy(vec_env.obs_rms)
        if vec_env.ret_rms is not None:
            eval_env.ret_rms = np.copy(vec_env.ret_rms)

    visualization_dir = args.log_dir / "visuals"
    visualization_callback = TrainingVisualizationCallback(visualization_dir)
    sync_callback = VecNormalizeSyncCallback(eval_env)

    checkpoint_callback = None
    if args.checkpoint_freq > 0:
        checkpoint_dir = args.model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_freq // args.num_envs),
            save_path=str(checkpoint_dir),
            name_prefix=f"{ALGO_NAME}_panda",
            save_vecnormalize=True,
        )
    else:
        checkpoint_dir = args.model_dir / "checkpoints"

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

    callbacks = [
        cb
        for cb in (checkpoint_callback, eval_callback, visualization_callback, sync_callback)
        if cb is not None
    ]

    def _latest_checkpoint(directory: Path) -> Path | None:
        if not directory.exists():
            return None
        candidates = [
            path
            for path in directory.glob("*.zip")
            if f"{ALGO_NAME}_panda" in path.stem  # avoid mixing PPO/SAC checkpoints
        ]
        candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None

    def _build_sb3_logger():
        logger = configure(str(sb3_log_dir), ["stdout", "csv"])
        if wandb_run is not None:
            logger.output_formats.append(WandbOutputFormat(wandb_run))
        return logger

    final_model_path = args.model_dir / f"{ALGO_NAME}_panda_final.zip"
    resume_path: Path | None = None
    if args.resume:
        if final_model_path.exists():
            resume_path = final_model_path
        else:
            resume_path = _latest_checkpoint(checkpoint_dir)

    if resume_path is not None:
        try:
            model = algo_cls.load(str(resume_path), env=vec_env, device=args.device)
            model.set_logger(_build_sb3_logger())
            reset_num_timesteps = False
        except Exception as exc:  # e.g., wrong algorithm checkpoint
            print(f"Failed to load checkpoint {resume_path}: {exc}. Starting fresh training.")
            resume_path = None

    if resume_path is None:
        policy_kwargs = build_policy_kwargs(args)
        train_freq = (args.train_freq, args.train_freq_unit)
        ent_coef = _parse_float_or_keep_string(args.ent_coef)
        target_entropy = _parse_float_or_keep_string(args.target_entropy)
        print(f"Using policy network architecture {policy_kwargs['net_arch']} with activation {args.policy_activation}.")
        print(
            "SAC hyperparameters: "
            f"buffer_size={args.buffer_size}, learning_starts={args.learning_starts}, batch_size={args.batch_size}, "
            f"gamma={args.gamma}, tau={args.tau}, lr={args.learning_rate}, train_freq={train_freq}, "
            f"gradient_steps={args.gradient_steps}, ent_coef={ent_coef}, target_entropy={target_entropy}, "
            f"target_update_interval={args.target_update_interval}"
        )
        model = algo_cls(
            args.policy,
            vec_env,
            verbose=1,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_rate=args.learning_rate,
            learning_starts=args.learning_starts,
            gamma=args.gamma,
            tau=args.tau,
            train_freq=train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            target_update_interval=args.target_update_interval,
            policy_kwargs=policy_kwargs,
        )
        model.set_logger(_build_sb3_logger())
        reset_num_timesteps = True

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks if callbacks else None,
            reset_num_timesteps=reset_num_timesteps,
        )

        model.save(str(final_model_path))
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.save(str(normalization_path))

        rollout_env_factory = make_env(args.seed + 20000)
        default_video_path = visualization_dir / "final_policy_rollout.mp4"
        video_path = args.rollout_video if args.rollout_video is not None else default_video_path
        record_policy_rollout(
            model,
            rollout_env_factory,
            video_path,
            episodes=args.rollout_episodes,
            normalization_path=normalization_path,
            wandb_run=wandb_run,
        )
    finally:
        vec_env.close()
        eval_env.close()
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
