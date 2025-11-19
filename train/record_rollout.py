from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Type

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as exc:  # pragma: no cover - tooling should provide imageio
    raise RuntimeError("imageio is required to export rollout videos") from exc

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


# Ensure the repository root is importable so we can load the custom environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.env import PandaObstacleEnv


ALGO_MAP: Dict[str, Type[BaseAlgorithm]] = {
    "sac": SAC,
    "ppo": PPO,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record rollout videos for a trained Panda obstacle policy.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models") / "sac_panda_final.zip",
        help="Path to the saved SB3 model checkpoint.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=Path("models") / "sac_panda_vecnormalize.pkl",
        help="Path to VecNormalize statistics saved during training.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("logs") / "visuals" / "manual_rollout.mp4",
        help="Where to save the rendered video.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=tuple(ALGO_MAP.keys()),
        help="Algorithm used to train the policy (determines which SB3 class loads the checkpoint).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda/auto).")
    parser.add_argument("--episodes", type=int, default=3, help="Number of rollout episodes to record.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for the evaluation environment.")
    parser.add_argument(
        "--render-width",
        type=int,
        default=1280,
        help="Width of the rendered frames (reducing this saves memory).",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=640,
        help="Height of the rendered frames (reducing this saves memory).",
    )
    parser.add_argument(
        "--stochastic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use stochastic actions instead of deterministic policy outputs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on steps per episode (defaults to environment limit).",
    )
    return parser.parse_args()


def make_env(seed: int, render_width: int, render_height: int) -> Callable[[], PandaObstacleEnv]:
    def _init() -> PandaObstacleEnv:
        env = PandaObstacleEnv(render_width=render_width, render_height=render_height)
        env.reset(seed=seed)
        return env

    return _init


def _build_rollout_env(
    env_factory: Callable[[], PandaObstacleEnv],
    normalization_path: Path | None,
) -> tuple[DummyVecEnv | VecNormalize | None, PandaObstacleEnv]:
    if normalization_path is not None and normalization_path.exists():
        vec_env = DummyVecEnv([env_factory])
        vec_env = VecMonitor(vec_env)
        vec_env = VecNormalize.load(str(normalization_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        base_env = vec_env.envs[0]
        return vec_env, base_env

    base_env = env_factory()
    return None, base_env


def record_policy_rollout(
    model: BaseAlgorithm,
    env_factory: Callable[[], PandaObstacleEnv],
    video_path: Path,
    *,
    episodes: int,
    deterministic: bool,
    normalization_path: Path | None,
    max_steps: int | None,
) -> None:
    vec_env, base_env = _build_rollout_env(env_factory, normalization_path)
    fps = base_env.metadata.get("render_fps", 30) if hasattr(base_env, "metadata") else 30

    video_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {episodes} episodes to {video_path} (fps={fps}, deterministic={deterministic}).")

    total_start = time.perf_counter()
    episode_durations: list[float] = []

    try:
        with imageio.get_writer(video_path, fps=fps) as writer:
            for ep in range(episodes):
                ep_start = time.perf_counter()
                terminated = False
                truncated = False
                step_count = 0
                step_limit = max_steps or getattr(base_env, "max_episode_steps", 1000)

                if vec_env is not None:
                    obs = vec_env.reset()
                else:
                    obs, _ = base_env.reset()

                # Write the initial frame so the video starts with the reset posture.
                writer.append_data(base_env.render())

                while not (terminated or truncated) and step_count < step_limit:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    if vec_env is not None:
                        obs, _, dones, infos = vec_env.step(action)
                        done_flag = bool(dones[0])
                        info = infos[0] if infos else {}
                        truncated = bool(info.get("TimeLimit.truncated", False))
                        terminated = done_flag and not truncated
                    else:
                        obs, _, terminated, truncated, _ = base_env.step(action)

                    writer.append_data(base_env.render())
                    step_count += 1

                episode_duration = time.perf_counter() - ep_start
                episode_durations.append(episode_duration)
                status = "terminated" if terminated else "truncated" if truncated else "max_steps"
                print(f"Episode {ep + 1}/{episodes} finished after {step_count} steps ({status}, {episode_duration:.2f}s).")
    finally:
        total_duration = time.perf_counter() - total_start
        if vec_env is not None:
            vec_env.close()
        else:
            base_env.close()

    avg_duration = float(np.mean(episode_durations)) if episode_durations else 0.0
    print(
        f"Saved policy rollout video to {video_path} "
        f"(episodes={episodes}, total={total_duration:.2f}s, avg={avg_duration:.2f}s)."
    )


def main() -> None:
    args = parse_args()
    algo_cls = ALGO_MAP[args.algo.lower()]

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    normalization_path = args.vecnormalize_path if args.vecnormalize_path.exists() else None
    if normalization_path is None:
        print(f"Warning: VecNormalize stats not found at {args.vecnormalize_path}; running without normalization.")

    model = algo_cls.load(str(args.model_path), device=args.device)
    env_factory = make_env(args.seed, args.render_width, args.render_height)

    record_policy_rollout(
        model,
        env_factory,
        video_path=args.video_path,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        normalization_path=normalization_path,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
