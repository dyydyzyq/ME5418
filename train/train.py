from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Ensure the repo root is importable so we can load the custom environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from env.env import PandaObstacleEnv  # noqa: E402

ALGORITHMS: Dict[str, Callable] = {"ppo": PPO, "sac": SAC}


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

    callbacks = [cb for cb in (checkpoint_callback, eval_callback) if cb is not None]

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
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
