"""Utility script to roll out random actions in the PandaObstacleEnv.

This module provides a tiny command-line tool used during development to
exercise the environment with random actions and (optionally) render the
simulation frames live using matplotlib.

Primary components:
- EpisodeStats: simple dataclass holding per-episode summary statistics.
- LiveRenderer: minimalist matplotlib-based live preview overlaying step/
    reward/action information on top of frames returned by the env's render().
- run_episode / main: run one or more random episodes and print simple
    diagnostics. Useful for smoke-testing environment integration and
    catching NaNs, invalid shapes, or immediate crashes.

Notes / behavior:
- The script samples actions via `env.action_space.sample()` so it exercises
    exactly the same action bounds the training pipeline would use.
- Because MuJoCo returns many arrays as float64, we sometimes cast to
    float32 for rendering/printing — this is only cosmetic.
- The script intentionally keeps dependencies minimal and does lazy imports
    for optional visualization so it can run in headless CI if needed.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np

from env import PandaObstacleEnv  # ensure package import path is correct


@dataclass
class EpisodeStats:
    episode: int
    steps: int = 0
    return_: float = 0.0
    terminated: bool = False
    truncated: bool = False


class LiveRenderer:
    def __init__(self, width: int, height: int) -> None:
        # Lazy import --- plotting is optional and can be disabled for
        # headless execution (e.g. CI or when using --no-render).
        import matplotlib.pyplot as plt  # type: ignore

        plt.ion()
        self._plt = plt
        # figsize arguments take inches; convert from pixels (assume 100 dpi)
        self._fig, self._ax = plt.subplots(figsize=(width / 100, height / 100))
        self._image = None
        self._overlay = None
        # Hide axes for a cleaner frame-only display
        self._ax.axis("off")

    def update(
        self,
        frame,
        step: int,
        reward: float,
        distance: float,
        action: np.ndarray,
    ) -> None:
        if self._image is None:
            self._image = self._ax.imshow(frame)
            self._overlay = self._ax.text(
                0.02,
                0.98,
                "",
                transform=self._ax.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
            )
        else:
            self._image.set_data(frame)

        action_preview = " ".join(f"{a:+.2f}" for a in action[:3])
        self._overlay.set_text(
            f"Step: {step:03d}\nReward: {reward:7.3f}\nDistance: {distance:.3f}"
            f"\nAction[0:3]: {action_preview}"
        )
        self._fig.canvas.draw_idle()
        self._plt.pause(1 / 60)

    def close(self) -> None:
        self._plt.ioff()
        self._plt.close(self._fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll out random actions in PandaObstacleEnv.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of random episodes to run.")
    parser.add_argument(
        "--steps", type=int, default=200, help="Maximum steps per episode before truncation."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable matplotlib rendering (useful for headless runs).",
    )
    return parser.parse_args()


def log_observation(obs: np.ndarray) -> None:
    if not np.all(np.isfinite(obs)):
        finite_ratio = np.isfinite(obs).sum() / obs.size
        print(f"Warning: Observation contains non-finite values ({finite_ratio:.2%} finite).")
    print("Observation shape:", obs.shape)
    print("Observation sample (first 5 values):", obs[:5])


def run_episode(
    env: PandaObstacleEnv,
    episode_idx: int,
    max_steps: int,
    renderer: Optional[LiveRenderer],
) -> EpisodeStats:
    stats = EpisodeStats(episode=episode_idx)
    obs, info = env.reset()

    log_observation(obs)
    print(f"Episode {episode_idx}: goal position {info['goal']}")

    for step in range(max_steps):
        action = env.action_space.sample()
        assert env.action_space.contains(action), "Sampled action is outside the action space."
        print(f"Episode {episode_idx}, step {step}: action {action}")
        # grasp_center = info.get("grasp_center", None)
        # print(f"Episode {episode_idx}, Step {step}: Grasp Center = {grasp_center}")

        obs, reward, terminated, truncated, info = env.step(action)
        stats.steps = step + 1
        stats.return_ += reward
        stats.terminated = terminated
        stats.truncated = truncated

        if renderer:
            frame = env.render()
            distance = float(info["distance_to_goal"][0])
            renderer.update(frame, step, reward, distance, action)
        if terminated or truncated:
            break

    status = "terminated" if stats.terminated else "truncated" if stats.truncated else "timeout"

    print(
        f"grasp center: {info['grasp_center']}, goal: {info['goal']}"
        f"Episode {episode_idx} finished after {stats.steps} steps "
        f"({status}), return={stats.return_: .2f}"
    )
    print("-" * 60)
    return stats


def main() -> None:

    args = parse_args()
    enable_render = not args.no_render
    env = PandaObstacleEnv(render_width=640, render_height=480, seed=args.seed)
    renderer = LiveRenderer(env.render_width, env.render_height) if enable_render else None
    

    returns = []
    for episode in range(1, args.episodes + 1):
        stats = run_episode(env, episode, args.steps, renderer)
        returns.append(stats.return_)

    if returns:
        print(f"Average return over {len(returns)} episodes: {np.mean(returns): .2f}")
        print(f"Return std: {np.std(returns): .2f}")

    if renderer:
        renderer.close()
    env.close()


if __name__ == "__main__":
    main()
