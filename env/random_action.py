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

# EpisodeStats is a tiny container used to summarize an episode after it
# finishes. Fields:
# - episode: index/ID for the episode (1-based in this script)
# - steps: number of steps taken in the episode
# - return_: cumulative reward (sum of rewards) collected during the episode
# - terminated: whether the episode ended via a terminal condition (success/collision)
# - truncated: whether the episode ended due to truncation (max steps)


class LiveRenderer:
    def __init__(self, width: int, height: int) -> None:
        # Lazy import --- plotting is optional and can be disabled for
        # headless execution (e.g. CI or when using --no-render). Importing
        # matplotlib only when needed reduces startup cost and avoids hard
        # dependency requirements for non-visual runs.
        import matplotlib.pyplot as plt  # type: ignore

        # Turn on interactive mode so the plot updates without blocking the
        # script. Store the module reference so we can call plt.pause/close.
        plt.ion()
        self._plt = plt

        # Create a figure sized in inches. We assume 100 dpi and convert the
        # supplied pixel-based width/height to inches for the figure size.
        # This determines the resolution of images displayed in the preview.
        self._fig, self._ax = plt.subplots(figsize=(width / 100, height / 100))

        # _image will hold the AxesImage returned by imshow; _overlay is a
        # Text artist used to display step/reward/action metadata on top of
        # the image. They are lazily created on the first update() call.
        self._image = None
        self._overlay = None

        # Hide axes ticks/labels for a cleaner video-style output.
        self._ax.axis("off")

    def update(
        self,
        frame,
        step: int,
        reward: float,
        distance: float,
        action: np.ndarray,
    ) -> None:
        # On the first call create the image and overlay text artist. We
        # reuse the same artists on subsequent frames which is much cheaper
        # than recreating them every update.
        if self._image is None:
            # imshow returns an AxesImage object we keep a reference to so
            # we can call set_data(frame) to update pixel data in-place.
            self._image = self._ax.imshow(frame)

            # Place a small text box in the upper-left of the image with a
            # semi-opaque background so the numbers remain readable over
            # varying frame content.
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
            # Update the image pixel data in-place for best performance.
            self._image.set_data(frame)

        # Build a concise preview of the first three action components. We
        # format each with sign and two decimal places so the overlay is
        # compact and stable across frames.
        action_preview = " ".join(f"{a:+.2f}" for a in action[:3])

        # Compose the overlay text. We use small width/precision format
        # specifiers to keep the overlay tidy; these are purely visual and
        # do not affect simulation data.
        self._overlay.set_text(
            f"Step: {step:03d}\nReward: {reward:7.3f}\nDistance: {distance:.3f}"
            f"\nAction[0:3]: {action_preview}"
        )

        # Request a non-blocking redraw and sleep a tiny amount so the GUI
        # event loop has time to process the update. Using pause() with a
        # short sleep keeps the viewer responsive without blocking logic.
        self._fig.canvas.draw_idle()
        self._plt.pause(1 / 60)

    def close(self) -> None:
        self._plt.ioff()
        self._plt.close(self._fig)


def parse_args() -> argparse.Namespace:
    # Build a small CLI to control the number of episodes, steps per
    # episode, reproducibility seed and whether to render frames.
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
    # Simple debug helper that prints a small summary of an observation
    # vector. This helps detect NaNs/infs or unexpected shapes early.
    if not np.all(np.isfinite(obs)):
        finite_ratio = np.isfinite(obs).sum() / obs.size
        print(f"Warning: Observation contains non-finite values ({finite_ratio:.2%} finite).")
    print("Observation shape:", obs.shape)
    # Print the first few values to aid manual inspection during runs.
    print("Observation sample (first 5 values):", obs[:5])


def run_episode(
    env: PandaObstacleEnv,
    episode_idx: int,
    max_steps: int,
    renderer: Optional[LiveRenderer],
) -> EpisodeStats:
    # Run a single episode using random actions sampled from the env's
    # action_space. The loop mirrors how an RL agent would interact with
    # the environment but uses env.action_space.sample() for action values.
    stats = EpisodeStats(episode=episode_idx)

    # Reset returns the initial observation and an info dict. We log the
    # observation shape/value to help catch shape mismatches early.
    obs, info = env.reset()
    log_observation(obs)
    print(f"Episode {episode_idx}: goal position {info['goal']}")

    # Step through the environment until termination, truncation, or
    # until max_steps is reached. We keep and report simple episode stats.
    for step in range(max_steps):
        # Sample a random action consistent with the environment's
        # declared action_space. Using action_space.sample() ensures the
        # script exercises the same action bounds the training code will.
        action = env.action_space.sample()
        # Sanity-check: action_space.contains is a defensive check
        # (it should always be True for a sample from the space).
        assert env.action_space.contains(action), "Sampled action is outside the action space."
        print(f"Episode {episode_idx}, step {step}: action {action}")

        # Step the environment and accumulate rewards / flags.
        obs, reward, terminated, truncated, info = env.step(action)
        stats.steps = step + 1
        stats.return_ += reward
        stats.terminated = terminated
        stats.truncated = truncated

        # If a renderer is enabled, request an RGB frame and overlay
        # diagnostic text showing step/reward/distance/action preview.
        if renderer:
            frame = env.render()
            # distance_to_goal is stored as a 1-element array in info
            distance = float(info["distance_to_goal"][0])
            renderer.update(frame, step, reward, distance, action)

        # Stop the episode if the env signalled termination or truncation.
        if terminated or truncated:
            break

    # Nicely format the episode result for human inspection.
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
