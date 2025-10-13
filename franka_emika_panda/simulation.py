import argparse
import math
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import mujoco
import mujoco.viewer

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "scene_withobstacles.xml"
DEFAULT_OUTPUT = BASE_DIR.parent / "recordings" / "obstacle_motion.mp4"
DEFAULT_DURATION = 1.0
DEFAULT_FPS = 60

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
box_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_box_x")
sphere_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_sphere_y")

box_amplitude = 0.12
sphere_amplitude = 0.12
motion_frequency = 0.25


def apply_obstacle_targets(sim_data: mujoco.MjData) -> None:
    """Drive obstacle actuators with sinusoidal position targets."""
    sim_data.ctrl[:] = 0.0
    phase = 2.0 * math.pi * motion_frequency * sim_data.time
    sim_data.ctrl[box_actuator_id] = box_amplitude * math.sin(phase)
    sim_data.ctrl[sphere_actuator_id] = sphere_amplitude * math.sin(phase + math.pi / 2.0)


def _write_video(frames, fps: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return output_path
    except Exception as exc:  # pragma: no cover
        fallback = output_path.with_suffix(".gif")
        with imageio.get_writer(fallback, mode="I", duration=1.0 / fps, loop=0) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Fallback to GIF recording due to writer error: {exc}")
        return fallback


# def record_video(
#     duration: float = DEFAULT_DURATION,
#     fps: int = DEFAULT_FPS,
#     output_path: Optional[Path] = None,
# ) -> Path:
#     """Simulate the moving obstacles passively and save a short clip."""

#     if output_path is None:
#         output_path = DEFAULT_OUTPUT

#     frame_period = 1.0 / fps
#     target_frames = max(1, int(duration * fps))
#     frames = []

#     with mujoco.Renderer(model, width=640, height=480) as renderer:
#         sim_data = mujoco.MjData(model)
#         apply_obstacle_targets(sim_data)
#         mujoco.mj_forward(model, sim_data)
#         renderer.update_scene(sim_data)
#         frames.append(renderer.render().copy())
#         next_capture_time = frame_period

#         while sim_data.time < duration or len(frames) < target_frames:
#             apply_obstacle_targets(sim_data)
#             mujoco.mj_step(model, sim_data)

#             while next_capture_time <= sim_data.time and len(frames) < target_frames:
#                 renderer.update_scene(sim_data)
#                 frames.append(renderer.render().copy())
#                 next_capture_time += frame_period

#     if not frames:
#         raise RuntimeError("No frames were captured while recording.")

#     saved_path = _write_video(frames, fps, Path(output_path))
#     print(f"Saved {len(frames)} frames to {saved_path}")
#     return saved_path


def run_viewer() -> None:
    """Launch the interactive MuJoCo viewer with moving obstacles."""

    sim_data = mujoco.MjData(model)
    apply_obstacle_targets(sim_data)
    mujoco.mj_forward(model, sim_data)

    with mujoco.viewer.launch_passive(model, sim_data) as viewer:
        while viewer.is_running():
            apply_obstacle_targets(sim_data)
            mujoco.mj_step(model, sim_data)
            viewer.sync()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record and/or view the Panda obstacle scene.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second for recording")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output video path")
    parser.add_argument("--no-record", action="store_true", help="Skip video recording")
    parser.add_argument("--no-viewer", action="store_true", help="Skip launching the interactive viewer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # if not args.no_record:
    #     record_video(duration=args.duration, fps=args.fps, output_path=args.output)

    if not args.no_viewer:
        run_viewer()


if __name__ == "__main__":
    main()
