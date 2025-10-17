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

box_amplitude = 0.05
sphere_amplitude = 0.05
motion_frequency = 0.1


def apply_obstacle_targets(sim_data: mujoco.MjData) -> None:
    """Drive obstacle actuators with sinusoidal position targets."""
    sim_data.ctrl[:] = 0.0
    phase = 2.0 * math.pi * motion_frequency * sim_data.time
    sim_data.ctrl[box_actuator_id] = box_amplitude * math.sin(phase)
    sim_data.ctrl[sphere_actuator_id] = sphere_amplitude * math.sin(phase + math.pi / 2.0)
    
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


    if not args.no_viewer:
        run_viewer()


if __name__ == "__main__":
    main()
