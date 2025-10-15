"""
Panda obstacle environment module.

This module contains a single Gym-compatible environment, PandaObstacleEnv,
which wraps a MuJoCo model of a Franka Emika Panda manipulator in an
obstacle-avoidance task. The environment exposes a continuous action space
that drives actuators on the manipulator and provides observations that
include joint positions, joint velocities, the current goal position, and
the positions of the moving obstacles.

Design goals / notes:
- Lightweight wrapper around MuJoCo for training RL agents (Stable-Baselines3
    compatible). The environment handles model loading, stepping the simulator
    (with optional frame-skip), simple obstacle motion, and reward calculation.
- The implementation favors clarity and debuggability over performance. For
    long-running training you may want to optimise array copies and sampling.
- This file intentionally keeps simulation logic in Python; actuators are
    commanded via `self.data.ctrl` and MuJoCo integrates the dynamics.

Assumptions:
- The MuJoCo XML model contains named joints `joint1..joint7`, sites
    `left_tip`/`right_tip` (used for grasp center), and body names
    `moving_box`/`moving_sphere` that represent moving obstacles.
- The model maps actuators controlling the manipulator into predictable
    actuator indices; the code queries names at init time and will raise
    (via MuJoCo) if names are missing.

If you change the model naming or structure, update the name lists used in
_init_manipulator_ids/_init_actuator_ids accordingly.
"""


from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR.parent / "franka_emika_panda" / "scene_withobstacles.xml"


class PandaObstacleEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gym environment for Panda manipulator with moving obstacles.

    Observations (numpy float32 array):
    - joint velocities for the manipulator (len = number of manip joints)
    - world-space positions of each manipulator joint/body (3 * n)
    - the 3D goal position for the end effector
    - world-space positions of each obstacle (3 * n_obstacles)

    Actions:
    - A 1-D continuous vector in range [-1, 1] for each actuator driving the
      manipulator. The environment maps this normalized action to actuator
      commands using the configured per-joint v_min / v_max scaling.

    Rewards and termination:
    - Positive reward for reaching the goal (goal_reward).
    - Large negative penalty for collisions with moving obstacles.
    - Small penalties for action magnitude, acceleration, jerk, and per-step.

    Rendering:
    - `render()` returns an RGB image array using MuJoCo's Renderer.

    Notes:
    - The class uses MuJoCo's low-level API (MjModel, MjData) directly; if
      you want to run in headless CI you may need to configure the MuJoCo
      OpenGL backend (e.g. MUJOCO_GL=egl/osmesa) before creating the env.
    """

    # Metadata exposed to Gym renderers / utilities.
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        model_path: Optional[Path] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 2000,
        seed: Optional[int] = None,
        render_width: int = 640,    
        render_height: int = 480,
        goal_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        goal_reach_threshold: float = 0.03,
    ) -> None:
        super().__init__()

        # Resolve and load MuJoCo model. If model_path is None the bundled
        # default model path inside the repo is used. Passing an explicit
        # Path allows users to point to a modified XML file.
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        # MjData holds the mutable simulation state (qpos, qvel, contacts, etc.)
        self.data = mujoco.MjData(self.model)

        # Frame-skip: we will call mujoco.mj_step() frame_skip times per env
        # step. This effectively multiplies the simulator timestep by frame_skip
        # for the purposes of agent actions and observations.
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_width = render_width
        self.render_height = render_height
        self.goal_reach_threshold = goal_reach_threshold
        # Effective control timestep used by accel/jerk calculations
        self.dt = self.model.opt.timestep * self.frame_skip

        # Goal sampling bounds (3D). If not provided, sensible defaults are
        # used; these should be inside the robot workspace defined by the XML.
        if goal_bounds is None:
            low = np.array([0.2, -0.4, 0.1], dtype=np.float64)
            high = np.array([0.8, 0.4, 0.6], dtype=np.float64)
            self.goal_bounds = (low, high)
        else:
            self.goal_bounds = goal_bounds

        # Current goal position (3-vector). It will be sampled in reset().
        self.goal_pos = np.zeros(3, dtype=np.float64)
        

        # Initialize indices and ID arrays for joints, bodies and actuators.
        # These helper methods query the MuJoCo model by name so that the
        # rest of the class can operate on integer IDs instead of strings.
        self._init_manipulator_ids()
        self._init_obstacle_ids()
        self._init_actuator_ids()
        self._init_spaces()

        # Obstacle motion parameters: amplitudes and frequency for the
        # scripted obstacle trajectories (sine waves). These are tunable
        # parameters and are independent of agent control.
        self.box_amplitude = 0.12
        self.sphere_amplitude = 0.12
        self.obstacle_frequency = 0.25

        # Reward shaping coefficients: penalties and bonuses used by
        # _compute_reward(). Keep these values consistent between training
        # runs to aid reproducibility.
        self.collision_penalty = -100.0
        self.goal_reward = 100.0
        self.action_cost = 1e-2
        self.accel_penalty = 5e-2
        self.jerk_penalty = 1e-2
        self.step_penalty = 1e-2

        # Renderer is lazily created on the first call to render(). The
        # MuJoCo Renderer holds GPU/GL state so we keep it optional.
        self._renderer: Optional[mujoco.Renderer] = None

        # RNG for environment-level randomness (goal sampling, etc.)
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0

        # Track previous action and acceleration to compute accel/jerk costs.
        # These arrays are sized to the action dimension which is computed in
        # _init_spaces() above.
        self._prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_accel = np.zeros(self.action_dim, dtype=np.float64)

        # Ensure MuJoCo internal state and derived quantities are up to date.
        mujoco.mj_forward(self.model, self.data)
        # Initialize scripted obstacles and sample an initial goal
        self._update_obstacles(self.data.time)
        self.goal_pos = self._sample_goal()

    def _init_manipulator_ids(self) -> None:
        self.manip_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.manip_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.manip_joint_names
            ],
            dtype=np.int32,
        )
        self.left_tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tip"
        )
        self.right_tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tip"
        )

        # qvel index for the manipulator joints
        self.manip_qvel_idx = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.manip_joint_ids],
            dtype=np.int32,
        )

        # get the body ids of the manipulator
        body_ids = set(self.model.jnt_bodyid[self.manip_joint_ids].tolist())
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        self.hand_body_id = hand_id
        body_ids.update([hand_id, left_finger_id, right_finger_id])
        self.manip_body_ids = np.array(sorted(body_ids), dtype=np.int32)

    def get_grasp_center(self) -> np.ndarray:
        L = self.data.site_xpos[self.left_tip_id]
        R = self.data.site_xpos[self.right_tip_id]
        return 0.5 * (L + R)

    def _init_obstacle_ids(self) -> None:  # initialize the mujoco model and data
        self.obstacle_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_box"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_sphere"),
            ],
            dtype=np.int32,
        )

    def _init_actuator_ids(self) -> None:
        names = [f"actuator{i}" for i in range(1, 8)]
        self.manip_actuator_ids = np.array(
            [self.model.actuator(n).id for n in names], dtype=np.int32
        )

        self.box_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_box_x"
        )
        self.sphere_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_sphere_y"
        )


    def _init_spaces(self) -> None:
        # ---- action_space ----
        self.action_dim = len(self.manip_actuator_ids) 
        self.v_max = np.array([1.5, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5], dtype=np.float64)
        self.v_min = -self.v_max
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        # ---- obs_state ----
        obs_dim = (
            3 * len(self.manip_joint_ids)  #joint positions
            + len(self.manip_joint_ids)  #joint velocities
            + 3     #goal position
            + 3 * len(self.obstacle_body_ids) #obstacle positions
        )
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32,
        )

    def seed(self, seed: Optional[int] = None) -> None:  # set random seed
        self._np_random, _ = gym.utils.seeding.np_random(seed)

    def _sample_goal(self) -> np.ndarray:  # sample a new goal position within the goal bounds
        low, high = self.goal_bounds
        return self._np_random.uniform(low, high)
    
    def _update_obstacles(self, time_value: float) -> None:  # update the obstacle positions based on the time value
        phase = 2.0 * np.pi * self.obstacle_frequency * time_value
        self.data.ctrl[self.box_actuator_id] = self.box_amplitude * np.sin(phase)
        self.data.ctrl[self.sphere_actuator_id] = self.sphere_amplitude * np.sin(
            phase + np.pi / 2.0
        )

    def _get_joint_positions(self) -> np.ndarray:  # get the joint positions in the world frame
        positions = []
        for body_id in self.model.jnt_bodyid[self.manip_joint_ids]:
            positions.append(self.data.xpos[body_id])
        return np.concatenate(positions)

    def _get_obstacle_positions(self) -> np.ndarray:  # get the obstacle positions in the world frame
        return self.data.xpos[self.obstacle_body_ids].ravel()

    def _get_obs(self) -> np.ndarray:  # get the observation of the environment
        joint_vel = self.data.qvel[self.manip_qvel_idx]
        joint_pos = self._get_joint_positions()
        obs = np.concatenate(
            [joint_vel, joint_pos, self.goal_pos, self._get_obstacle_positions()]
        )
        return obs.astype(np.float32)

    def _compute_accel_and_jerk(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:  # calculate the acceleration and jerk based on the current action
        accel = (action - self._prev_action) / self.dt
        jerk = (accel - self._prev_accel) / self.dt

        # update previous action and acceleration
        self._prev_action = action
        self._prev_accel = accel

        return accel, jerk

    def _detect_collision(self) -> bool:  # detect if there is a collision between the manipulator and the obstacles
        obstacle_bodies = set(self.obstacle_body_ids.tolist())
        manip_bodies = set(self.manip_body_ids.tolist())
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if (body1 in manip_bodies and body2 in obstacle_bodies) or (
                body2 in manip_bodies and body1 in obstacle_bodies
            ):
                return True
        return False

    def distance_to_goal(self) -> float:  # calculate the distance between the end effector and the goal position
        ee_pos = self.get_grasp_center()
        return float(np.linalg.norm(ee_pos - self.goal_pos))

    def _compute_reward(
        self,
        action: np.ndarray,
        accel: np.ndarray,
        jerk: np.ndarray,
        collided: bool,
        reached_goal: bool,
    ) -> float:  # compute the reward
        reward = -self.step_penalty
        reward -= self.action_cost * float(np.linalg.norm(action))
        reward -= self.accel_penalty * float(np.linalg.norm(accel))
        reward -= self.jerk_penalty * float(np.linalg.norm(jerk))
        if collided:
            reward += self.collision_penalty
        if reached_goal:
            reward += self.goal_reward
        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if seed is not None:
            self.seed(seed)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self._step_count = 0
        self._prev_action[:] = 0.0
        self._prev_accel[:] = 0.0

        self.goal_pos = self._sample_goal()
        self.data.ctrl[:] = 0.0
        self._update_obstacles(self.data.time)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {"goal": self.goal_pos.copy()}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, np.ndarray]]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = 0.5 * (a + 1.0) * (self.v_max - self.v_min) + self.v_min

        accel, jerk = self._compute_accel_and_jerk(v_cmd)

        for _ in range(self.frame_skip):
            self._update_obstacles(self.data.time)
            self.data.ctrl[self.manip_actuator_ids] = v_cmd
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        collided = self._detect_collision()
        reached_goal = self.distance_to_goal() <= self.goal_reach_threshold
        reward = self._compute_reward(v_cmd, accel, jerk, collided, reached_goal)

        obs = self._get_obs()
        terminated = collided or reached_goal
        truncated = self._step_count >= self.max_episode_steps

        info: Dict[str, np.ndarray] = {
            "goal": self.goal_pos.copy(),
            "distance_to_goal": np.array([self.distance_to_goal()], dtype=np.float64),
            "grasp_center": self.get_grasp_center(),
            "collided": np.array([collided], dtype=bool),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:  # render the environment and return an RGB image
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, width=self.render_width, height=self.render_height
            )
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:  # close the renderer if it exists
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
