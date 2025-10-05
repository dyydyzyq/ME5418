from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "franka_emika_panda" / "scene_withobstacles.xml"
)


class PandaObstacleEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        model_path: Optional[Path] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 200,
        seed: Optional[int] = None,
        render_width: int = 640,
        render_height: int = 480,
        goal_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        goal_reach_threshold: float = 0.03,
    ) -> None:
        super().__init__()

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_width = render_width
        self.render_height = render_height
        self.goal_reach_threshold = goal_reach_threshold

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.dt = self.model.opt.timestep * self.frame_skip

        low = np.array([0.2, -0.4, 0.1], dtype=np.float64)
        high = np.array([0.8, 0.4, 0.6], dtype=np.float64)
        if goal_bounds is None:
            self.goal_bounds = (low, high)
        else:
            self.goal_bounds = goal_bounds

        self.goal_pos = np.zeros(3, dtype=np.float64)

        self.manip_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.manip_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.manip_joint_names
            ],
            dtype=np.int32,
        )
        self.manip_qvel_idx = np.arange(len(self.manip_joint_ids))
        self.manip_body_ids = set(self.model.jnt_bodyid[self.manip_joint_ids].tolist())
        self.hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.manip_body_ids.update(
            [
                self.hand_body_id,
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"),
            ]
        )
        self.manip_body_ids = np.array(sorted(self.manip_body_ids), dtype=np.int32)

        self.obstacle_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_box"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_sphere"),
            ],
            dtype=np.int32,
        )

        self.manip_actuator_ids = np.array(range(len(self.manip_joint_ids)), dtype=np.int32)
        self.box_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_box_x"
        )
        self.sphere_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "move_sphere_y"
        )

        self.box_amplitude = 0.12
        self.sphere_amplitude = 0.12
        self.obstacle_frequency = 0.25

        self.action_dim = len(self.manip_actuator_ids)
        self.action_limit = np.ones(self.action_dim, dtype=np.float32) * 2.0
        self.action_space = gym.spaces.Box(
            low=-self.action_limit,
            high=self.action_limit,
            dtype=np.float32,
        )

        obs_dim = (
            self.action_dim
            + 3 * len(self.manip_joint_ids)
            + 3
            + 3 * len(self.obstacle_body_ids)
        )
        obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.collision_penalty = -100.0
        self.goal_reward = 100.0
        self.action_cost = 1e-2
        self.accel_penalty = 5e-2
        self.jerk_penalty = 1e-2
        self.step_penalty = 1e-2

        self._renderer: Optional[mujoco.Renderer] = None
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0
        self._prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_accel = np.zeros(self.action_dim, dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)
        self._update_obstacles(self.data.time)
        self.goal_pos = self._sample_goal()

    def seed(self, seed: Optional[int] = None) -> None:
        self._np_random, _ = gym.utils.seeding.np_random(seed)

    def _sample_goal(self) -> np.ndarray:
        low, high = self.goal_bounds
        return self._np_random.uniform(low, high)

    def _update_obstacles(self, time_value: float) -> None:
        phase = 2.0 * np.pi * self.obstacle_frequency * time_value
        self.data.ctrl[self.box_actuator_id] = self.box_amplitude * np.sin(phase)
        self.data.ctrl[self.sphere_actuator_id] = self.sphere_amplitude * np.sin(
            phase + np.pi / 2.0
        )

    def _get_joint_positions(self) -> np.ndarray:
        positions = []
        for body_id in self.model.jnt_bodyid[self.manip_joint_ids]:
            positions.append(self.data.xpos[body_id])
        return np.concatenate(positions)

    def _get_obstacle_positions(self) -> np.ndarray:
        return self.data.xpos[self.obstacle_body_ids].ravel()

    def _get_obs(self) -> np.ndarray:
        joint_vel = self.data.qvel[self.manip_qvel_idx]
        joint_pos = self._get_joint_positions()
        obs = np.concatenate(
            [joint_vel, joint_pos, self.goal_pos, self._get_obstacle_positions()]
        )
        return obs.astype(np.float32)

    def _compute_accel_and_jerk(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        accel = (action - self._prev_action) / self.dt
        jerk = (accel - self._prev_accel) / self.dt
        return accel, jerk

    def _detect_collision(self) -> bool:
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

    def _end_effector_distance(self) -> float:
        ee_pos = self.data.xpos[self.hand_body_id]
        return float(np.linalg.norm(ee_pos - self.goal_pos))

    def _compute_reward(
        self,
        action: np.ndarray,
        accel: np.ndarray,
        jerk: np.ndarray,
        collided: bool,
        reached_goal: bool,
    ) -> float:
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

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._prev_action[:] = 0.0
        self._prev_accel[:] = 0.0

        if options and "initial_state" in options:
            qpos, qvel = options["initial_state"]
            self.data.qpos[: len(qpos)] = qpos
            self.data.qvel[: len(qvel)] = qvel

        self.goal_pos = self._sample_goal()
        self.data.ctrl[:] = 0.0
        self._update_obstacles(self.data.time)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {"goal": self.goal_pos.copy()}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, np.ndarray]]:
        action = np.asarray(action, dtype=np.float64)
        clipped = np.clip(action, -self.action_limit, self.action_limit)

        accel, jerk = self._compute_accel_and_jerk(clipped)

        for _ in range(self.frame_skip):
            self._update_obstacles(self.data.time)
            self.data.ctrl[self.manip_actuator_ids] = clipped
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        collided = self._detect_collision()
        reached_goal = self._end_effector_distance() <= self.goal_reach_threshold
        reward = self._compute_reward(clipped, accel, jerk, collided, reached_goal)

        obs = self._get_obs()
        terminated = collided or reached_goal
        truncated = self._step_count >= self.max_episode_steps

        self._prev_action = clipped
        self._prev_accel = accel

        info: Dict[str, np.ndarray] = {
            "goal": self.goal_pos.copy(),
            "distance_to_goal": np.array([self._end_effector_distance()], dtype=np.float64),
            "collided": np.array([collided], dtype=bool),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, width=self.render_width, height=self.render_height)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
