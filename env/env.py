from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import gymnasium as gym
import mujoco
import numpy as np


BASE_DIR = Path(__file__).resolve().parent    #load the default mujoco model path
DEFAULT_MODEL_PATH = BASE_DIR.parent / "franka_emika_panda" / "scene_withobstacles.xml"


class PandaObstacleEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 120}  

    def __init__(
        self,
        model_path: Optional[Path] = None,
        frame_skip: int = 5,
        max_episode_steps: int = 1600,
        seed: Optional[int] = None,
        render_width: int = 1280,    
        render_height: int = 640,
        goal_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        goal_reach_threshold: float = 0.02,
        safety_distance: float = 0.2,
        avoidance_weights: Optional[np.ndarray] = None,
        avoidance_gain: float = 2.0,
    ) -> None:
        
        super().__init__()

        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)       
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.render_width = render_width
        self.render_height = render_height
        self.goal_reach_threshold = goal_reach_threshold
        self.safety_distance = safety_distance
        self.dt = self.model.opt.timestep * self.frame_skip
        self._ensure_offscreen_buffer_capacity()

        if goal_bounds is None:
            low  = np.array([0.3, -0.4, 0.3], dtype=np.float64)
            high = np.array([0.5,  0.4, 0.7], dtype=np.float64)
            self.goal_bounds = (low, high)
        else:
            self.goal_bounds = goal_bounds

        self.goal_pos = np.zeros(3, dtype=np.float64)

        self._init_manipulator_ids()
        self._init_obstacle_ids()
        self._init_actuator_ids()
        self._init_spaces() 
        self._set_avoidance_weights(avoidance_weights)
        self._configure_critical_links()
        self.avoidance_gain = float(avoidance_gain)

        self.w_plan = -10.0
        self.box_amplitude = 0.1
        self.sphere_amplitude = 0.1
        self.obstacle_frequency = 0.125
        
        self.collision_penalty = -250.0
        self.goal_reward = 200.0
        self.accel_penalty = 0.00001
        self.jerk_penalty = 0.00001
        self.step_penalty = 0.02

        self._renderer: Optional[mujoco.Renderer] = None
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._step_count = 0
        self._prev_qvel = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_accel = np.zeros(self.action_dim, dtype=np.float64)
        self._prev_obstacle_distances: Optional[np.ndarray] = None

        mujoco.mj_forward(self.model, self.data)
        self._update_obstacles(self.data.time)
        self.goal_pos = self._sample_goal()
        self._refresh_obstacle_distance_buffer()

    def _compute_reward(     #compute the reward
        self,
        accel: np.ndarray,
        jerk: np.ndarray,
        collided: bool,
        reached_goal: bool,
    ) -> Tuple[float, np.ndarray]:
        obstacle_distances = self._compute_link_obstacle_distances()
        avoidance_reward = self._compute_avoidance_reward(obstacle_distances)
        reward = 0.0
        # steps_elapsed = max(0, self._step_count - 1)
        # reward -= self.step_penalty * steps_elapsed
        # reward -= self.accel_penalty * float(np.linalg.norm(accel))
        # reward -= self.jerk_penalty * float(np.linalg.norm(jerk))
        min_link_obstacle_distance = float(np.min(obstacle_distances))
        if min_link_obstacle_distance >= self.safety_distance:
            reward += self.w_plan * self.distance_to_goal()
        if collided:
            reward += self.collision_penalty
        if reached_goal:
            reward += self.goal_reward
        reward += avoidance_reward
        return reward, obstacle_distances

    def _compute_avoidance_reward(self, distances: np.ndarray) -> float:
        """Reward encouraging increasing link-obstacle distances inside the safety zone."""
        prev = self._prev_obstacle_distances
        if prev is None:
            self._prev_obstacle_distances = distances.copy()
            return 0.0

        if self._critical_link_indices.size == 0:
            self._prev_obstacle_distances = distances.copy()
            return 0.0

        current = distances[self._critical_link_indices]
        previous = prev[self._critical_link_indices]
        if current.ndim == 1:
            current = current[:, None]
            previous = previous[:, None]

        current_min = np.min(current, axis=1)
        previous_min = np.min(previous, axis=1)
        close_mask = current_min < self.safety_distance
        if not np.any(close_mask):
            self._prev_obstacle_distances = distances.copy()
            return 0.0

        delta = current_min - previous_min
        self._prev_obstacle_distances = distances.copy()
        delta_close = delta[close_mask]
        weight_close = self._critical_link_weights[close_mask]
        reward = np.sum(weight_close * delta_close)
        return float(self.avoidance_gain * reward)

    def _init_manipulator_ids(self) -> None:  #initialize the manipulator joint and body ids
        self.manip_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.manip_joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in self.manip_joint_names
            ],
            dtype=np.int32,
        )
        self.left_tip_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
        self.right_tip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")

        # qvel index for the manipulator joints
        self.manip_qvel_idx = np.array(
        [self.model.jnt_dofadr[jid] for jid in self.manip_joint_ids],
        dtype=np.int32
    )

        # get the body ids of the manipulator
        body_ids = set(self.model.jnt_bodyid[self.manip_joint_ids].tolist())
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        left_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        self.hand_body_id = hand_id
        body_ids.update([hand_id, left_finger_id, right_finger_id])
        self.manip_body_ids = np.array(sorted(body_ids), dtype=np.int32) 

    def get_joint_velocities(self) -> np.ndarray:  #get the joint velocities of the manipulator
        return self.data.qvel[self.manip_qvel_idx].copy()

        
    def get_grasp_center(self) -> np.ndarray:  # as the position of the end effector     
        L = self.data.site_xpos[self.left_tip_id]
        R = self.data.site_xpos[self.right_tip_id]
        return 0.5 * (L + R)


    def _init_obstacle_ids(self) -> None:  #initialize the mujoco model and data
        self.obstacle_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_box"),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "moving_sphere"),
            ],
            dtype=np.int32,
        )   

    def _init_actuator_ids(self) -> None:  #initialize the actuator ids for the manipulator and obstacles
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

    def _ensure_offscreen_buffer_capacity(self) -> None:
        """Resize MuJoCo's offscreen framebuffer if higher-resolution renders are requested."""
        vis_global = getattr(self.model.vis, "global_", None)
        if vis_global is None:
            return

        desired_width = int(self.render_width)
        desired_height = int(self.render_height)
        current_width = int(getattr(vis_global, "offwidth", 0) or 0)
        current_height = int(getattr(vis_global, "offheight", 0) or 0)

        new_width = max(current_width, desired_width)
        new_height = max(current_height, desired_height)

        updated = False
        if new_width != current_width:
            vis_global.offwidth = new_width
            updated = True
        if new_height != current_height:
            vis_global.offheight = new_height
            updated = True

        if updated:
            warnings.warn(
                "Increased MuJoCo offscreen framebuffer from "
                f"{current_width or 'default'}x{current_height or 'default'} to "
                f"{new_width}x{new_height} for PandaObstacleEnv renders "
                f"({desired_width}x{desired_height}). "
                "Consider setting <visual><global offwidth=\"...\" offheight=\"...\"/></visual> "
                "inside scene_withobstacles.xml for permanent support.",
                RuntimeWarning,
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

    def _set_avoidance_weights(self, weights: Optional[np.ndarray]) -> None:
        num_links = len(self.manip_body_ids)
        num_obstacles = len(self.obstacle_body_ids)
        default_shape = (num_links, num_obstacles)
        if weights is None:
            self._avoidance_weights = np.ones(default_shape, dtype=np.float64)
            return

        arr = np.asarray(weights, dtype=np.float64)
        if arr.shape != default_shape:
            raise ValueError(
                f"avoidance_weights must have shape {default_shape}, got {arr.shape}"
            )
        self._avoidance_weights = arr

    def _configure_critical_links(self) -> None:
        """Select key links near the end-effector and assign avoidance weights."""
        link_index_lookup: Dict[str, int] = {}
        for idx, body_id in enumerate(self.manip_body_ids):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
            if name:
                link_index_lookup[name] = idx

        preferred_names = ("link5", "link6", "link7")
        selected_indices: List[int] = [
            link_index_lookup[name] for name in preferred_names if name in link_index_lookup
        ]

        if not selected_indices:
            num_links = len(self.manip_body_ids)
            selected_indices = list(range(max(0, num_links - 3), num_links))

        if not selected_indices:
            self._critical_link_indices = np.empty(0, dtype=np.int32)
            self._critical_link_weights = np.empty(0, dtype=np.float64)
            return

        base_weights = np.array([50.0, 100.0, 150.0], dtype=np.float64)
        weight_count = len(selected_indices)
        weights = np.empty(weight_count, dtype=np.float64)
        for idx in range(weight_count):
            weights[idx] = base_weights[min(idx, base_weights.size - 1)]

        self._critical_link_indices = np.array(selected_indices, dtype=np.int32)
        self._critical_link_weights = weights

    def seed(self, seed: Optional[int] = None) -> None: #set random seed 
        self._np_random, _ = gym.utils.seeding.np_random(seed)

    def _sample_goal(self) -> np.ndarray: #sample a new goal position within the goal bounds
        low = np.array([0.3, -0.4, 0.3], dtype=np.float64)
        high = np.array([0.6, 0.4, 0.6], dtype=np.float64)
        low, high = self.goal_bounds
        return self._np_random.uniform(low, high)
    
    def _update_obstacles(self, time_value: float) -> None:  #update the obstacle positions based on the time value
        phase = 2.0 * np.pi * self.obstacle_frequency * time_value
        self.data.ctrl[self.box_actuator_id] = self.box_amplitude * np.sin(phase)
        self.data.ctrl[self.sphere_actuator_id] = self.sphere_amplitude * np.sin(
            phase + np.pi / 2.0
        )

    def _get_joint_positions(self) -> np.ndarray:  # get the jonit positions in the world frame
        positions = []
        for body_id in self.model.jnt_bodyid[self.manip_joint_ids]:
            positions.append(self.data.xpos[body_id])
        return np.concatenate(positions)

    def _get_obstacle_positions(self) -> np.ndarray: #get the obstacle positions in the world frame
        return self.data.xpos[self.obstacle_body_ids].ravel()

    def _get_link_positions(self) -> np.ndarray:
        """Return the world-frame positions of all manipulator bodies."""
        return self.data.xpos[self.manip_body_ids]

    def _compute_link_obstacle_distances(self) -> np.ndarray:
        """Compute pairwise distances between links and obstacle centers."""
        link_pos = self._get_link_positions()  # (num_links, 3)
        obstacle_pos = self.data.xpos[self.obstacle_body_ids]  # (num_obstacles, 3)
        diff = link_pos[:, None, :] - obstacle_pos[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def _refresh_obstacle_distance_buffer(self) -> None:
        self._prev_obstacle_distances = self._compute_link_obstacle_distances()

    def _get_obs(self) -> np.ndarray:   #get the observation of the environment
        joint_vel = self.get_joint_velocities()
        joint_pos = self._get_joint_positions()
        obs = np.concatenate(
            [joint_vel, joint_pos, self.goal_pos, self._get_obstacle_positions()]
        )
        return obs.astype(np.float32)
    
    def _compute_accel_and_jerk(self, qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: #calculate the acceleration and jerk based on the current action
        accel = (qvel - self._prev_qvel) / self.dt
        jerk = (accel - self._prev_accel) / self.dt

        # update previous action and acceleration
        self._prev_qvel = qvel
        self._prev_accel = accel

        return accel, jerk

    def _detect_collision(self) -> bool: #detect if there is a collision between the manipulator and the obstacles
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

    def distance_to_goal(self) -> float: #calculate the distance between the end effector and the goal position
        ee_pos = self.get_grasp_center()
        return float(np.linalg.norm(ee_pos - self.goal_pos))



    def reset(         #Reset the simulation, sample a new goal, and return the initial observation.
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.seed(seed)

        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._prev_qvel[:] = 0.0
        self._prev_accel[:] = 0.0

        self.goal_pos = self._sample_goal()
        self.data.ctrl[:] = 0.0
        self._update_obstacles(self.data.time)
        mujoco.mj_forward(self.model, self.data)
        self._refresh_obstacle_distance_buffer()

        obs = self._get_obs()
        info = {"goal": self.goal_pos.copy()}   
        return obs, info
    
    def step(   #Advance the simulation by one step with the given action and compute the reward and termination information.
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, np.ndarray]]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = 0.5 * (a + 1.0) * (self.v_max - self.v_min) + self.v_min

        for _ in range(self.frame_skip):
            self._update_obstacles(self.data.time)
            self.data.ctrl[self.manip_actuator_ids] = v_cmd
            mujoco.mj_step(self.model, self.data)

        joint_vel = self.get_joint_velocities()
        accel, jerk = self._compute_accel_and_jerk(joint_vel)

        self._step_count += 1

        collided = self._detect_collision()
        curr_goal_dist = self.distance_to_goal()
        reached_goal = curr_goal_dist <= self.goal_reach_threshold
        reward, obstacle_distances = self._compute_reward(accel, jerk, collided, reached_goal)

        obs = self._get_obs()
        terminated = reached_goal or collided
        
        truncated = self._step_count >= self.max_episode_steps

        info: Dict[str, np.ndarray] = {
            "goal": self.goal_pos.copy(),
            "distance_to_goal": np.array([self.distance_to_goal()], dtype=np.float64),
            "grasp_center": self.get_grasp_center(),
            "collided": np.array([collided], dtype=bool),
            "is_success": np.array([reached_goal], dtype=bool),
            "min_link_obstacle_distance": np.array([np.min(obstacle_distances)], dtype=np.float64),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray: #render the environment and return an RGB image
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, width=self.render_width, height=self.render_height)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None: #close the renderer if it exists
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
