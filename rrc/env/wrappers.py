"""Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import os
import os.path as osp
import time

import cv2
import gym
import numpy as np
import pybullet as p
from collections import OrderedDict
from gym import ObservationWrapper
from gym.spaces import flatten_space
from rrc.env.cube_env import Action, ActionType
from rrc.env.env_utils import PolicyMode
from rrc.env import cube_env
from scipy.spatial.transform import Rotation
from stable_baselines3.common.monitor import Monitor
from trifinger_simulation import TriFingerPlatform, camera, trifingerpro_limits
from trifinger_simulation.tasks import move_cube

try:
    from xvfbwrapper import Xvfb
except ImportError:
    Xvfb = None

EXCEP_MSSG = (
    "================= captured exception =================\n"
    + "{message}\n"
    + "{error}\n"
    + "=================================="
)


class frameskip_to:
    """
    A Context Manager that sets action type and action space temporally
    This applies to all wrappers and the origianl environment recursively
    """

    def __init__(self, frameskip, env):
        self.frameskip = frameskip
        self.env = env
        self.org_frameskip = env.unwrapped.frameskip

    def __enter__(self):
        self.env.unwrapped.frameskip = self.frameskip

    def __exit__(self, type, value, traceback):
        self.env.unwrapped.frameskip = self.org_frameskip


class action_type_to:
    """
    A Context Manager that sets action type and action space temporally
    This applies to all wrappers and the origianl environment recursively
    """

    def __init__(self, action_type, env):
        self.action_type = action_type
        self.action_space = self._get_action_space(action_type)
        self.get_config(env)
        self.env = env

    def get_config(self, env):
        self.orig_action_spaces = [env.action_type]
        self.orig_action_types = [env.action_space]
        while hasattr(env, "env"):
            env = env.env
            self.orig_action_types.append(env.action_type)
            self.orig_action_spaces.append(env.action_space)

    def __enter__(self):
        env = self.env
        env.action_space = self.action_space
        env.action_type = self.action_type
        while hasattr(env, "env"):
            env = env.env
            env.action_space = self.action_space
            env.action_type = self.action_type

    def __exit__(self, type, value, traceback):
        ind = 0
        env = self.env
        env.action_space = self.orig_action_spaces[ind]
        env.action_type = self.orig_action_types[ind]
        while hasattr(env, "env"):
            ind += 1
            env = env.env
            env.action_space = self.orig_action_spaces[ind]
            env.action_type = self.orig_action_types[ind]

    def _get_action_space(self, action_type):
        spaces = TriFingerPlatform.spaces
        if action_type == ActionType.TORQUE:
            action_space = spaces.robot_torque.gym
        elif action_type == ActionType.POSITION:
            action_space = spaces.robot_position.gym
        elif action_type == ActionType.TORQUE_AND_POSITION:
            action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError("unknown action type")
        return action_space


class NewToOldObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "tip_force",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": env.observation_space["robot"]["position"],
                "robot_velocity": env.observation_space["robot"]["velocity"],
                "robot_torque": env.observation_space["robot"]["torque"],
                "robot_tip_positions": env.observation_space["robot"]["tip_positions"],
                "object_position": env.observation_space["achieved_goal"]["position"],
                "object_orientation": env.observation_space["achieved_goal"][
                    "orientation"
                ],
                "goal_object_position": env.observation_space["desired_goal"][
                    "position"
                ],
                "goal_object_orientation": env.observation_space["desired_goal"][
                    "orientation"
                ],
                "tip_force": env.observation_space["robot"]["tip_force"],
                "action_torque": env.observation_space["robot"]["torque"],
                "action_position": env.observation_space["robot"]["position"],
            }
        )

    def observation(self, obs):
        old_obs = {
            "robot_position": obs["robot"]["position"],
            "robot_velocity": obs["robot"]["velocity"],
            "robot_torque": obs["robot"]["torque"],
            "robot_tip_positions": obs["robot"]["tip_positions"],
            "tip_force": obs["robot"]["tip_force"],
            "object_position": obs["achieved_goal"]["position"],
            "object_orientation": obs["achieved_goal"]["orientation"],
            "goal_object_position": obs["desired_goal"]["position"],
            "goal_object_orientation": obs["desired_goal"]["orientation"],
        }
        if self.action_space == self.observation_space["robot_position"]:
            old_obs["action_torque"] = np.zeros_like(obs["action"])
            old_obs["action_position"] = obs["action"]
        elif self.action_space == self.observation_space["robot_torque"]:
            old_obs["action_torque"] = obs["action"]
            old_obs["action_position"] = np.zeros_like(obs["action"])
        else:
            old_obs["action_torque"] = obs["action"]["torque"]
            old_obs["action_position"] = obs["action"]["position"]
        return old_obs


class AdaptiveActionSpaceWrapper(gym.Wrapper):
    """Create a unified action space for torque and position control."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.robot_position.low,
                    high=trifingerpro_limits.robot_position.high,
                ),
                "torque": gym.spaces.Box(
                    low=trifingerpro_limits.robot_torque.low,
                    high=trifingerpro_limits.robot_torque.high,
                ),
                "frameskip": gym.spaces.Box(low=np.zeros(1), high=np.inf * np.ones(1)),
            }
        )

    def _clip_action(self, action):
        clipped_action = {
            "torque": None,
            "position": None,
            "frameskip": action["frameskip"],
        }
        if action["torque"] is not None:
            clipped_action["torque"] = np.clip(
                action["torque"],
                self.action_space["torque"].low,
                self.action_space["torque"].high,
            )
        if action["position"] is not None:
            clipped_action["position"] = np.clip(
                action["position"],
                self.action_space["position"].low,
                self.action_space["position"].high,
            )
        return clipped_action

    def step(self, action):
        action = self._clip_action(action)
        with frameskip_to(action["frameskip"], self.env):
            if action["torque"] is None:
                with action_type_to(ActionType.POSITION, self.env):
                    return self.env.step(action["position"])
            elif action["position"] is None:
                with action_type_to(ActionType.TORQUE, self.env):
                    return self.env.step(action["torque"])
            else:
                with action_type_to(ActionType.TORQUE_AND_POSITION, self.env):
                    return self.env.step(
                        {"position": action["position"], "torque": action["torque"]}
                    )


class TimingWrapper(gym.Wrapper):
    """Set timing constraints for realtime control based on the frameskip and
    action frequency."""

    def __init__(self, env, dt):
        super().__init__(env)
        self.dt = dt

    def reset(self):
        self.t = None
        self.frameskip = None
        return self.env.reset()

    def step(self, action):
        if self.t is not None:
            elapsed_time = time.time() - self.t
            min_elapsed_time = self.frameskip * self.dt
            if elapsed_time < min_elapsed_time:
                time.sleep(min_elapsed_time - elapsed_time)

        self.t = time.time()
        self.frameskip = action["frameskip"]
        return self.env.step(action)


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cameras = camera.TriFingerCameras(image_size=(360, 270))
        self.metadata = {"render.modes": ["rgb_array"]}
        self._initial_reset = True
        self._accum_reward = 0
        self._reward_at_step = 0

    def reset(self):
        import pybullet as p

        obs = self.env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0],
        )
        self._accum_reward = 0
        self._reward_at_step = 0
        if self._initial_reset:
            self._episode_idx = 0
            self._initial_reset = False
        else:
            self._episode_idx += 1
        return obs

    def step(self, action):
        observation, reward, is_done, info = self.env.step(action)
        self._accum_reward += reward
        self._reward_at_step = reward
        return observation, reward, is_done, info

    def render(self, mode="rgb_array", **kwargs):
        assert mode == "rgb_array", "RenderWrapper Only supports rgb_array mode"
        images = (
            self.cameras.cameras[0].get_image(),
            self.cameras.cameras[1].get_image(),
        )
        height = images[0].shape[1]
        two_views = np.concatenate((images[0], images[1]), axis=1)
        two_views = cv2.putText(
            two_views,
            "step_count: {:06d}".format(self.env.unwrapped.step_count),
            (10, 40),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        two_views = cv2.putText(
            two_views,
            "episode: {}".format(self._episode_idx),
            (10, 70),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        two_views = cv2.putText(
            two_views,
            "reward: {:.2f}".format(self._reward_at_step),
            (10, height - 130),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        two_views = cv2.putText(
            two_views,
            "acc_reward: {:.2f}".format(self._accum_reward),
            (10, height - 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        return two_views


class PyBulletClearGUIWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(PyBulletClearGUIWrapper, self).__init__(env)
        self.camera_kwargs = kwargs

    def reset(self, camera_kwargs={}, **kwargs):
        cam_kwargs = dict(
            cameraDistance=0.6,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0],
        )
        cam_kwargs.update(self.camera_kwargs)
        cam_kwargs.update(camera_kwargs)
        obs = self.env.reset(**kwargs)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(**cam_kwargs)
        return obs


class MonitorPyBulletWrapper(gym.Wrapper):
    def __init__(self, env, save_dir, save_freq=1):
        env = PyBulletClearGUIWrapper(env)
        super(MonitorPyBulletWrapper, self).__init__(env)
        assert (
            Xvfb is not None
        ), "xvfbwrapper not installed, make sure `pip install xvfbwrapper` is called"
        assert (
            env.unwrapped.visualization
        ), "passed MonitorPyBullet env with visualization=False"
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.xvfb = None
        self.videos = []
        self.episode_count = 0
        self.recording = False
        if save_dir is not None and not osp.exists(save_dir):
            os.makedirs(save_dir)

    def reset(self):
        if self.recording:
            self.stop_recording()
        self.episode_count += 1
        if self.xvfb is None:
            env_display = os.environ.get("DISPLAY", "")
            if not env_display or "localhost" in env_display:
                self.xvfb = Xvfb()
                self.xvfb.start()
        obs = self.env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if (self.episode_count - 1) % self.save_freq == 0:
            filepath = self.create_filepath()
            p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4,
                filepath,
                physicsClientId=self.env.unwrapped._pybullet_client_id,
            )
            self.recording = True
        p.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0, 0, 0],
        )
        return obs

    def step(self, action):
        obs, r, d, i = super(MonitorPyBulletWrapper, self).step(action)
        if d and self.recording:
            self.stop_recording()
        return obs, r, d, i

    def stop_recording(self):
        self.recording = False
        if self.env.unwrapped._pybullet_client_id >= 0 and len(self.videos) > 0:
            print("Stopping recording {}".format(self.videos[-1]))
            if (self.episode_count - 1) % self.save_freq == 0:
                p.stopStateLogging(
                    p.STATE_LOGGING_VIDEO_MP4,
                    physicsClientId=self.env.unwrapped._pybullet_client_id,
                )
            # self.env.unwrapped._disconnect_from_pybullet()

    def close(self):
        if self.xvfb:
            self.xvfb.stop()
        super().close()

    def create_filepath(self):
        # MP4 logging
        filepath = lambda i: osp.join(self.save_dir, f"sim-{i}.mp4")
        for i in range(25):
            if not osp.exists(filepath(i)):
                break
        filepath = filepath(i)
        self.videos.append(filepath)
        return filepath


class ResidualPDWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        Kp=np.eye(3) * np.array([100, 100, 1]),
        Kd=1,
        include_ac=False,
        force_factor=0.5,
        torque_factor=0.1,
    ):
        super(ResidualPDWrapper, self).__init__(env)
        if not isinstance(Kp, np.ndarray) or Kp.shape != (3, 3):
            Kp = np.eye(3) * Kp
        self.Kp = Kp
        self.Kd = Kd
        self._obs = self._prev_obs = None
        self.include_ac = include_ac
        self.force_factor = force_factor
        self.torque_factor = torque_factor
        if self.include_ac:
            obs_dict = self.env.observation_space.spaces
            obs_dict = {k: obs_dict["observation"][k] for k in obs_dict["observation"]}
            obs_dict["pd_action"] = gym.spaces.Box(low=-np.ones(3), high=np.ones(3))
            self.observation_space.spaces["observation"] = obs_dict

    def reset(self):
        obs = super(ResidualPDWrapper, self).reset()
        self._prev_obs = None
        self._obs = obs
        if self.include_ac:
            obs["observation"]["pd_action"] = self.pd_action(self._obs, self._prev_obs)
        return obs

    def step(self, action):
        action[:3] *= self.force_factor
        action[3:] *= self.torque_factor
        # TODO: restore residual actions
        pd_action = self.pd_action(self._obs, self._prev_obs)
        # ac = action + pd_action
        ac = np.clip(pd_action, -1, 1)
        self._prev_obs = self._obs
        obs, r, d, i = self.env.step(ac)
        self._obs = obs
        if self.include_ac:
            obs["observation"]["pd_action"] = self.pd_action(self._obs, self._prev_obs)
        return obs, r, d, i

    def pd_action(self, observation, prev_observation):
        if prev_observation is None:
            return np.zeros(6)
        if observation["observation"].get("pd_action") is not None:
            return observation["observation"]["pd_action"]
        err = (
            -observation["desired_goal"]["position"]
            + observation["achieved_goal"]["position"]
        )

        u = -self.Kp @ err
        if prev_observation is None:
            return np.concatenate([u, np.zeros(3)], axis=-1)
        err_diff = err - (
            -prev_observation["desired_goal"]["position"]
            + prev_observation["achieved_goal"]["position"]
        )
        u -= self.Kd * err_diff / self.env.time_step_s
        return np.concatenate([u, np.zeros(3)], axis=-1)


class FlattenGoalObs(ObservationWrapper):
    def __init__(self, env, observation_keys):
        super().__init__(env)
        obs_space = self.env.observation_space
        obs_dict = OrderedDict(
            [(k, flatten_space(obs_space[k])) for k in observation_keys]
        )
        self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, obs):
        n_obs = {}
        for k in self.observation_space.spaces:
            if isinstance(obs[k], dict):
                obs_list = [
                    obs[k][k2].flatten() for k2 in self.env.observation_space[k]
                ]
                n_obs[k] = np.concatenate(obs_list)
            else:
                n_obs[k] = obs[k]
        return n_obs


class PyBulletMonitor(gym.Wrapper):

    _render_width = 256
    _render_height = 192
    _cam_dist = 0.65
    _cam_yaw = 0.0
    _cam_pitch = -45.0

    def __init__(self, env):
        super(PyBulletMonitor, self).__init__(env)
        self.metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 60}

    @property
    def _pybullet_client_id(self):
        return self.platform.simfinger._pybullet_client_id

    def camera_adjust(self):
        pass

    def render(self, mode="human", close=False):
        if mode == "human":
            self.isRender = True
        if self._pybullet_client_id >= 0:
            self.camera_adjust()

        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]

        if self._pybullet_client_id >= 0:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2,
                physicsClientId=self._pybullet_client_id,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=float(self._render_width) / self._render_height,
                nearVal=0.1,
                farVal=100.0,
                physicsClientId=self._pybullet_client_id,
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=self._render_width,
                height=self._render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self._pybullet_client_id,
            )

            p.configureDebugVisualizer(
                p.COV_ENABLE_SINGLE_STEP_RENDERING,
                1,
                physicsClientId=self._pybullet_client_id,
            )
        else:
            px = np.array(
                [[[255, 255, 255, 255]] * self._render_width] * self._render_height,
                dtype=np.uint8,
            )
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(
            np.array(px), (self._render_height, self._render_width, -1)
        )
        rgb_array = rgb_array[:, :, :3]
        return rgb_array


class HierarchicalPolicyWrapper(ObservationWrapper):
    def __init__(self, env, policy):
        assert isinstance(
            env.unwrapped, cube_env.RealRobotCubeEnv
        ), "env expects type CubeEnv or RealRobotCubeEnv"
        self.env = env
        self.reward_range = self.env.reward_range
        # set observation_space and action_space below
        spaces = TriFingerPlatform.spaces
        self._action_space = gym.spaces.Dict(
            {"torque": spaces.robot_torque.gym, "position": spaces.robot_position.gym}
        )
        self._last_action = np.zeros(9)
        self.set_policy(policy)
        self._platform = None

    @property
    def impedance_control_mode(self):
        return self.mode == PolicyMode.IMPEDANCE or (
            self.mode == PolicyMode.RL_PUSH and self.rl_observation_space is None
        )

    @property
    def action_space(self):
        if self.impedance_control_mode:
            return self._action_space["torque"]
        else:
            return self.wrapped_env.action_space

    @property
    def action_type(self):
        if self.impedance_control_mode:
            return ActionType.TORQUE
        else:
            return ActionType.POSITION

    @property
    def mode(self):
        assert self.policy, "Need to first call self.set_policy() to access mode"
        return self.policy.mode

    @property
    def frameskip(self):
        if self.mode == PolicyMode.RL_PUSH:
            return self.policy.rl_frameskip
        return 4

    @property
    def step_count(self):
        return self.env.step_count

    @step_count.setter
    def step_count(self, v):
        self.env.step_count = v

    def set_policy(self, policy):
        self.policy = policy
        if policy:
            self.rl_observation_names = policy.observation_names
            self.rl_observation_space = policy.rl_observation_space
            obs_dict = {"impedance": self.env.observation_space}
            if self.rl_observation_space is not None:
                obs_dict["rl"] = self.rl_observation_space
            self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, observation):
        obs_dict = {"impedance": observation}
        if "rl" in self.observation_space.spaces:
            observation_rl = self.process_observation_rl(observation)
            obs_dict["rl"] = observation_rl
        return obs_dict

    def get_goal_object_ori(self, obs):
        val = obs["desired_goal"]["orientation"]
        goal_rot = Rotation.from_quat(val)
        actual_rot = Rotation.from_quat(np.array([0, 0, 0, 1]))
        y_axis = [0, 1, 0]
        actual_vector = actual_rot.apply(y_axis)
        goal_vector = goal_rot.apply(y_axis)
        N = np.array([0, 0, 1])
        proj = goal_vector - goal_vector.dot(N) * N
        proj = proj / np.linalg.norm(proj)
        ori_error = np.arccos(proj.dot(actual_vector))
        xyz = np.zeros(3)
        xyz[2] = ori_error
        val = Rotation.from_euler("xyz", xyz).as_quat()
        return val

    def process_observation_rl(self, obs, return_dict=False):
        t = self.step_count
        obs_dict = {}
        cpu = self.policy.impedance_controller.custom_pinocchio_utils
        for on in self.rl_observation_names:
            if on == "robot_position":
                val = obs["observation"]["position"]
            elif on == "robot_velocity":
                val = obs["observation"]["velocity"]
            elif on == "robot_tip_positions":
                val = cpu.forward_kinematics(obs["observation"]["position"]).flatten()
            elif on == "object_position":
                val = obs["achieved_goal"]["position"]
            elif on == "object_orientation":
                actual_rot = Rotation.from_quat(obs["achieved_goal"]["orientation"])
                xyz = actual_rot.as_euler("xyz")
                xyz[:2] = 0.0
                val = Rotation.from_euler("xyz", xyz).as_quat()
                val = obs["achieved_goal"]["orientation"]
            elif on == "goal_object_position":
                val = 0 * np.asarray(obs["desired_goal"]["position"])
            elif on == "goal_object_orientation":
                # disregard x and y axis rotation for goal_orientation
                val = self.get_goal_object_ori(obs)
            elif on == "relative_goal_object_position":
                val = 0.0 * np.asarray(obs["desired_goal"]["position"]) - np.asarray(
                    obs["achieved_goal"]["position"]
                )
            elif on == "relative_goal_object_orientation":
                goal_rot = Rotation.from_quat(self.get_goal_object_ori(obs))
                actual_rot = Rotation.from_quat(obs_dict["object_orientation"])
                if self.policy.rl_env.use_quat:
                    val = (goal_rot * actual_rot.inv()).as_quat()
                else:
                    val = get_theta_z_wf(goal_rot, actual_rot)
            elif on == "action":
                val = self._last_action
                if isinstance(val, dict):
                    val = val["torque"]
            obs_dict[on] = np.asarray(val, dtype="float64").flatten()
        if return_dict:
            return obs_dict

        self._prev_obs = obs_dict
        obs = np.concatenate([obs_dict[k] for k in self.rl_observation_names])
        return obs

    def reset(self, **platform_kwargs):
        self.resetting = True
        if self._platform is None:
            initial_object_pose = move_cube.sample_goal(-1)
            self._platform = trifinger_simulation.TriFingerPlatform(
                visualization=False,
                initial_object_pose=initial_object_pose,
            )
        obs = super(HierarchicalPolicyWrapper, self).reset(**platform_kwargs)
        initial_object_pose = move_cube.Pose.from_dict(
            obs["impedance"]["achieved_goal"]
        )
        # initial_object_pose = move_cube.sample_goal(difficulty=-1)
        self.policy.reset_policy(obs["impedance"], self._platform)
        self._prev_action = np.zeros(9)
        self.resetting = False
        return obs

    def _step(self, action):
        if self.unwrapped.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError("Given action is not contained in the action space.")

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > self.episode_length:
            excess = step_count_after - self.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            self.step_count = t = self.unwrapped.platform.append_desired_action(
                robot_action
            )

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            if osp.exists("/output"):
                observation = self.unwrapped._create_observation(t, action)
            else:
                observation = self.unwrapped._create_observation(t + 1, action)

            reward += self.unwrapped.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.unwrapped.info,
            )

            if self.step_count >= self.episode_length:
                break

        is_done = self.step_count == self.episode_length
        info = self.env.info
        info["num_steps"] = self.step_count
        return observation, reward, is_done, info

    def _gym_action_to_robot_action(self, gym_action):
        if self.action_type == ActionType.TORQUE:
            robot_action = Action(torque=gym_action, position=np.repeat(np.nan, 9))
        elif self.action_type == ActionType.POSITION:
            robot_action = Action(position=gym_action, torque=np.zeros(9))
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def scale_action(self, action, wrapped_env):
        obs = self._prev_obs
        poskey, velkey = "robot_position", "robot_velocity"
        current_position, _ = obs[poskey], obs[velkey]
        if wrapped_env.relative:
            goal_position = current_position + 0.8 * wrapped_env.scale * action
            pos_low, pos_high = (
                wrapped_env.env.action_space.low,
                wrapped_env.env.action_space.high,
            )
        else:
            pos_low, pos_high = (
                wrapped_env.spaces.robot_position.low,
                wrapped_env.spaces.robot_position.high,
            )
            pos_low = np.max([current_position - wrapped_env.scale, pos_low], axis=0)
            pos_high = np.min([current_position + wrapped_env.scale, pos_high], axis=0)
            goal_position = action
        action = np.clip(goal_position, pos_low, pos_high)
        self._clipped_action = np.abs(action - goal_position)
        return action

    def step(self, action):
        # RealRobotCubeEnv handles gym_action_to_robot_action
        # print(self.mode)
        self._last_action = action
        self.unwrapped.frameskip = self.frameskip

        obs, r, d, i = self._step(action)
        obs = self.observation(obs)
        return obs, r, d, i

    def get_goal_object_pose(self, observation):
        goal_pose = self.unwrapped.goal
        goal_pose = move_cube.Pose.from_dict(goal_pose)
        if not isinstance(observation, dict):
            observation = self.unflatten_observation(observation)
        pos, ori = (
            observation["object_position"],
            observation["object_orientation"],
        )
        object_pose = move_cube.Pose(position=pos, orientation=ori)
        return goal_pose, object_pose


class WrenchPolicyWrapper(ObservationWrapper):
    def __init__(self, env):
        assert isinstance(
            env.unwrapped, cube_env.RobotWrenchCubeEnv
        ), "env expects type CubeEnv or RobotWrenchCubeEnv"
        super(WrenchPolicyWrapper, self).__init__(env)

        self.observation_space["observation"].spaces["cp_list"] = gym.spaces.Box(
            low=np.concatenate([trifingerpro_limits.object_position.low, np.zeros(4)]),
            high=np.concatenate([trifingerpro_limits.object_position.high, np.ones(4)]),
        )

    def observation(self, observation):
        obj_pose = move_cube.Pose.from_dict(observation["achieved_goal"])
        observation["observation"]["cp_list"] = np.array(
            [
                np.concatenate(x)
                for x in self.get_cp_of_list(
                    self.env.cp_params, obj_pose, self.env.use_actual_cp
                )
            ]
        )
        return observation
