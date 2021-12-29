import gym
import numpy as np
import torch
import pybullet as p
import rrc.cic.control_main_class as cic_control_main_class
from copy import deepcopy
from typing import Callable, Union

from gym import logger as gymlogger
from rrc.env.env_utils import ContactResult
from rrc.env import fop
from rrc_iprl_package.control import controller_utils_cube as c_utils
from rrc.mp.const import (
    CUBE_HALF_WIDTH,
    CUBE_MASS,
    CUBE_WIDTH,
    CUBOID_MASS,
    CUBOID_SIZE,
    CUSTOM_LOGDIR,
    INIT_JOINT_CONF,
)
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation
from .cube_env_real import ActionType
from .reward_fns import (
    _corner_error,
    _position_error,
    _tip_distance_to_cube,
    competition_reward,
    training_reward5,
)
from .viz import CuboidMarker, VisualMarkers, Viz
from trifinger_simulation import trifingerpro_limits
from trifinger_simulation.tasks import move_cube


class RobotCubeEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        action_type: ActionType = ActionType.POSITION,
        frameskip: int = 1,
        time_step_s: float = 0.004,
        sim: bool = True,
        gravity: float = -9.81,
        visualization: bool = False,
        reward_fn: callable = competition_reward,
        termination_fn: callable = None,
        initializer: callable = None,
        episode_length: int = move_cube.episode_length,
        path: str = None,
        debug: bool = False,
        action_scale: Union[float, np.ndarray] = None,
        contact_timeout: int = 50,
        min_tip_dist: float = 0.11,
        object_mass: float = 0.016,
        use_benchmark_controller: bool = False,
        return_timestamp: bool = False,
        tip_positions_object_frame: bool = False,
    ):
        """Initialize.

        Args:
            cube_goal_pose (dict): Goal pose for the cube.  Dictionary with
                keys "position" and "orientation".
            goal_difficulty (int): Difficulty level of the goal (needed for
                reward computation).
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            time_step_s (float): length of one timestep in simulator
            sim (bool): whether or not real robot trifinger platform is used
            gravity (float): gravity constant used in sim
            visualization (bool): whether or not to visualize simulator
            reward_fn (callable): function to compute reward from observation
            termination_fn (callable): function to compute end of episode from observation
            initializer (callable): function to sample goal and initial positions
            episode_length (int): length of an episode
            path (str): path to save logs of observations, rewards, and custom metrics
            debug (bool): print debug lines
            action_scale (Union[float, np.ndarray]): constant factor to scale actions
            contact_timeout (int): steps until episode is terminated early
            min_tip_dist (float): minimum tip distance to reach to aviod timeout
            object_mass (float): set object mass to custom value
            use_benchmark_controller (bool): use benchmark-rrc cartesian impedance controller
            return_timestamp (bool): returns cam0_timestamp with observation
            tip_positions_object_frame (bool): if true, observations contain tip positions in object frame
        """
        # Basic initialization
        # ====================
        self.path = path
        self.debug = debug
        if debug:
            gymlogger.set_level(10)
            np.set_printoptions(3)

        if gravity is None:
            gravity = -9.81
        self._gravity = gravity
        self.object_mass = object_mass
        self.tip_positions_object_frame = tip_positions_object_frame

        self._compute_reward = reward_fn
        self._termination_fn = termination_fn if sim else None
        self.initializer = initializer if sim else None
        if cube_goal_pose is None:
            cube_goal_pose = self.initializer.get_goal().to_dict()
        self.goal = {
            k: np.array(v, dtype=np.float32) for k, v in cube_goal_pose.items()
        }
        self.info = {"difficulty": goal_difficulty}
        self.difficulty = goal_difficulty

        self.action_type = action_type
        if use_benchmark_controller:
            self.main_ctrl = cic_control_main_class.TriFingerController(self)
        else:
            self.main_ctrl = None
        self.return_timestamp = return_timestamp

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip
        self.time_step_s = time_step_s
        self.contact_timeout = contact_timeout
        self._min_tip_dist = self._last_tip_dist = min_tip_dist
        self._tip_dist_buffer = deque(maxlen=10)

        # will be initialized in reset()
        self.real_platform = None
        self.platform = None
        self._pybullet_client_id = -1
        self.simulation = sim
        self.visualization = visualization
        self.episode_length = episode_length
        self.custom_logs = {}
        self.reward_list = []
        self.observation_list = []
        self.change_goal_last = -1  # needed for evaluation
        self.reach_finish_point = -1  # needed for evaluation
        self.reach_start_point = -1  # needed for evaluation
        self.init_align_obj_error = -1  # needed for evaluation
        self.init_obj_pose = None  # needed for evaluation
        self.align_obj_error = -1  # needed for evaluation
        self.goal_list = []  # needed for multiple goal environments
        if self.visualization:
            self.cube_viz = Viz()

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )

        # verify that the given goal pose is contained in the cube state space
        if not object_state_space.contains(self.goal):
            raise ValueError("Invalid goal pose.")

        if self.action_type == ActionType.TORQUE:
            if action_scale is None:
                self.action_space = robot_torque_space
            else:
                self.action_space = gym.spaces.Box(
                    low=robot_torque_space.low * action_scale,
                    high=robot_torque_space.high * action_scale,
                )

            self.initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            if action_scale is None:
                self.action_space = robot_position_space
            else:
                self.action_space = gym.spaces.Box(
                    low=robot_position_space.low * action_scale,
                    high=robot_position_space.high * action_scale,
                )
            self.initial_action = (
                INIT_JOINT_CONF  # trifingerpro_limits.robot_position.default
            )
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self.initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": INIT_JOINT_CONF,  # trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")
        obs_space = {
            "position": robot_position_space,
            "velocity": robot_velocity_space,
            "torque": robot_torque_space,
            "tip_positions": gym.spaces.Box(
                low=np.array([trifingerpro_limits.object_position.low] * 3),
                high=np.array([trifingerpro_limits.object_position.high] * 3),
            ),
            "tip_force": gym.spaces.Box(low=np.zeros(3), high=np.ones(3)),
            "action": self.action_space,
        }
        if return_timestamp:
            obs_space["cam0_timestamp"] = gym.spaces.Box(low=0, high=np.inf, shape=())
        obs_space = gym.spaces.Dict(obs_space)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": obs_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

        self.pinocchio_utils = PinocchioUtils()
        self.prev_observation = None
        self._prev_step_report = 0

    @property
    def min_tip_dist(self):
        return self._min_tip_dist

    @min_tip_dist.setter
    def min_tip_dist(self, x):
        if x >= 0.07:  # min_tip_dist limit
            self._min_tip_dist = x

    def compute_reward(self, achieved_goal, desired_goal, info):
        if not isinstance(info, dict):
            p_obs = [d["p_obs"] for d in info]
            obs = [d["obs"] for d in info]
            for i, (p_d, d) in enumerate(zip(p_obs, obs)):
                virtual_goal = desired_goal[i]
                if not isinstance(desired_goal, dict):
                    virtual_goal = {
                        "position": virtual_goal[4:],
                        "orientation": virtual_goal[:4],
                    }
                d["desired_goal"] = virtual_goal
                p_d["desired_goal"] = virtual_goal
            return np.array(
                [self._compute_reward(p, o, i) for p, o, i in zip(p_obs, obs, info)]
            )
        else:
            p_obs, obs = info["p_obs"], info["obs"]
            if not isinstance(desired_goal, dict):
                desired_goal = {
                    "position": desired_goal[4:],
                    "orientation": desired_goal[:4],
                }
            obs["desired_goal"] = desired_goal
            return self._compute_reward(p_obs, obs, info)

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float) : amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the difficulty level of
              the goal.
        """
        if self.real_platform is None and not self.simulation:
            raise RuntimeError("Call `reset()` before starting to step.")

        if self.platform is None:
            raise RuntimeError("platform is not instantiated.")

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
            if self.simulation:
                t = self.platform.append_desired_action(robot_action)
                observation = self._create_observation(t, action)
                self.observation_list.append(observation)
            else:
                t = self.real_platform.append_desired_action(robot_action)
                observation = self._create_observation(t, action)
                self._set_sim_state(observation)

            if self.prev_observation is None:
                self.prev_observation = observation
            p_obs = deepcopy(self.prev_observation.copy())
            reward += self._compute_reward(
                self.prev_observation, observation, self.info
            )
            self.prev_observation = observation

            self.step_count += t - self.t_prev
            self.t_prev = t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length:
                break

        info = self.info.copy()
        info = self.get_info_keys(info, observation, p_obs)

        # if self._last_tip_dist <= self._min_tip_dist:
        #    self.steps_since_contact = self.step_count
        is_done = self.step_count >= self.episode_length
        # if (self.step_count - self.steps_since_contact) >= self.contact_timeout:
        #     is_done = True
        if termination_fn(observation):
            is_done = True
        if self._termination_fn is not None:
            term_bonus = 500
            is_done = is_done or self._termination_fn(observation)
            info["is_success"] = self._termination_fn(observation)
            reward += term_bonus * self._termination_fn(observation)
        else:
            info["is_success"] = pos_and_rot_close_to_goal(observation)

        # report current step_count
        if self.step_count - self._prev_step_report > 200:
            gymlogger.debug("current step_count: %d", self.step_count)
            self._prev_step_report = self.step_count

        if is_done:
            gymlogger.debug("is_done is True. Episode terminates.")
            gymlogger.debug(
                "Episode terminatated: episode length %d, step_count %d",
                self.episode_length,
                self.step_count,
            )
            self.save_custom_logs()

        if self.visualization:
            self.cube_viz.update_cube_orientation(
                observation["achieved_goal"]["position"],
                observation["achieved_goal"]["orientation"],
                observation["desired_goal"]["position"],
                observation["desired_goal"]["orientation"],
            )
            time.sleep(0.01)

        self.reward_list.append(reward)

        return deepcopy(observation), reward, is_done, info

    def get_info_keys(self, info, obs, p_obs):
        info["ori_err"] = _orientation_error(obs)
        info["pos_err"] = _position_error(obs)
        info["corner_err"] = _corner_error(obs)
        info["tip_pos_err"] = np.linalg.norm(
            obs["observation"]["tip_positions"] - obs["achieved_goal"]["position"],
            axis=1,
        )
        info["tot_tip_pos_err"] = _tip_distance_to_cube(obs)
        self._last_tip_dist = np.mean(info["tip_pos_err"])

        # TODO (cleanup): Skipping keys to avoid storing unnecessary keys in RB
        info["p_obs"] = {
            k: p_obs[k] for k in ["desired_goal", "achieved_goal", "observation"]
        }
        info["obs"] = {
            k: obs[k] for k in ["desired_goal", "achieved_goal", "observation"]
        }
        return info

    def reset(self):
        # By changing the `_reset_*` method below you can switch between using
        # the platform frontend, which is needed for the submission system, and
        # the direct simulation, which may be more convenient if you want to
        # pre-train locally in simulation.
        if self.simulation:
            self._reset_direct_simulation()
            cam_kwargs = dict(
                cameraDistance=0.6,
                cameraYaw=0,
                cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0],
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self._pybullet_client_id
            )
            p.resetDebugVisualizerCamera(
                **cam_kwargs, physicsClientId=self._pybullet_client_id
            )
            self._tip_dist_buffer.append(self._last_tip_dist)
        else:
            self._reset_platform_frontend()
            self._reset_direct_simulation()
        if self.visualization:
            self.cube_viz.reset()

        p.setGravity(
            0,
            0,
            self._gravity,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
        )

        self.maybe_update_tip_dist()
        self.step_count = 0

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        self.prev_observation, _, _, _ = self.step(self.initial_action)
        if self.path is not None:
            self.set_reach_start()
        return self.prev_observation

    def maybe_update_tip_dist(self):
        if (
            len(self._tip_dist_buffer) == 10
            and np.mean(self._tip_dist_buffer) < self.min_tip_dist
        ):
            gymlogger.debug(
                "Changing minimum tip distance to: %1.3f", self.min_tip_dist
            )
            self.min_tip_dist -= 0.01
            self._tip_dist_buffer.clear()

        self.steps_since_contact = 0
        return

    def _reset_platform_frontend(self):
        """Reset the platform frontend."""
        # reset is not really possible
        if self.real_platform is not None:
            raise RuntimeError("Once started, this environment cannot be reset.")

        self.real_platform = robot_fingers.TriFingerPlatformFrontend()

    def _reset_direct_simulation(self):
        """Reset direct simulation.

        With this the env can be used without backend.
        """

        # reset simulation
        del self.platform

        # initialize simulation
        if self.initializer is None:
            initial_object_pose = move_cube.sample_goal(difficulty=-1)
        else:
            initial_object_pose = self.initializer.get_initial_state()
            cube_goal_pose = self.initializer.get_goal().to_dict()
            self.goal = {
                k: np.array(v, dtype=np.float32) for k, v in cube_goal_pose.items()
            }

        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_object_pose=initial_object_pose,
            time_step_s=self.time_step_s,
            object_mass=self.object_mass,
        )
        self.t_prev = 0
        self._pybullet_client_id = self.platform.simfinger._pybullet_client_id
        if self.visualization:
            cam_kwargs = dict(
                cameraDistance=0.6,
                cameraYaw=0,
                cameraPitch=-40,
                cameraTargetPosition=[0, 0, 0],
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_GUI, 0, physicsClientId=self._pybullet_client_id
            )
            p.resetDebugVisualizerCamera(
                **cam_kwargs, physicsClientId=self._pybullet_client_id
            )
        # use mass of real cube
        p.changeDynamics(
            bodyUniqueId=self.platform.cube.block,
            linkIndex=-1,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
            mass=CUBE_MASS,
        )
        # p.setTimeStep(0.001)
        # visualize the goal
        if self.visualization:
            self.goal_marker = CuboidMarker(
                size=CUBOID_SIZE,
                position=self.goal["position"],
                orientation=self.goal["orientation"],
                pybullet_client_id=self.platform.simfinger._pybullet_client_id,
            )

    def seed(self, seed=None):
        """Sets the seed for this env’s random number generator.

        .. note::

           Spaces need to be seeded separately.  E.g. if you want to sample
           actions directly from the action space using
           ``env.action_space.sample()`` you can set a seed there using
           ``env.action_space.seed()``.

        Returns:
            List of seeds used by this environment.  This environment only uses
            a single seed, so the list contains only one element.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t, action):
        if self.simulation:
            robot_observation = self.platform.get_robot_observation(t)
            camera_observation = self.platform.get_camera_observation(t)
            obj_obs = camera_observation.object_pose
        else:
            robot_observation = self.real_platform.get_robot_observation(t)
            camera_observation = self.real_platform.get_camera_observation(t)
            obj_obs = camera_observation.filtered_object_pose
            if np.allclose(obj_obs.orientation, 0.0):
                obj_obs = camera_observation.object_pose

        tip_positions = np.array(
            self.pinocchio_utils.forward_kinematics(robot_observation.position)
        )
        if self.tip_positions_object_frame:
            tip_positions = c_utils.get_of_from_wf(tip_positions, obj_obs)
        obs = {
            "position": robot_observation.position,
            "velocity": robot_observation.velocity,
            "torque": robot_observation.torque,
            "tip_positions": tip_positions,
            "tip_force": robot_observation.tip_force,
            "action": action,
        }
        if self.return_timestamp:
            obs["cam0_timestamp"] = camera_observation.cameras[0].timestamp
        observation = {
            "observation": obs,
            "desired_goal": self.goal,
            "achieved_goal": {
                "position": obj_obs.position,
                "orientation": obj_obs.orientation,
            },
        }
        return observation

    def _set_sim_state(self, obs):
        # set cube position & orientation
        self.platform.cube.set_state(
            obs["achieved_goal"]["position"], obs["achieved_goal"]["orientation"]
        )
        # set robot position & velocity
        self.platform.simfinger.reset_finger_positions_and_velocities(
            obs["robot"]["position"], obs["robot"]["velocity"]
        )

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.simfinger.Action(gym_action, None, Action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.simfinger.Action(None, gym_action, Action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.simfinger.Action(
                torque=gym_action["torque"],
                position=gym_action["position"],
                action_cls=Action,
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def register_custom_log(self, name, data):
        if name in self.custom_logs:
            self.custom_logs[name].append({"step_count": self.step_count, "data": data})
        else:
            self.custom_logs[name] = [{"step_count": self.step_count, "data": data}]

    def save_custom_logs(self):
        if self.path is None:
            return

        gymlogger.debug("saving custom logs...")
        custom_logdir = self.path
        if not os.path.isdir(custom_logdir):
            gymlogger.debug(
                "{} does not exist. skip saving custom logs.".format(custom_logdir)
            )
            return
        else:
            custom_logdir = self.path
        path = os.path.join(custom_logdir, "custom_data")
        with shelve.open(path, writeback=True) as f:
            for key, val in self.custom_logs.items():
                f[key] = val

        # save the rewards
        path = os.path.join(custom_logdir, "reward.pkl")
        with open(path, "wb") as handle:
            pkl.dump(self.reward_list, handle)

        # if ran in simulation save the observation
        if self.simulation:
            path = os.path.join(custom_logdir, "observations.pkl")
            with open(path, "wb") as handle:
                pkl.dump(self.observation_list, handle)

        # store the goal to a file, i.e. the last goal,...
        import json

        self.set_goal()
        self.set_reach_finish()
        goal_file = os.path.join(custom_logdir, "goal.json")
        goal_info = {
            "difficulty": self.difficulty,
            "goal": json.loads(
                json.dumps(
                    {
                        "position": self.goal["position"].tolist(),
                        "orientation": self.goal["orientation"].tolist(),
                    }
                )
            ),
            "changegoal": self.change_goal_last,
            "reachstart": self.reach_start_point,
            "reachfinish": self.reach_finish_point,
            "align_obj_error": self.align_obj_error,
            "init_align_obj_error": self.init_align_obj_error,
            "init_obj_pose": self.init_obj_pose,
        }
        with open(goal_file, "w") as fh:
            json.dump(goal_info, fh, indent=4)

    def set_goal(self, pos=None, orientation=None, log_timestep=True):
        ex_state = move_cube.sample_goal(difficulty=-1)  # ensures that on the ground
        if not (pos is None):
            self.goal["position"] = ex_state.position
            self.goal["position"][:3] = pos[:3]
        if not (orientation is None):
            self.goal["orientation"] = ex_state.orientation
            self.goal["orientation"] = orientation

        if self.visualization:
            self.cube_viz.goal_viz = None

        if log_timestep:
            if self.simulation:
                self.change_goal_last = self.platform.get_current_timeindex()
            else:
                self.change_goal_last = self.real_platform.get_current_timeindex()

    def _get_obs_with_timeidx(self):
        from .reward_fns import _orientation_error

        # simply set the reach finish timestep
        if self.simulation:
            timeidx = self.platform.get_current_timeindex()
        else:
            timeidx = self.real_platform.get_current_timeindex()
        obs = self._create_observation(timeidx, action=None)
        return obs, timeidx

    def set_reach_finish(self):
        obs, timeidx = self._get_obs_with_timeidx()
        self.align_obj_error = _orientation_error(obs)
        self.reach_finish_point = timeidx

    def set_reach_start(self):
        import json

        from .reward_fns import _orientation_error

        obs, timeidx = self._get_obs_with_timeidx()
        self.init_align_obj_error = _orientation_error(obs)
        self.init_obj_pose = json.loads(
            json.dumps(
                {
                    "position": obs["achieved_goal"]["position"].tolist(),
                    "orientation": obs["achieved_goal"]["orientation"].tolist(),
                }
            )
        )
        self.reach_start_point = timeidx


class RobotWrenchCubeEnv(RobotCubeEnv):
    """Real Robot env with wrench optimization"""

    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        frameskip: int = 1,
        time_step_s: float = 0.004,
        gravity: float = -9.81,
        visualization: bool = False,
        reward_fn: callable = competition_reward,
        termination_fn: callable = None,
        initializer: callable = None,
        episode_length: int = move_cube.episode_length,
        path: str = None,
        object_frame: bool = False,
        cone_approx: bool = False,
        force_factor: float = 0.5,
        torque_factor: float = 0.25,
        integral_control_freq: float = 10,
        ki: float = 0.1,
        use_traj_opt: bool = True,
        use_actual_cp: bool = False,
        use_benchmark_controller: bool = False,
        debug: bool = False,
    ):
        """Initialize.

        Args:
            cube_goal_pose (dict): Goal pose for the cube.  Dictionary with
                keys "position" and "orientation".
            goal_difficulty (int): Difficulty level of the goal (needed for
                reward computation).
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
        """
        # TODO: remove hardcoding of sim=True in super().__init__()
        super(RobotWrenchCubeEnv, self).__init__(
            cube_goal_pose,
            goal_difficulty,
            ActionType.TORQUE,
            frameskip,
            time_step_s,
            True,
            gravity,
            visualization,
            reward_fn,
            termination_fn,
            initializer,
            episode_length,
            path=path,
            debug=debug,
            use_benchmark_controller=use_benchmark_controller,
        )
        self.time_step_s = time_step_s
        # TODO: remove hard-coded force_factor, meant to only translate cube
        self.force_factor = np.array([force_factor, force_factor, force_factor])
        self.torque_factor = torque_factor

        self.object_frame = object_frame
        self.cone_approx = cone_approx
        self.use_traj_opt = use_traj_opt
        self.integral_control_freq = integral_control_freq
        self.ki = ki
        self.use_actual_cp = use_actual_cp
        self.cp_force_lines = None

        # Create the action and observation spaces
        # ========================================

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )

        # verify that the given goal pose is contained in the cube state space
        if not object_state_space.contains(self.goal):
            raise ValueError("Invalid goal pose.")

        self.robot_torque_space = robot_torque_space
        self.wrench_space = gym.spaces.Box(low=-np.ones(6), high=np.ones(6))
        self.action_space = self.wrench_space
        self.initial_action = np.zeros(9)
        self.cp_params = None
        self.contact_viz = None
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                        "tip_positions": gym.spaces.Box(
                            low=np.array([trifingerpro_limits.object_position.low] * 3),
                            high=np.array(
                                [trifingerpro_limits.object_position.high] * 3
                            ),
                        ),
                        "tip_force": gym.spaces.Box(low=np.zeros(3), high=np.ones(3)),
                        "action": robot_torque_space,
                    }
                ),
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

        self.pinocchio_utils = PinocchioUtils()
        self.prev_observation = None
        self._prev_step_report = 0
        self.setup_contact_force_opt()

    def visualize_markers(self):
        position = self.prev_observation["achieved_goal"]["position"]
        orientation = self.prev_observation["achieved_goal"]["orientation"]
        cube_ftip_pos_wf = c_utils.get_cp_pos_wf_from_cp_params(
            self.cp_params, position, orientation
        )
        if self.visualization:
            if self.contact_viz is None:
                self.contact_viz = VisualMarkers()
                self.contact_viz.add(cube_ftip_pos_wf)
            else:
                for i, marker in enumerate(self.contact_viz.markers):
                    pos = cube_ftip_pos_wf[i]
                    marker.set_state(pos)

    def visualize_forces(self, cp_pos_list, ft_force_list):
        if not self.cp_force_lines or len(self.cp_force_lines) != len(cp_pos_list):
            self.cp_force_lines = [
                p.addUserDebugLine(
                    pos,
                    pos + 0.5 * np.linalg.norm(l_len) * (l_len),
                    [1, 0, 0],
                    physicsClientId=self.platform.simfinger._pybullet_client_id,
                )
                for pos, l_len in zip(cp_pos_list, ft_force_list)
            ]
        else:
            self.cp_force_lines = [
                p.addUserDebugLine(
                    pos,
                    pos + 0.5 * np.linalg.norm(l_len) * (l_len),
                    [1, 0, 0],
                    replaceItemUniqueId=l_id,
                    physicsClientId=self.platform.simfinger._pybullet_client_id,
                )
                for pos, l_len, l_id in zip(
                    cp_pos_list, ft_force_list, self.cp_force_lines
                )
            ]

    def get_contact_points(self):
        finger_contacts = [
            p.getContactPoints(
                bodyA=self.platform.simfinger.finger_id,
                linkIndexA=tip,
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )
            for tip in self.platform.simfinger.pybullet_tip_link_indices
        ]
        finger_contact_states = []
        for x in finger_contacts:
            if x and len(x[0]) == 14:
                finger_contact_states.append(ContactResult(*x[0]))
            else:
                finger_contact_states.append(None)
        return finger_contact_states

    def update_contact_state(self, des_tip_forces):
        finger_contact_states = self.get_contact_points()
        obs_tip_forces = []
        obs_tip_dir = []
        obs_tip_pos = []
        for contact in finger_contact_states:
            if contact is not None:
                contact_dir = np.array(
                    [
                        contact.lateralFrictionDir1,
                        contact.lateralFrictionDir2,
                    ]
                )
                contact_pos = np.array([contact.positionOnA, contact.positionOnB])
                contact_force = np.array(
                    [
                        contact.normalForce,
                        contact.lateralFriction1,
                        contact.lateralFriction2,
                    ]
                )
                obs_tip_forces.append(contact_force)
                obs_tip_dir.append(contact_dir)
                obs_tip_pos.append(contact_pos)
            else:
                obs_tip_forces.append(np.zeros((3, 3)))
                obs_tip_dir.append(None)
                obs_tip_pos.append(np.zeros((3, 3)))

        self._current_tip_force, self._current_contact_ori = (
            obs_tip_forces,
            rotations,
        ) = self.rotate_obs_force(obs_tip_forces, obs_tip_dir)
        self._current_contact_pos = [x[1] for x in obs_tip_pos]

        if self.step_count % 50 == 0:
            gymlogger.debug(
                f"del_tip_forces: {np.abs(des_tip_forces - obs_tip_forces)}"
            )

        self.register_custom_log("des_tip_forces", des_tip_forces)
        self.register_custom_log("obs_tip_forces", obs_tip_forces)
        self.register_custom_log("obs_tip_dir", obs_tip_dir)
        self.register_custom_log("obs_tip_pos", obs_tip_pos)

    def rotate_obs_force(self, obs_force, tip_dir):
        forces = []
        tip_rotations = []
        for f, td in zip(obs_force, tip_dir):
            if td is None:
                forces.append(np.zeros(3))
                tip_rotations.append(None)
            else:
                v, w = td
                u = np.cross(v, w)
                R = np.vstack([u, v, w]).T
                R = Rotation.from_matrix(R).as_matrix()
                tip_rotations.append(R)
                forces.append(-R @ f)
        return forces, tip_rotations

    def reset(self):
        self._reset_direct_simulation()
        # self.custom_pinocchio_utils = CustomPinocchioUtils(
        #    self.platform.simfinger.finger_urdf_path,
        #    self.platform.simfinger.tip_link_names,
        # )
        self.maybe_update_tip_dist()
        self.step_count = 0
        self._current_tip_force = None
        self._current_contact_pos = self._current_contact_ori = None
        self._integral = 0
        self.action_space = self.robot_torque_space
        self.prev_observation, _, _, _ = super().step(self.initial_action)
        self.action_space = self.wrench_space
        self.execute_grasp()
        return deepcopy(self.prev_observation)

    def action(self, des_wrench, step=0):
        if step == 0:
            # reset integral everytime set_point changes
            self._integral = 0
            self._des_tip_forces = tip_forces_wf = self.compute_tip_forces(des_wrench)

        if self._current_tip_force is None:
            # TODO: Try using des_wrench instead of zeros here, initializing
            # error term to 0
            tip_forces_wf = self.compute_tip_forces(des_wrench)
        else:
            tip_forces_wf = self._current_tip_force
        error = np.asarray(self._des_tip_forces) - np.asarray(tip_forces_wf)
        integral_weight = np.where(np.abs(error) > 0.1, 0.9, 0.5)
        self._integral = error * self.ki + integral_weight * self._integral
        tip_forces_wf = np.clip(self._integral + self._des_tip_forces, -4, 4)
        torque = self.compute_joint_torques(des_wrench, tip_forces_wf)
        torque = np.clip(
            torque, self.robot_torque_space.low, self.robot_torque_space.high
        )
        self.register_custom_log("des_torque", torque)
        self.register_custom_log(
            "q_current", self.prev_observation["observation"]["position"]
        )
        return torque, tip_forces_wf

    def compute_tip_forces(self, wrench):
        obj_pose = np.concatenate(
            [
                self.prev_observation["achieved_goal"]["position"],
                self.prev_observation["achieved_goal"]["orientation"],
            ]
        )
        tip_forces_of = self.get_balance_contact_forces(wrench, obj_pose)
        if tip_forces_of is not None:
            tip_forces_of = tip_forces_of.reshape((3, 3))
        else:
            return None
        tip_forces_wf = []
        obj_pose = move_cube.Pose.from_dict(self.prev_observation["achieved_goal"])
        cp_list = self.get_cp_of_list(self.cp_params, obj_pose, self.use_actual_cp)

        for i, l_cf in enumerate(tip_forces_of):
            cp_of = cp_list[i]
            R_cp_2_o = Rotation.from_quat(cp_of[1])
            R_o_2_w = Rotation.from_quat(
                self.prev_observation["achieved_goal"]["orientation"]
            )
            l_o = R_cp_2_o.apply(l_cf)
            l_wf = R_o_2_w.apply(l_o)
            tip_forces_wf.append(l_wf)
        return tip_forces_wf

    def compute_joint_torques(self, des_wrench, tip_forces_wf):
        if self.main_ctrl is not None:
            x_goal = self.forward_sim_path(des_wrench)
            q_curr = self.prev_observation["observation"]["position"]
            current_ft_pos = self.pinocchio_utils.forward_kinematics(q_curr)
            x_goal = x_goal.reshape((3, 3))
            ft_pos_goal = [ft_des - ft for ft_des, ft in zip(x_goal, current_ft_pos)]
            tip_forces = tip_forces_wf.reshape((3, 3))
            dir1 = tip_forces[0]
            dir1 = (dir1) / np.sqrt(np.sum((dir1) ** 2))
            dir2 = tip_forces[1]
            dir2 = (dir2) / np.sqrt(np.sum((dir2) ** 2))
            dir3 = tip_forces[2]
            dir3 = (dir3) / np.sqrt(np.sum((dir3) ** 2))
            torque = self.main_ctrl.kinematics.imp_ctrl_3_fingers(
                q_curr,
                [dir1, dir2, dir3],
                [4, 8, 12],
                [np.zeros((3))] * 3,
                [
                    ft_pos - ft_curr
                    for ft_pos, ft_curr in zip(ft_pos_goal, current_ft_pos)
                ],
                np.zeros(9),
                [250] * 3,
                [250] * 3,
                [0, 0, 0],
            )
            # if trying to lift
            # if des_wrench[2] > 0:
            #     torque += self.main_ctrl.kinematics.add_additional_force_3_fingers(
            #         q_curr,
            #         self.force_factor,
            #         self.grasp.get_edge_directions(robot_state, state),
            #         [0, 0, 0],
            #         correct_torque=True,
            #     )
        else:
            q_curr = self.prev_observation["observation"]["position"]
            current_ft_pos = self.pinocchio_utils.forward_kinematics(q_curr)
            # x_goal, dx_goal = self.forward_sim_path(des_wrench)
            # x_goal = x_goal.reshape((3, 3))
            delta_x = [np.zeros(3) for _ in zip(current_ft_pos, current_ft_pos)]
            torque = 0
            for fid in range(3):
                torque += self.compute_torque_single_finger(
                    tip_forces_wf[fid], fid, delta_x[fid]
                )
        return torque

    def compute_torque_single_finger(self, tip_force_wf, finger_id, delta_x):
        Kp = [20, 20, 40, 20, 20, 40, 20, 20, 40]

        Kp_x = Kp[finger_id * 3 + 0]
        Kp_y = Kp[finger_id * 3 + 1]
        Kp_z = Kp[finger_id * 3 + 2]
        Kp = np.diag([Kp_x, Kp_y, Kp_z])
        # Kv = [0.7] * 9
        q_current = self.prev_observation["observation"]["position"]
        # dq_current = self.prev_observation["observation"]["velocity"]
        delta_x = np.array(delta_x)
        # delta_dx = np.expand_dims(np.array(tip_vel_desired), 1) - np.array(dx_current)

        Ji = self.pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
        # Just take first 3 rows, which correspond to linear velocities of fingertip
        Ji = Ji[:3, :]
        # Get g matrix for gravity compensation
        _, g = self.pinocchio_utils.get_lambda_and_g_matrix(
            finger_id, q_current, Ji, self._gravity
        )
        # gymlogger.debug("Gravity: {}".format(g))
        torque = np.squeeze(Ji.T @ (Kp @ delta_x) + 0.5 * Ji.T @ tip_force_wf) + g
        return torque

    def get_cp_of_list(self, cp_params, obj_pose, use_actual_cp=True):
        cp_list = []
        if use_actual_cp and self._current_contact_pos is not None:
            R_w_2_of = Rotation.from_quat(obj_pose.orientation).inv().as_matrix()
            flip_z = Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
            for pos, ori in zip(self._current_contact_pos, self._current_contact_ori):
                if ori is None:
                    cp_list.append(None)
                else:
                    cp_of = c_utils.get_of_from_wf(pos, obj_pose)
                    cp_quat = Rotation.from_matrix(R_w_2_of @ flip_z @ ori).as_quat()
                    cp_list.append((cp_of, cp_quat))
        else:
            for cp_param in cp_params:
                if cp_param is not None:
                    cp_of = c_utils.get_cp_of_from_cp_param(cp_param)
                    cp_list.append((cp_of.pos_of, cp_of.quat_of))
        if use_actual_cp and self.debug and self.step_count % 50 == 0:
            cp_pos = np.array([cp.pos_of for cp in cp_list])
            cp_ori = np.array([cp.quat_of for cp in cp_list])
            gymlogger.debug(f"Obs CP List: pos ({cp_pos}),\nori ({cp_ori})")
            des_cp_list = self.get_cp_of_list(cp_params, use_actual_cp=False)
            des_cp_pos = np.array([cp.pos_of for cp in des_cp_list])
            des_cp_ori = np.array([cp.quat_of for cp in des_cp_list])
            gymlogger.debug(f"Des CP List: pos ({des_cp_pos}),\nori ({des_cp_ori})")
        return cp_list

    def get_cp_wf_list(self, cp_params, obj_pose, use_actual_cp=True):
        cp_of_list = self.get_cp_of_list(cp_params, obj_pose, use_actual_cp)
        R_o_2_wf = Rotation.from_quat(obj_pose.orientation)
        cp_list = []
        for cp_of in cp_of_list:
            if cp_of is not None:
                c_pos, c_ori = cp_of
                c_pos_wf = c_utils.get_wf_from_of(c_pos, obj_pose)
                # c_pos_wf = R_o_2_wf.as_matrix() @ c_pos
                c_ori_wf = R_o_2_wf * Rotation.from_quat(c_ori)
                c_ori_wf = c_ori_wf.as_quat()
                cp_list.append((c_pos_wf, c_ori_wf))
            else:
                cp_list.append(None)
        return cp_list

    def setup_contact_force_opt(self):
        self.fop = fop.ForceOptProblem(
            obj_mu=1.0,
            mass=0.016,
            gravity=self._gravity,
            target_n=0.1,
            cone_approx=self.cone_approx,
        )
        return

    def get_balance_contact_forces(self, des_wrench, obj_pose):
        cp_list = self.get_cp_wf_list(
            self.cp_params,
            move_cube.Pose(obj_pose[:3], obj_pose[3:]),
            self.use_actual_cp,
        )
        cp_list = np.array([np.concatenate(cpwf) for cpwf in cp_list if cpwf])
        balance_force_t = self.fop(
            torch.from_numpy(des_wrench.astype("float32")).to(self.fop.device),
            # obj_pose,
            cp_list,
        )
        # Check if solution was feasible
        if balance_force_t is None:
            gymlogger.debug(
                "Did not solve contact force opt for "
                f"desired wrench: {des_wrench}, contact points: {self.cp_params}"
            )
        else:
            balance_force = balance_force_t.detach().cpu().numpy()
            balance_force = np.squeeze(balance_force)
        f = self.fop.balance_force_test(
            des_wrench, balance_force, cp_list
        )  # , obj_pose)
        if self.step_count % 50 == 0:
            # gymlogger.debug(f"Ext wrench: {w_ext} = {des_wrench} + {weight}")
            gymlogger.debug(f"Balance force test: {f}. stepct: {self.step_count}")
        return balance_force

    def step(self, action):
        action[:3] *= self.force_factor
        action[3:] *= self.torque_factor
        self.action_space = self.robot_torque_space
        for step in range(self.integral_control_freq):
            torque, des_tip_forces = self.action(action, step)
            obs, rew, done, info = super(RobotWrenchCubeEnv, self).step(torque)
            self.update_contact_state(des_tip_forces)
            if self.debug and self.visualization:
                self.visualize_forces(
                    self._current_contact_pos, self._current_tip_force
                )
            if not done and len(self._current_tip_force) != 3:
                # need to regrasp
                gymlogger.debug("~~~~~~Executing Re-grasp~~~~~~")
                # if done is True, quits re-grasp attempt and returns end of episode
                done = self.execute_grasp(pre_grasp_ngrid=6, grasp_ngrid=12)
                self.update_contact_state(des_tip_forces)
            if len(self._current_tip_force) != 3:
                # TODO: remove hard-coded drop penalty
                rew = -100
                done = True
            if done:
                break
        self.action_space = self.wrench_space
        return obs, rew, done, info

    def execute_simple_traj(self, t, q, dq):
        KP = [200, 200, 400, 200, 200, 400, 200, 200, 400]
        KV = [0.7, 0.7, 0.8, 0.7, 0.7, 0.8, 0.7, 0.7, 0.8]
        obj_pose = move_cube.Pose.from_dict(self.prev_observation["achieved_goal"])
        self.cp_params = c_utils.get_lifting_cp_params(obj_pose)

        # Get joint positions, velocities
        # current_position = self.prev_observation["robot"]["position"]
        # current_velocity = self.prev_observation["robot"]["velocity"]
        current_ft_pos = self.pinocchio_utils.forward_kinematics(q)
        ft_goal = current_ft_pos + np.array(
            [0, 0, np.sin(t) * 0.02, 0, 0, np.sin(t) * 0.02, 0, 0, np.sin(t) * 0.02]
        ).reshape((3, 3))
        ft_vel = np.array(
            [0, 0, np.cos(t) * 0.02, 0, 0, np.cos(t) * 0.02, 0, 0, np.cos(t) * 0.02]
        ).reshape((3, 3))
        self.visualize_markers()
        torque = c_utils.impedance_controller(
            ft_goal,
            ft_vel,
            q,
            dq,
            self.pinocchio_utils,
            tip_forces_wf=None,
            Kp=KP,
            Kv=KV,
            grav=self._gravity,
        )
        torque = np.clip(torque, self.action_space.low, self.action_space.high)
        self.prev_observation, _, _, _ = super(RobotWrenchCubeEnv, self).step(torque)
        return

    def execute_grasp(
        self,
        pre_grasp_ngrid=6,
        grasp_ngrid=40,
        interp_n=10,
        tol=0.005,
    ):
        if self.use_traj_opt:
            done = self.execute_pre_grasp(pre_grasp_ngrid)
            if done:
                return done

        FT_RADIUS = 0.0075
        # get aligned cube pose in case of noisy observation
        obj_pose = move_cube.Pose.from_dict(self.prev_observation["achieved_goal"])
        self.cp_params = c_utils.get_lifting_cp_params(obj_pose)
        # obj_pose = c_utils.get_aligned_pose(obj_pose)

        # Get joint positions
        current_position = self.prev_observation["observation"]["position"]

        # Get current fingertip positions
        current_ft_pos = self.pinocchio_utils.forward_kinematics(current_position)

        # Get list of desired fingertip positions
        cp_wf_list = c_utils.get_cp_pos_wf_from_cp_params(
            self.cp_params, obj_pose.position, obj_pose.orientation
        )

        R_list = c_utils.get_ft_R(current_position)

        # Deal with None fingertip_goal here
        # If cp_wf is None, set ft_goal to be  current ft position
        for i in range(len(cp_wf_list)):
            if cp_wf_list[i] is None:
                cp_wf_list[i] = current_ft_pos[i]
            else:
                # Transform cp to ft center
                R = R_list[i]
                ftip_radius_pos_offset = R @ np.array([0, 0, FT_RADIUS])
                new_pos = np.array(cp_wf_list[i]) + ftip_radius_pos_offset[:3]
                cp_wf_list[i] = new_pos

        ft_goal = np.asarray(cp_wf_list).flatten()
        # ft_goal += np.array([0, 0, 0.02] * 3)
        if self.use_traj_opt:
            dt = 0.01  # self.time_step_s * self.frameskip
            finger_nlp = c_utils.define_static_object_opt(grasp_ngrid, dt)
            ft_pos_traj, ft_vel_traj = self.run_finger_traj_opt(
                current_position, obj_pose, ft_goal, finger_nlp
            )

            gymlogger.debug("~~~~~~Executing Grasp~~~~~~")
            gymlogger.debug(f"current_position {current_position}")
            gymlogger.debug(f"ft_goal {ft_goal}")
            gymlogger.debug(f"ft_pos_traj shape {ft_pos_traj.shape}")

            self.visualize_markers()
            self.action_space = self.robot_torque_space
            done = self.execute_traj(ft_pos_traj, ft_vel_traj)
            self.action_space = self.wrench_space
        else:
            # using PinocchioUtils.inverse_kinematics, set to position control
            self.action_type = ActionType.POSITION
            self.action_space = self.observation_space["observation"]["position"]
            # TODO: tol currently hardcoded to 0.001, make it smaller and a variable
            ft_curr = np.concatenate(current_ft_pos)
            goals = np.linspace(np.asarray(ft_curr), np.asarray(ft_goal), interp_n)

            gymlogger.debug("~~~~~~Executing Grasp~~~~~~")
            gymlogger.debug(f"current_position {current_position}")
            gymlogger.debug(f"ft_goals {goals}")

            i = 0
            g = goals[i]
            err = np.abs(g - np.concatenate(current_ft_pos))
            done = False
            while not done and i < len(goals) and self.step_count < 1000:
                q = self.pinocchio_utils.inverse_kinematics_three_fingers(
                    g, current_position
                )
                self.last_prev_observation = self.prev_observation
                self.prev_observation, _, done, _ = super(
                    RobotWrenchCubeEnv, self
                ).step(q)
                current_position = self.prev_observation["observation"]["position"]
                current_ft_pos = self.pinocchio_utils.forward_kinematics(
                    current_position
                )
                g = goals[i]
                err = np.abs(g - np.concatenate(current_ft_pos))
                if self.step_count % 50 == 0:
                    gymlogger.debug(f"error to goal: {err}")
                # only increment goal if errors are less than tolerance
                i += np.all(err < tol)

            self.action_type = ActionType.TORQUE
            self.action_space = self.wrench_space
        return done

    def execute_pre_grasp(self, ngrid=6):
        # Get object positions
        obj_pose = move_cube.Pose.from_dict(self.prev_observation["achieved_goal"])
        self.cp_params = c_utils.get_lifting_cp_params(obj_pose)
        # obj_pose = c_utils.get_aligned_pose(obj_pose)

        # Get joint positions
        current_position = self.prev_observation["observation"]["position"]
        current_ft_pos = self.pinocchio_utils.forward_kinematics(current_position)
        ft_goal = c_utils.get_pre_grasp_ft_goal(
            obj_pose, current_ft_pos, self.cp_params
        )
        gymlogger.debug("~~~~~~Executing Pre-Grasp~~~~~~")
        gymlogger.debug(f"ft_goal:{ft_goal}")

        # if self.use_traj_opt:
        #     dt = 0.01  # self.time_step_s * self.frameskip
        #     release_nlp = c_utils.define_static_object_opt(ngrid, dt)
        #     ft_pos_traj, ft_vel_traj = self.run_finger_traj_opt(
        #         current_position, obj_pose, ft_goal, release_nlp
        #     )
        #     self.visualize_markers()
        #     self.action_space = self.robot_torque_space
        #     done = self.execute_traj(ft_pos_traj, ft_vel_traj)
        #     self.action_space = self.wrench_space
        # else:
        # using PinocchioUtils.inverse_kinematics, set to position control
        self.action_type = ActionType.POSITION
        self.action_space = self.observation_space["observation"]["position"]

        # TODO: tol currently hardcoded to 0.01, make it smaller and a variable
        err = np.abs(ft_goal - np.concatenate(current_ft_pos))
        tol = 0.005
        start = self.step_count
        done = False

        while not done:
            q = self.pinocchio_utils.inverse_kinematics_three_fingers(
                ft_goal, current_position
            )
            self.last_prev_observation = self.prev_observation
            self.prev_observation, _, done, _ = super(RobotWrenchCubeEnv, self).step(q)
            done = done and not np.all(err < tol) and self.step_count < start + 500
            current_position = self.prev_observation["observation"]["position"]
            current_ft_pos = self.pinocchio_utils.forward_kinematics(current_position)
            err = np.abs(ft_goal - np.concatenate(current_ft_pos))

        self.action_type = ActionType.TORQUE
        self.action_space = self.wrench_space
        return done

    def execute_traj(self, ft_pos_traj, ft_vel_traj):
        KP = [200, 200, 400, 200, 200, 400, 200, 200, 400]
        KV = [0.7, 0.7, 0.8, 0.7, 0.7, 0.8, 0.7, 0.7, 0.8]
        for waypoint in range(ft_pos_traj.shape[0]):
            ft_pos_goal_list, ft_vel_goal_list = [], []
            for f_i in range(3):
                new_pos = ft_pos_traj[waypoint, f_i * 3 : f_i * 3 + 3]
                new_vel = ft_vel_traj[waypoint, f_i * 3 : f_i * 3 + 3]

                ft_pos_goal_list.append(new_pos)
                ft_vel_goal_list.append(new_vel)
            current_position = self.prev_observation["observation"]["position"]
            current_velocity = self.prev_observation["observation"]["velocity"]
            current_ft_pos = np.array(
                self.pinocchio_utils.forward_kinematics(current_position)
            ).reshape((3, 3))
            if self.main_ctrl is not None:
                torque = self.main_ctrl.kinematics.imp_ctrl_3_fingers(
                    [current_position.reshape((3, 3))] * 3,
                    [[0.0] * 3] * 3,
                    [4, 8, 12],
                    [np.zeros((3))] * 3,
                    [
                        ft_pos - ft_curr
                        for ft_pos, ft_curr in zip(ft_pos_goal_list, current_ft_pos)
                    ],
                    [np.zeros(3)] * 3,
                    [250] * 3,
                    [250] * 3,
                    [0, 0, 0],
                )
            else:
                torque = c_utils.impedance_controller(
                    ft_pos_goal_list,
                    ft_vel_goal_list,
                    current_position,
                    current_velocity,
                    self.pinocchio_utils,
                    tip_forces_wf=None,
                    Kp=KP,
                    Kv=KV,
                    grav=self._gravity,
                )
            torque = np.clip(torque, self.action_space.low, self.action_space.high)
            if self.step_count % 50 == 0:
                gymlogger.debug(f"ft_pos_goal_list: {ft_pos_goal_list}")
                gymlogger.debug(f"torque: {torque}")
            self.last_prev_observation = self.prev_observation
            self.prev_observation, _, done, _ = super(RobotWrenchCubeEnv, self).step(
                torque
            )
            self.visualize_markers()
            if done:
                return done
        return done

    def forward_sim_path(self, des_wrench, dt=0.04):
        q = self.prev_observation["observation"]["position"]
        dq = self.prev_observation["observation"]["velocity"]
        v0 = []
        for fid in range(3):
            Ji = self.pinocchio_utils.get_tip_link_jacobian(fid, q)
            # Just take first 3 rows, which correspond to linear velocities of fingertip
            Ji = Ji[:3, :]
            v0.append(Ji @ np.expand_dims(np.array(dq), 1))
        v0 = np.concatenate(v0).squeeze()
        x_ddot = np.repeat(des_wrench[:3] / self.fop.mass, 3, axis=0).squeeze()
        x0 = np.concatenate(self.pinocchio_utils.forward_kinematics(q))
        x = dt * 0.5 * (x_ddot * dt) + v0 * dt + x0
        dx = x_ddot * dt + v0
        return x, dx

    def run_finger_traj_opt(
        self, current_position, obj_pose, ft_goal, nlp, interp_n=26
    ):
        """
        Run trajectory optimization for fingers, given fingertip goal positions
        ft_goal: (9,) array of fingertip x,y,z goal positions in world frame
        interp_n: Number of interpolation points
        """
        nGrid = nlp.nGrid
        self.traj_waypoint_counter = 0

        ft_pos, ft_vel = c_utils.get_finger_waypoints(
            nlp, ft_goal, current_position, obj_pose
        )

        # Linearly interpolate between each waypoint (row)
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid - 1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        ft_pos_traj = itp_pos(row_coord_out)

        # Zero-order hold for velocity waypoints
        ft_vel_traj = np.repeat(ft_vel, repeats=interp_n + 1, axis=0)[:-interp_n, :]

        return ft_pos_traj, ft_vel_traj


class RobotContactCubeEnv(RobotCubeEnv):
    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        action_type: ActionType = ActionType.TORQUE,
        frameskip: int = 1,
        time_step_s: float = 0.004,
        sim: bool = True,
        gravity: float = -9.81,
        visualization: bool = False,
        reward_fn: callable = competition_reward,
        termination_fn: callable = None,
        initializer: callable = None,
        episode_length: int = move_cube.episode_length,
        path: str = None,
        debug: bool = False,
        action_scale: Union[float, np.ndarray] = None,
        contact_timeout: int = 50,
        min_tip_dist: float = 0.11,
        object_mass: float = 0.016,
        use_benchmark_controller: bool = False,
        return_timestamp: bool = False,
    ):
        """Initialize.

        Args:
            cube_goal_pose (dict): Goal pose for the cube.  Dictionary with
                keys "position" and "orientation".
            goal_difficulty (int): Difficulty level of the goal (needed for
                reward computation).
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            time_step_s (float): length of one timestep in simulator
            sim (bool): whether or not real robot trifinger platform is used
            gravity (float): gravity constant used in sim
            visualization (bool): whether or not to visualize simulator
            reward_fn (callable): function to compute reward from observation
            termination_fn (callable): function to compute end of episode from observation
            initializer (callable): function to sample goal and initial positions
            episode_length (int): length of an episode
            path (str): path to save logs of observations, rewards, and custom metrics
            debug (bool): print debug lines
            action_scale (Union[float, np.ndarray]): constant factor to scale actions
            contact_timeout (int): steps until episode is terminated early
            min_tip_dist (float): minimum tip distance to reach to aviod timeout
            object_mass (float): set object mass to custom value
            use_benchmark_controller (bool): use benchmark-rrc cartesian impedance controller
            return_timestamp (bool): returns cam0_timestamp with observation
        """
        super(RobotContactCubeEnv, self).__init__(
            cube_goal_pose,
            goal_difficulty,
            action_type,
            frameskip,
            time_step_s,
            sim,
            gravity,
            visualization,
            reward_fn,
            termination_fn,
            initializer,
            episode_length,
            path,
            debug,
            action_scale,
            contact_timeout,
            min_tip_dist,
            object_mass,
            use_benchmark_controller,
            return_timestamp,
        )
        self.action_space = self.ftip_force_space = gym.spaces.Box(
            low=-np.ones(9) * 2, high=np.ones(9) * 2
        )
        self.robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )

    def visualize_forces(self, cp_pos_list, ft_force_list):
        if not self.cp_force_lines:
            self.cp_force_lines = [
                self._p.addUserDebugLine(
                    pos, pos + np.linalg.norm(l_len) * (l_len), [1, 0, 0]
                )
                for pos, l_len in zip(cp_pos_list, ft_force_list)
            ]
        else:
            self.cp_force_lines = [
                self._p.addUserDebugLine(
                    pos,
                    pos + np.linalg.norm(l_len) * (l_len),
                    [1, 0, 0],
                    replaceItemUniqueId=l_id,
                )
                for pos, l_len, l_id in zip(
                    cp_pos_list, ft_force_list, self.cp_force_lines
                )
            ]

    def reset(self):
        self.initial_action = np.zeros(9)
        self.prev_observation = None
        return super(RobotContactCubeEnv, self).reset()

    def step(self, action):
        torque = self.action(action)
        self.action_space = self.robot_torque_space
        obs, rew, done, info = super(RobotContactCubeEnv, self).step(torque)
        self.action_space = self.ftip_force_space
        self.update_contact_state(action)
        if self.debug and self.visualization:
            self.visualize_forces(self._current_contact_pos, self._current_tip_force)
        self.prev_observation = obs
        return obs, rew, done, info

    def action(self, tip_forces_wf):
        torque = self.compute_joint_torques(tip_forces_wf)
        torque = np.clip(
            torque, self.robot_torque_space.low, self.robot_torque_space.high
        )
        self.register_custom_log("des_torque", torque)
        if self.prev_observation is not None:
            self.register_custom_log(
                "q_current", self.prev_observation["observation"]["position"]
            )
        return torque

    def compute_joint_torques(self, tip_forces_wf):
        # x_goal, dx_goal = self.forward_sim_path(des_wrench)
        # x_goal = x_goal.reshape((3, 3))
        torque = 0
        for fid in range(3):
            torque += self.compute_torque_single_finger(
                tip_forces_wf[3 * fid : 3 * (fid + 1)], fid
            )
        return torque

    def get_contact_points(self):
        finger_contacts = [
            p.getContactPoints(
                bodyA=self.platform.simfinger.finger_id,
                linkIndexA=tip,
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )
            for tip in self.platform.simfinger.pybullet_tip_link_indices
        ]
        finger_contact_states = []
        for x in finger_contacts:
            if x and len(x[0]) == 14:
                finger_contact_states.append(ContactResult(*x[0]))
            else:
                finger_contact_states.append(None)
        return finger_contact_states

    def update_contact_state(self, des_tip_forces):
        finger_contact_states = self.get_contact_points()
        obs_tip_forces = []
        obs_tip_dir = []
        obs_tip_pos = []
        for contact in finger_contact_states:
            if contact is not None:
                contact_dir = np.array(
                    [
                        contact.lateralFrictionDir1,
                        contact.lateralFrictionDir2,
                    ]
                )
                contact_pos = np.array([contact.positionOnA, contact.positionOnB])
                contact_force = np.array(
                    [
                        contact.normalForce,
                        contact.lateralFriction1,
                        contact.lateralFriction2,
                    ]
                )
                obs_tip_forces.append(contact_force)
                obs_tip_dir.append(contact_dir)
                obs_tip_pos.append(contact_pos)
            else:
                obs_tip_forces.append(np.zeros((3, 3)))
                obs_tip_dir.append(None)
                obs_tip_pos.append(np.zeros((3, 3)))

        self._current_tip_force, self._current_contact_ori = (
            obs_tip_forces,
            rotations,
        ) = self.rotate_obs_force(obs_tip_forces, obs_tip_dir)
        self._current_contact_pos = [x[1] for x in obs_tip_pos]

        des_tip_forces_contact = des_tip_forces.reshape((3, 3))[
            [fid for fid in range(3) if finger_contact_states[fid] is not None]
        ]

        if self.step_count % 50 == 0:
            gymlogger.debug(
                f"del_tip_forces: {np.abs(des_tip_forces_contact - obs_tip_forces)}"
            )

        self.register_custom_log("des_tip_forces", des_tip_forces)
        self.register_custom_log("obs_tip_forces", obs_tip_forces)
        self.register_custom_log("obs_tip_dir", obs_tip_dir)
        self.register_custom_log("obs_tip_pos", obs_tip_pos)

    def rotate_obs_force(self, obs_force, tip_dir):
        forces = []
        tip_rotations = []
        for f, td in zip(obs_force, tip_dir):
            if td is None:
                forces.append(0)
                tip_rotations.append(None)
            else:
                v, w = td
                u = np.cross(v, w)
                R = np.vstack([u, v, w]).T
                R = Rotation.from_matrix(R).as_matrix()
                tip_rotations.append(R)
                forces.append(-R @ f)
        return forces, tip_rotations

    def compute_torque_single_finger(self, tip_force_wf, finger_id):
        Kp = [20, 20, 40, 20, 20, 40, 20, 20, 40]

        Kp_x = Kp[finger_id * 3 + 0]
        Kp_y = Kp[finger_id * 3 + 1]
        Kp_z = Kp[finger_id * 3 + 2]
        Kp = np.diag([Kp_x, Kp_y, Kp_z])
        # Kv = [0.7] * 9
        if self.prev_observation is not None:
            q_current = self.prev_observation["observation"]["position"]
        else:
            q_current = INIT_JOINT_CONF

        Ji = self.pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
        # Just take first 3 rows, which correspond to linear velocities of fingertip
        Ji = Ji[:3, :]
        # Get g matrix for gravity compensation
        _, g = self.pinocchio_utils.get_lambda_and_g_matrix(
            finger_id, q_current, Ji, self._gravity
        )
        # gymlogger.debug("Gravity: {}".format(g))
        torque = np.squeeze(Ji.T @ tip_force_wf) + g
        return torque


def get_theta_z_wf(goal_rot, actual_rot):
    y_axis = [0, 1, 0]

    actual_direction_vector = actual_rot.apply(y_axis)

    goal_direction_vector = goal_rot.apply(y_axis)
    N = np.array([0, 0, 1])  # normal vector of ground plane
    proj = goal_direction_vector - goal_direction_vector.dot(N) * N
    goal_direction_vector = proj / np.linalg.norm(proj)  # normalize projection

    orientation_error = np.arccos(goal_direction_vector.dot(actual_direction_vector))

    return orientation_error
