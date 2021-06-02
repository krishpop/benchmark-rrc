"""Gym environment for the Real Robot Challenge Phase 2."""
import os
import enum
import shelve
import pickle as pkl

import gym
import numpy as np

try:
    import robot_interfaces
    import robot_fingers
except:
    robot_fingers = robot_interfaces = False

import pybullet_data
import trifinger_simulation
from trifinger_simulation import trifingerpro_limits, collision_objects
from trifinger_simulation.tasks import move_cube
from rrc.mp.const import CUSTOM_LOGDIR, INIT_JOINT_CONF, CUBOID_SIZE, CUBOID_MASS
from rrc.mp.const import CUBE_WIDTH, CUBE_HALF_WIDTH, CUBE_MASS
import pybullet as p
import os.path as osp
import inspect

from .reward_fns import competition_reward
from .pinocchio_utils import PinocchioUtils
from .initializers import random_init, training_init, centered_init
from .viz import Viz, CuboidMarker, CubeMarker
import time


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class CubeEnv(gym.GoalEnv):
    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        frameskip: int = 1,
        visualization: bool = False,
        reward_fn: callable = competition_reward,
        termination_fn: callable = None,
        initializer: callable = centered_init,
        episode_length: int = move_cube.episode_length,
        path: str = None,
        save_mp4: bool = False,
        save_dir: str = None
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
        # Basic initialization
        # ====================
        self.path=path

        self._compute_reward = reward_fn
        self._termination_fn = termination_fn
        if initializer is not None and inspect.isclass(initializer):
            self.initializer = initializer(goal_difficulty)
        else:
            self.initializer = initializer
        if cube_goal_pose:
            self.goal = {k: np.array(v) for k, v in cube_goal_pose.items()}
            self.default_goal = {k: np.array(v) for k, v in cube_goal_pose.items()}
        else:
            self.goal = None
            self.default_goal = None
        self.info = {"difficulty": goal_difficulty}
        self.difficulty = goal_difficulty

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # Setup mp4 saving 
        # ================

        self.save_mp4 = save_mp4
        self.save_dir = save_dir
        if save_dir is not None and not osp.exists(save_dir):
            os.makedirs(save_dir)
        self.mp4_logging = False

        if save_mp4: 
            assert osp.exists(save_dir), f"save_dir {save_dir} not initialized or given"

        # will be initialized in reset()
        self.visualization = visualization
        self.episode_length = episode_length
        self.goal_list = [] # needed for multiple goal environments
        self.cube = None
        self.time_step_s = 0.004
        if self.visualization:
            self.cube_viz = Viz()

        # Create the action and observation spaces
        # ========================================

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

        self.action_space = gym.spaces.Box(low=-np.ones(6), high=np.ones(6))
        self.initial_action = np.zeros(6)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "position": object_state_space.spaces['position'],
                        "velocity": gym.spaces.Box(low=-np.ones(6), high=np.ones(6)),
                        "orientation": object_state_space.spaces['orientation']}
                ),
                "action": self.action_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

        self.prev_observation = None
        self._prev_step_report = 0

    def start_mp4_rendering(self):
        # MP4 logging
        if self.save_mp4:
            filepath = lambda i: osp.join(self.save_dir, f"sim-{i}.mp4")
            for i in range(25):
                if not osp.exists(filepath(i)):
                  break
            filepath = filepath(i)
            p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, filepath)
        return

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
        action = np.clip(action, -1, 1)
        if not self.mp4_logging:
            self.start_mp4_rendering()
            self.mp4_logging = True

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > self.episode_length:
            excess = step_count_after - self.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            # send action to robot
            p.applyExternalForce(self.cube.block, -1,
                    action[:3], [0,0,0], p.LINK_FRAME)
            p.applyExternalTorque(self.cube.block, -1,
                    action[3:], p.LINK_FRAME)

            p.stepSimulation(
                physicsClientId=self._pybullet_client_id,
            )
            observation = self._create_observation(action)

            if self.prev_observation is None:
                self.prev_observation = observation
            reward += self._compute_reward(
                self.prev_observation,
                observation,
                self.info
            )
            self.prev_observation = observation

            self.step_count += 1  # t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length - 1:
                break

        is_done = self.step_count >= self.episode_length - 1
        if self._termination_fn is not None:
            is_done = is_done or self._termination_fn(observation)
        if not self.observation_space['position'].contains(observation['position']):
            is_done = True
            reward = -100  # try making length of episode * min reward

        if self.visualization:
            self.cube_viz.update_cube_orientation(
                observation['achieved_goal']['position'],
                observation['achieved_goal']['orientation'],
                observation['desired_goal']['position'],
                observation['desired_goal']['orientation']
            )
            time.sleep(0.01)

        return observation, reward, is_done, self.info.copy()

    def reset(self):
        # By changing the `_reset_*` method below you can switch between using
        # the platform frontend, which is needed for the submission system, and
        # the direct simulation, which may be more convenient if you want to
        # pre-train locally in simulation.
        self._reset_direct_simulation()
        if self.visualization:
            self.cube_viz.reset()

        self.step_count = 0

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        self.prev_observation, _, _, _ = self.step(self.initial_action)
        return self.prev_observation

    @staticmethod
    def __connect_to_pybullet(enable_visualization):
        """
        Connect to the Pybullet client via either GUI (visual rendering
        enabled) or DIRECT (no visual rendering) physics servers.

        In GUI connection mode, use ctrl or alt with mouse scroll to adjust
        the view of the camera.
        """
        if enable_visualization:
            pybullet_client_id = p.connect(p.GUI)
        else:
            pybullet_client_id = p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        return pybullet_client_id

    def __setup_pybullet_simulation(self):
        """
        Set the physical parameters of the world in which the simulation
        will run, and import the models to be simulated
        """
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._pybullet_client_id,
        )
        p.setGravity(
            0,
            0,
            -9.81,
            physicsClientId=self._pybullet_client_id,
        )
        p.setTimeStep(
            self.time_step_s, physicsClientId=self._pybullet_client_id
        )

        p.loadURDF(
            "plane_transparent.urdf",
            [0, 0, 0],
            physicsClientId=self._pybullet_client_id,
        )

    def _reset_direct_simulation(self):
        """Reset direct simulation.

        With this the env can be used without backend.
        """

        # reset simulation
        del self.cube

        self._pybullet_client_id = self.__connect_to_pybullet(
            enable_visualization=self.visualization
        )
        self.__setup_pybullet_simulation()

        # initialize simulation
        if self.initializer is None:
            initial_object_pose = move_cube.sample_goal(difficulty=-1)
        else:
            initial_object_pose = self.initializer.get_initial_state()
            if self.default_goal is None:
                self.goal = self.initializer.get_goal().to_dict()
                self.goal_list.append(self.goal)
            else:
                self.goal = self.default_goal

        if not self.observation_space['desired_goal'].contains(self.goal):
            raise ValueError("Invalid goal pose.")

        self.cube = collision_objects.Cube(
            position=initial_object_pose.position,
            orientation=initial_object_pose.orientation,
            pybullet_client_id=self._pybullet_client_id,
            half_width=CUBE_HALF_WIDTH,
            mass=CUBE_MASS
        )

       # use mass of real cube
        p.changeDynamics(bodyUniqueId=self.cube.block, linkIndex=-1,
                         physicsClientId=self._pybullet_client_id,
                         mass=CUBE_MASS)
        # p.setTimeStep(0.001)
        # visualize the goal
        if self.visualization:
            self.goal_marker = CubeMarker(
                width=CUBE_WIDTH,
                position=self.goal["position"],
                orientation=self.goal["orientation"],
                pybullet_client_id=self._pybullet_client_id,
            )

    def _create_observation(self, action):
        cube_state = self.cube.get_state()
        position = np.asarray(cube_state[0])
        orientation = np.asarray(cube_state[1])
        velocity = p.getBaseVelocity(
            self.cube.block,
            physicsClientId=self.cube._pybullet_client_id,
        )
        velocity = np.concatenate([np.array(a) for a in velocity])
        return {'achieved_goal':
                {'position': position, 'orientation': orientation},
                'desired_goal': self.goal,
                'action': action,
                'observation': {'position': position,
                                'orientation': orientation,
                                'velocity': velocity}
                }


class RealRobotCubeEnv(gym.GoalEnv):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        action_type: ActionType = ActionType.POSITION,
        frameskip: int = 1,
        sim: bool = False,
        visualization: bool = False,
        reward_fn: callable = competition_reward,
        termination_fn: callable = None,
        initializer: callable = None,
        episode_length: int = move_cube.episode_length,
        path: str = None,
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
        """
        # Basic initialization
        # ====================
        self.path=path

        self._compute_reward = reward_fn
        self._termination_fn = termination_fn if sim else None
        self.initializer = initializer if sim else None
        self.goal = {k: np.array(v) for k, v in cube_goal_pose.items()}
        self.info = {"difficulty": goal_difficulty}
        self.difficulty = goal_difficulty

        self.action_type = action_type

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.real_platform = None
        self.platform = None
        self.simulation = sim
        self.visualization = visualization
        self.episode_length = episode_length
        self.custom_logs = {}
        self.reward_list = []
        self.observation_list = []
        self.change_goal_last = -1 # needed for evaluation
        self.reach_finish_point = -1 # needed for evaluation
        self.reach_start_point = -1 # needed for evaluation
        self.init_align_obj_error = -1 # needed for evaluation
        self.init_obj_pose = None # needed for evaluation
        self.align_obj_error = -1 # needed for evaluation
        self.goal_list = [] # needed for multiple goal environments
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
            self.action_space = robot_torque_space
            self.initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self.initial_action = INIT_JOINT_CONF  # trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space,
                }
            )
            self.initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": INIT_JOINT_CONF  # trifingerpro_limits.robot_position.default,
            }
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot": gym.spaces.Dict(
                    {
                        "position": robot_position_space,
                        "velocity": robot_velocity_space,
                        "torque": robot_torque_space,
                        "tip_positions": gym.spaces.Box(
                            low=np.array([trifingerpro_limits.object_position.low] * 3),
                            high=np.array([trifingerpro_limits.object_position.high] * 3),
                        ),
                        "tip_force": gym.spaces.Box(low=np.zeros(3),
                                                    high=np.ones(3))
                    }
                ),
                "action": self.action_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

        self.pinocchio_utils = PinocchioUtils()
        self.prev_observation = None
        self._prev_step_report = 0

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
            raise ValueError(
                "Given action is not contained in the action space."
            )

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
            reward += self._compute_reward(
                self.prev_observation,
                observation,
                self.info
            )
            self.prev_observation = observation

            self.step_count = t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length - 1:
                break

        is_done = self.step_count >= self.episode_length
        if self._termination_fn is not None:
            is_done = is_done or self._termination_fn(observation)

        # report current step_count
        if self.step_count - self._prev_step_report > 200:
            print('current step_count:', self.step_count)
            self._prev_step_report = self.step_count

        if is_done:
            print('is_done is True. Episode terminates.')
            print('episode length', self.episode_length)
            print('step_count', self.step_count)
            self.save_custom_logs()

        if self.visualization:
            self.cube_viz.update_cube_orientation(
                observation['achieved_goal']['position'],
                observation['achieved_goal']['orientation'],
                observation['desired_goal']['position'],
                observation['desired_goal']['orientation']
            )
            time.sleep(0.01)

        self.reward_list.append(reward)

        return observation, reward, is_done, self.info.copy()

    def reset(self):
        # By changing the `_reset_*` method below you can switch between using
        # the platform frontend, which is needed for the submission system, and
        # the direct simulation, which may be more convenient if you want to
        # pre-train locally in simulation.
        if self.simulation:
            self._reset_direct_simulation()
        else:
            self._reset_platform_frontend()
            self._reset_direct_simulation()
        if self.visualization:
            self.cube_viz.reset()

        self.step_count = 0

        # need to already do one step to get initial observation
        # TODO disable frameskip here?
        self.prev_observation, _, _, _ = self.step(self.initial_action)
        return self.prev_observation

    def _reset_platform_frontend(self):
        """Reset the platform frontend."""
        # reset is not really possible
        if self.real_platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

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
            self.goal = self.initializer.get_goal().to_dict()
        self.platform = trifinger_simulation.TriFingerPlatform(
            visualization=self.visualization,
            initial_object_pose=initial_object_pose,
        )
        # use mass of real cube
        p.changeDynamics(bodyUniqueId=self.platform.cube.block, linkIndex=-1,
                         physicsClientId=self.platform.simfinger._pybullet_client_id,
                         mass=CUBOID_MASS)
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

        observation = {
            "robot": {
                "position": robot_observation.position,
                "velocity": robot_observation.velocity,
                "torque": robot_observation.torque,
                "tip_positions": np.array(self.pinocchio_utils.forward_kinematics(robot_observation.position)),
                "tip_force": robot_observation.tip_force,
            },
            "action": action,
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
            obs['achieved_goal']['position'],
            obs['achieved_goal']['orientation']
        )
        # set robot position & velocity
        self.platform.simfinger.reset_finger_positions_and_velocities(
            obs['robot']['position'],
            obs['robot']['velocity']
        )

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = robot_interfaces.trifinger.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = robot_interfaces.trifinger.Action(
                position=gym_action
            )
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = robot_interfaces.trifinger.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def register_custom_log(self, name, data):
        if name in self.custom_logs:
            self.custom_logs[name].append({
                'step_count': self.step_count,
                'data': data
            })
        else:
            self.custom_logs[name] = [{
                'step_count': self.step_count,
                'data': data
            }]

    def save_custom_logs(self):
        print('saving custom logs...')
        custom_logdir = None
        if not(self.path is None):
            custom_logdir = self.path
        elif not os.path.isdir(CUSTOM_LOGDIR):
            print('{} does not exist. skip saving custom logs.'.format(CUSTOM_LOGDIR))
            return
        else:
            custom_logdir = CUSTOM_LOGDIR
        path = os.path.join(custom_logdir, 'custom_data')
        with shelve.open(path, writeback=True) as f:
            for key, val in self.custom_logs.items():
                f[key] = val

        # save the rewards
        path = os.path.join(custom_logdir, 'reward.pkl')
        with open(path, 'wb') as handle:
            pkl.dump(self.reward_list,handle)

        # if ran in simulation save the observation
        if (self.simulation):
            path = os.path.join(custom_logdir, 'observations.pkl')
            with open(path, 'wb') as handle:
                pkl.dump(self.observation_list,handle)

        # store the goal to a file, i.e. the last goal,...
        import json
        goal_file = os.path.join(custom_logdir, "goal.json")
        goal_info = {
            "difficulty": self.difficulty,
            "goal": json.loads(json.dumps({
        'position': self.goal["position"].tolist(),
        'orientation': self.goal["orientation"].tolist(),
        })),
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
        ex_state = move_cube.sample_goal(difficulty=-1) # ensures that on the ground
        if not(pos is None):
            self.goal["position"] = ex_state.position
            self.goal["position"][:3] = pos[:3]
        if not(orientation is None):
            self.goal["orientation"] = ex_state.orientation
            self.goal["orientation"] = orientation

        if (self.visualization):
            self.cube_viz.goal_viz = None

        if (log_timestep):
            if (self.simulation):
                self.change_goal_last = self.platform.get_current_timeindex()
            else:
                self.change_goal_last = self.real_platform.get_current_timeindex()

    def _get_obs_with_timeidx(self):
        from .reward_fns import _orientation_error

        # simply set the reach finish timestep
        if (self.simulation):
            timeidx = self.platform.get_current_timeindex()
        else:
            timeidx = self.real_platform.get_current_timeindex()
        obs = self._create_observation(timeidx, action=None)
        return obs, timeidx

    def set_reach_finish(self):
        from .reward_fns import _orientation_error
        obs, timeidx = self._get_obs_with_timeidx()
        self.align_obj_error = _orientation_error(obs)
        self.reach_finish_point = timeidx

    def set_reach_start(self):
        import json
        from .reward_fns import _orientation_error
        obs, timeidx = self._get_obs_with_timeidx()
        self.init_align_obj_error = _orientation_error(obs)
        self.init_obj_pose = json.loads(json.dumps({
            'position': obs['achieved_goal']['position'].tolist(),
            'orientation': obs['achieved_goal']['orientation'].tolist(),
        }))
        self.reach_start_point = timeidx
