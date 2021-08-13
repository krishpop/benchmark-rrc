"""Gym environment for the Real Robot Challenge Phase 2."""
import os
import enum
import shelve
import pickle as pkl
import copy

import gym
import numpy as np

try:
    import robot_interfaces
    import robot_fingers
    from robot_interfaces.trifinger import Action
except:
    robot_fingers = robot_interfaces = False
    from trifinger_simulation.action import Action

import cvxpy as cp
from diffcp import SolverError
from cvxpylayers.torch import CvxpyLayer
import pybullet_data
import trifinger_simulation
import pybullet as p
import os.path as osp
import inspect
import time
import torch
from gym import logger as gymlogger

from pybullet_utils import bullet_client
from trifinger_simulation import trifingerpro_limits, collision_objects
from trifinger_simulation.tasks import move_cube
from rrc.mp.const import CUSTOM_LOGDIR, INIT_JOINT_CONF, CUBOID_SIZE, CUBOID_MASS
from rrc.mp.const import CUBE_WIDTH, CUBE_HALF_WIDTH, CUBE_MASS
from rrc_iprl_package.control import controller_utils_cube as c_utils
from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag  
from typing import Union, Callable

from .reward_fns import training_reward1, training_reward2, training_reward3, training_reward, competition_reward, _orientation_error, _position_error
from .pinocchio_utils import PinocchioUtils
from .initializers import random_init, training_init, centered_init, fixed_g_init
from .viz import Viz, CuboidMarker, CubeMarker, VisualMarkers


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


def termination_fn(observation):
    low, high = trifingerpro_limits.object_position.low, trifingerpro_limits.object_position.high,
    curr_pos = observation['achieved_goal']['position']
    if not np.all(np.clip(curr_pos, low, high) == curr_pos):
        return True
    return False


class CubeEnv(gym.GoalEnv):
    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        drop_pen: float = -100.,
        frameskip: int = 1,
        visualization: bool = False,
        relative_goal: bool = False,
        reward_fn: callable = training_reward2,
        termination_fn: callable = None,
        initializer: callable = fixed_g_init,
        episode_length: int = move_cube.episode_length,
        force_factor: float = 0.5,
        torque_factor: float = 0.25,
        clip_action: bool = True,
        gravity: Union[float, Callable[[], int]] = -9.81,
        debug: bool = False
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
        self.relative_goal = relative_goal
        self.info = {"difficulty": goal_difficulty}
        self.difficulty = goal_difficulty
        self.drop_pen = drop_pen
        self.force_factor = force_factor
        self.torque_factor = torque_factor
        self.clip_action = clip_action
        if gravity is None:
            gravity = -9.81
        self._gravity = gravity
        self.debug = debug
        if debug:
            gymlogger.set_level(10)

        # TODO: The name "frameskip" makes sense for an atari environment but
        # not really for our scenario.  The name is also misleading as
        # "frameskip = 1" suggests that one frame is skipped while it actually
        # means "do one step per step" (i.e. no skip).
        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.visualization = visualization
        self.episode_length = episode_length
        self.goal_list = [] # needed for multiple goal environments
        self.cube = None
        self.goal_marker = None
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
        self._pybullet_client_id = -1
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(
                    {
                        "position": object_state_space.spaces['position'],
                        "velocity": gym.spaces.Box(low=-np.ones(6), high=np.ones(6)),
                        "orientation": object_state_space.spaces['orientation'],
                        "action": self.action_space},
                ),
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

        self.prev_observation = None
        self._prev_step_report = 0

    def compute_reward(self, achieved_goal, desired_goal, info):
        if not isinstance(info, dict):
            p_obs = [d['p_obs'] for d in info]
            obs = [d['obs'] for d in info]
            for i, (p_d, d) in enumerate(zip(p_obs, obs)):
                virtual_goal = desired_goal[i]
                if not isinstance(desired_goal, dict):
                    virtual_goal = {'position': virtual_goal[4:],
                                    'orientation': virtual_goal[:4]}
                d['desired_goal'] = virtual_goal
                p_d['desired_goal'] = virtual_goal
            return np.array([self._compute_reward(p, o, i)
                             for p, o, i in zip(p_obs, obs, info)])
        else:
            p_obs, obs = info['p_obs'], info['obs']
            if not isinstance(desired_goal, dict):
                desired_goal = {'position': desired_goal[4:] ,'orientation': desired_goal[:4]}
            obs['desired_goal'] = desired_goal
            obs['action'] = info['action']
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
        if self.clip_action:
            action = np.clip(action, -1, 1)
        action[:3] *= self.force_factor
        action[3:] *= self.torque_factor
        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip
        num_steps = int(np.asarray(num_steps).item())

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > self.episode_length + 1:
            excess = step_count_after - self.episode_length + 1
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        p_obs = None
        for _ in range(num_steps):
            # send action to robot
            p.applyExternalForce(self.cube.block, -1,
                    action[:3], [0,0,0], p.LINK_FRAME, physicsClientId=self._pybullet_client_id)
            p.applyExternalTorque(self.cube.block, -1,
                    action[3:], p.LINK_FRAME, physicsClientId=self._pybullet_client_id)

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
            p_obs = self.prev_observation.copy()
            self.prev_observation = observation

            self.step_count += 1  # t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length - 1:
                break

        info = self.info.copy()
        is_done = self.step_count >= self.episode_length + 1
        if termination_fn(observation):
            is_done = True
        if self._termination_fn is not None:
            is_done = is_done or self._termination_fn(observation)
            info['is_success'] = self._termination_fn(observation)
            reward += 500 * self._termination_fn(observation)

        info['ori_err'] = _orientation_error(observation)
        info['pos_err'] = _position_error(observation)
        # TODO (cleanup): Skipping keys to avoid storing unnecessary keys in RB
        info['p_obs'] = {k: p_obs[k] for k in ['desired_goal', 'achieved_goal']}
        info['obs'] = {k: observation[k] for k in ['desired_goal', 'achieved_goal', 'observation']}

        if self.visualization:
            self.cube_viz.update_cube_orientation(
                observation['achieved_goal']['position'],
                observation['achieved_goal']['orientation'],
                observation['desired_goal']['position'],
                observation['desired_goal']['orientation']
            )
            time.sleep(0.01)
            if self.debug:
                self._p.addUserDebugText("R:{:3.2f}, Pos:{:3.2f}, Ori:{:3.2f}".format(
                    reward, info['pos_err'], info['ori_err']),
                    [-0.5, -0.2, 0.05], textColorRGB=[0,0,0], lifeTime=0.14, textSize=1.5,
                    physicsClientId=self._pybullet_client_id)

        return observation, reward, is_done, info

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

    def __connect_to_pybullet(self, enable_visualization):
        """
        Connect to the Pybullet client via either GUI (visual rendering
        enabled) or DIRECT (no visual rendering) physics servers.

        In GUI connection mode, use ctrl or alt with mouse scroll to adjust
        the view of the camera.
        """
        if self._pybullet_client_id < 0:
            if enable_visualization:
                self._p = bullet_client.BulletClient(connection_mode=p.GUI)
            else:
                self._p = bullet_client.BulletClient()
            self._p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        return self._p._client

    def __setup_pybullet_simulation(self):
        """
        Set the physical parameters of the world in which the simulation
        will run, and import the models to be simulated
        """
        self._p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            # physicsClientId=self._pybullet_client_id,
        )
        self._p.setGravity(
            0,
            0,
            self.gravity,
            # physicsClientId=self._pybullet_client_id,
        )
        self._p.setTimeStep(
            self.time_step_s, # physicsClientId=self._pybullet_client_id
        )

        self._p.loadURDF(
            "plane_transparent.urdf",
            [0, 0, 0],
            # physicsClientId=self._pybullet_client_id,
        )
        self.__load_stage()

    def _disconnect_from_pybullet(self):
        """Disconnect from the simulation.

        Disconnects from the simulation and sets simulation to disabled to
        avoid any further function calls to it.
        """
        if self._pybullet_client_id is not None and p.isConnected(physicsClientId=self._pybullet_client_id):
            p.disconnect(
                physicsClientId=self._pybullet_client_id,
            )

    def _reset_direct_simulation(self):
        """Reset direct simulation.

        With this the env can be used without backend.
        """

        # reset simulation
        del self.cube

        if self.visualization:
            del self.goal_marker

        # self._disconnect_from_pybullet()
        if self._pybullet_client_id >= 0:
            self._p.resetSimulation()

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
        self._p.changeDynamics(bodyUniqueId=self.cube.block, linkIndex=-1,
                         # physicsClientId=self._pybullet_client_id,
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
        obs_pos = position = np.asarray(cube_state[0])
        obs_rot = orientation = np.asarray(cube_state[1])
        velocity = p.getBaseVelocity(
            self.cube.block,
            physicsClientId=self.cube._pybullet_client_id,
        )
        velocity = np.concatenate([np.array(a) for a in velocity])

        if self.relative_goal:
            obs_pos = position - self.goal['position']
            goal_rot = Rotation.from_quat(self.goal['orientation'])
            actual_rot = Rotation.from_quat(obs_rot)
            obs_rot = (goal_rot*actual_rot.inv()).as_quat()
        return {'achieved_goal':
                {'position': position, 'orientation': orientation},
                'desired_goal': self.goal,
                'observation': {'position': obs_pos,
                                'orientation': obs_rot,
                                'velocity': velocity,
                                'action': action}
                }

    def __load_stage(self, high_border=True):
        """Create the stage (table and boundary).

        Args:
            high_border:  Only used for the TriFinger.  If set to False, the
                old, low boundary will be loaded instead of the high one.
        """

        def mesh_path(filename):
            trifinger_path = osp.split(trifinger_simulation.__file__)[0]
            robot_properties_path = osp.join(trifinger_path, "robot_properties_fingers")
            return os.path.join(
                robot_properties_path, "meshes", "stl", filename
            )

        table_colour = (0.18, 0.15, 0.19, 1.0)
        high_border_colour = (0.73, 0.68, 0.72, 1.0)
        if high_border:
            collision_objects.import_mesh(
                mesh_path("trifinger_table_without_border.stl"),
                position=[0, 0, 0],
                is_concave=False,
                color_rgba=table_colour,
                pybullet_client_id=self._pybullet_client_id,
            )
            collision_objects.import_mesh(
                mesh_path("high_table_boundary.stl"),
                position=[0, 0, 0],
                is_concave=True,
                color_rgba=high_border_colour,
                pybullet_client_id=self._pybullet_client_id,
            )
        else:
            collision_objects.import_mesh(
                mesh_path("BL-M_Table_ASM_big.stl"),
                position=[0, 0, 0],
                is_concave=True,
                color_rgba=table_colour,
                pybullet_client_id=self._pybullet_client_id,
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

    @property
    def gravity(self):
        if callable(self._gravity):
            return self._gravity()
        else:
            return self._gravity

    @gravity.setter
    def gravity(self, x: Union[float, Callable[[], float]]):
        self._gravity = x

    def close(self):
        self._disconnect_from_pybullet()
        super().close()


class ContactForceCubeEnv(CubeEnv):
    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        drop_pen: float = -100.,
        frameskip: int = 1,
        visualization: bool = False,
        relative_goal: bool = False,
        reward_fn: callable = training_reward2,
        termination_fn: callable = None,
        initializer: callable = fixed_g_init,
        episode_length: int = move_cube.episode_length,
        force_factor: float = 0.5,
        torque_factor: float = 0.25,
        clip_action: bool = True,
        gravity: Union[float, Callable[[], int]] = -9.81,
        reset_contacts: bool = False,
        tip_wf: bool = True,
        debug: bool = True
    ):
        super(ContactForceCubeEnv, self).__init__(cube_goal_pose,
                goal_difficulty,
                drop_pen,
                frameskip,
                visualization,
                relative_goal,
                reward_fn,
                termination_fn,
                initializer,
                episode_length,
                force_factor,
                torque_factor,
                clip_action,
                gravity,
                debug)

        self.reset_contacts = reset_contacts
        low = -np.ones(9)
        high = np.ones(9)
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.initial_action = np.max([np.zeros(9), low], axis=0)
        self.cp_params = [np.array([0., 1., 0.]), np.array([1., 0., 0.]), np.array([-1.,  0.,  0.])]
        # sets sign for contact forces in x-y-z, depending on if it is +/- wrt object origin
        # only used for x/y directions
        self.observation_space.spaces['observation'].spaces['action'] = self.action_space
        self.observation_space.spaces['observation'].spaces['tip_positions'] = gym.spaces.Box(
                    low=np.concatenate(
                        [trifingerpro_limits.object_position.low for _ in range(3)]),
                    high=np.concatenate(
                        [trifingerpro_limits.object_position.high for _ in range(3)]),
                )
        if self.visualization:
            self.contact_viz = None
        self.cp_force_lines = []

    def step(self, action):
        # assert self.action_space.contains(action), f'Action: {action} not contained in action space'
        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > self.episode_length + 1:
            excess = step_count_after - self.episode_length + 1
            num_steps = max(1, num_steps - excess)

        reward = 0.
        p_obs = None
        for i in range(num_steps):
            obs = self.prev_observation
            self.apply_action(action, obs)
            observation = self._create_observation(action)

            if self.prev_observation is None:
                self.prev_observation = observation

            p_obs = self.prev_observation.copy()
            self.prev_observation = observation
            reward += self._compute_reward(
                self.prev_observation,
                observation,
                self.info
            )
            self.step_count += 1  # t
            # make sure to not exceed the episode length
            if self.step_count >= self.episode_length - 1:
                break

        info = self.info.copy()
        is_done = self.step_count >= self.episode_length + 1
        if termination_fn(observation):
            is_done = True
        if self._termination_fn is not None:
            term_bonus = 500
            is_done = is_done or self._termination_fn(observation)
            info['is_success'] = self._termination_fn(observation)
            reward += term_bonus * self._termination_fn(observation)

        info['ori_err'] = _orientation_error(observation)
        info['pos_err'] = _position_error(observation)
        # TODO (cleanup): Skipping keys to avoid storing unnecessary keys in RB
        info['p_obs'] = {k: p_obs[k] for k in ['desired_goal', 'achieved_goal']}
        info['obs'] = {k: observation[k] for k in ['desired_goal', 'achieved_goal', 'observation']}

        if self.visualization:
            self.cube_viz.update_cube_orientation(
                observation['achieved_goal']['position'],
                observation['achieved_goal']['orientation'],
                observation['desired_goal']['position'],
                observation['desired_goal']['orientation']
            )
            time.sleep(0.01)
            self._p.addUserDebugText("R:{:3.2f}, Pos:{:3.2f}, Ori:{:3.2f}".format(
                reward, info['pos_err'], info['ori_err']),
                [-0.5, -0.2, 0.05], textColorRGB=[0,0,0], lifeTime=0.14, textSize=1.5,
                physicsClientId=self._pybullet_client_id)

        # if is_done:
            # self._disconnect_from_pybullet()

        return observation, reward, is_done, info

    def apply_action(self, action, observation):
        if observation is None:
            observation = self._create_observation(action)
        cube_pose = move_cube.Pose.from_dict(observation['achieved_goal'])
        if self.cp_params is None:
            self.cp_params = c_utils.get_lifting_cp_params(cube_pose)
            gymlogger.debug("Closest ground face: {}".format(
                c_utils.get_closest_ground_face(cube_pose)))

        action = action.reshape((3,3))

        cp_wf_list = observation['observation']['tip_positions'].reshape((3,3))
        cp_list = self.get_cp_of_list(self.cp_params)
        ft_force_des = []

        for i, l_cf in enumerate(action):
            cube_ftip_pos_wf = c_utils.get_cp_pos_wf_from_cp_param(
                    self.cp_params[i], cube_pose.position, cube_pose.orientation)
            cp_of = cp_list[i]
            R_cp_2_o = Rotation.from_quat(cp_of.quat_of)
            R_o_2_w = Rotation.from_quat(observation['achieved_goal']['orientation'])
            l_o = R_cp_2_o.apply(l_cf)
            l_wf = R_o_2_w.apply(l_o)
            gymlogger.debug(f"{l_o}, {cp_of.pos_of}")
            ft_force_des.append(l_wf)
            p.applyExternalForce(self.cube.block, -1,
                forceObj=l_wf, posObj=cube_ftip_pos_wf,
                flags=p.WORLD_FRAME, physicsClientId=self._pybullet_client_id)

        # Visualize forces to check if they are being applied on contact points
        if self.debug:
            self.visualize_forces(cp_wf_list, ft_force_des) 

        p.stepSimulation(
                physicsClientId=self._pybullet_client_id,
            )
        return

    def get_cp_of_list(self, cp_params):
        cp_list = []
        # cube_ftip_pos_wf = c_utils.get_cp_pos_wf_from_cp_params(
        #            self.cp_params)
 
        for cp_param in cp_params:
            if cp_param is not None:
                cp_of = c_utils.get_cp_of_from_cp_param(cp_param)
                cp_list.append(cp_of)
        return cp_list

    def _create_observation(self, action):
        cube_state = self.cube.get_state()
        obs_pos = position = np.asarray(cube_state[0])
        obs_rot = orientation = np.asarray(cube_state[1])
        velocity = p.getBaseVelocity(
            self.cube.block,
            physicsClientId=self.cube._pybullet_client_id,
        )
        velocity = np.concatenate([np.array(a) for a in velocity])

        if self.relative_goal:
            obs_pos = position - self.goal['position']
            goal_rot = Rotation.from_quat(self.goal['orientation'])
            actual_rot = Rotation.from_quat(obs_rot)
            obs_rot = (goal_rot*actual_rot.inv()).as_quat()

        if self.cp_params is None:
            cube_pose = move_cube.Pose(position=position, orientation=orientation)
            self.cp_params = c_utils.get_lifting_cp_params(cube_pose)
        cube_ftip_pos_wf = c_utils.get_cp_pos_wf_from_cp_params(
                    self.cp_params, position, orientation)
        if self.visualization:
            if self.contact_viz is None:
                self.contact_viz = VisualMarkers()
                self.contact_viz.add(cube_ftip_pos_wf)
            else:
                for i, marker in enumerate(self.contact_viz.markers):
                    pos = cube_ftip_pos_wf[i]
                    marker.set_state(pos)
        cube_ftip_pos_wf = np.concatenate(cube_ftip_pos_wf)
        # cube_ftip_pos_of = np.concatenate([
        #    c_utils.get_cp_of_from_cp_param(cp_param) 
        #    for cp_param in self.cp_params])

        obs = {'achieved_goal':
                {'position': position, 'orientation': orientation},
                'desired_goal': self.goal,
                'observation': {'position': obs_pos,
                                'orientation': obs_rot,
                                'velocity': velocity,
                                'tip_positions': cube_ftip_pos_wf,
                                'action': action}
                }
        if self.reset_contacts:
            self.cp_params = None
        return obs

    def reset(self):
        if self.visualization and self.contact_viz is not None:
            del self.contact_viz
            self.contact_viz = None
        ret = super(ContactForceCubeEnv, self).reset()
        return ret

    def visualize_forces(self, cp_pos_list, ft_force_list):
        # import pdb; pdb.set_trace()
        if not self.cp_force_lines:
            self.cp_force_lines = [self._p.addUserDebugLine(
                pos, pos+np.linalg.norm(l_len)*(l_len), [1,0,0])
                for pos, l_len in zip(cp_pos_list, ft_force_list)]
        else:
            self.cp_force_lines = [self._p.addUserDebugLine(
                pos, pos+np.linalg.norm(l_len)*(l_len), [1,0,0],
                replaceItemUniqueId=l_id) for pos, l_len, l_id
                in zip(cp_pos_list, ft_force_list, self.cp_force_lines)]


class ContactForceWrenchCubeEnv(ContactForceCubeEnv):
    INFEASIBLE_PENALTY = 0.
    def __init__(
        self,
        cube_goal_pose: dict,
        goal_difficulty: int,
        drop_pen: float = -100.,
        frameskip: int = 1,
        visualization: bool = False,
        relative_goal: bool = False,
        reward_fn: callable = training_reward2,
        termination_fn: callable = None,
        initializer: callable = fixed_g_init,
        episode_length: int = move_cube.episode_length,
        force_factor: float = 0.5,
        torque_factor: float = 0.25,
        clip_action: bool = True,
        gravity: Union[float, Callable[[], int]] = -9.81,
        reset_contacts: bool = False,
        tip_wf: bool = True,
        debug: bool = True,
        cone_approx: bool = False,
        use_relaxed: bool = False
    ):
        super(ContactForceWrenchCubeEnv, self).__init__(cube_goal_pose,
                goal_difficulty,
                drop_pen,
                frameskip,
                visualization,
                relative_goal,
                reward_fn,
                termination_fn,
                initializer,
                episode_length,
                force_factor,
                torque_factor,
                clip_action,
                gravity,
                reset_contacts,
                tip_wf,
                debug)
        self.cone_approx = cone_approx
        self.use_relaxed = use_relaxed
        low = -np.ones(6)
        high = np.ones(6)
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.initial_action = np.max([np.zeros(6), low], axis=0)
        self.default_cp_params = self.cp_params = [
                np.array([0., 1., 0.]), np.array([1., 0., 0.]), 
                np.array([-1.,  0.,  0.])]
        self.observation_space.spaces['observation'].spaces['action'] = self.action_space
        if self.visualization:
            self.contact_viz = None
        self.setup_contact_force_opt()

    def setup_contact_force_opt(self):
        self.obj_mu = 1.
        self.mass = 0.016
        self.target_n = 0.5

        # Try solving optimization problem
        # contact force decision variable
        self.target_n_t = torch.as_tensor(np.array([self.target_n,0,0]*3), dtype=torch.float32)
        self.target_n_cp = cp.Parameter((9,), name='target_n', value=self.target_n_t.data.numpy())
        self.L = cp.Variable(9, name='l')
        self.W = cp.Parameter((6,), name='w_des')
        self.G = cp.Parameter((6, 9), name='grasp_m')
        self.R_w_2_o = cp.Parameter((6, 6), name='r_w_2_o')
        cm = np.vstack((np.eye(3), np.zeros((3, 3)))) * self.mass
        # self.Cm = cp.Parameter((6, 3), value=cm*self.mass, name='com')

        f_g = np.array([0, 0, self._gravity])
        w_ext = -self.W + self.R_w_2_o @ cm @ f_g
        f = self.G @ self.L + w_ext  # generated contact forces must balance wrench

        # Objective function - minimize force magnitudes
        contact_force = self.L - self.target_n_cp
        cost = cp.sum_squares(contact_force)

        # Friction cone constraints; >= 0
        self.constraints = []
        self.cone_constraints = []
        if self.cone_approx:
            #constraints.append(cp.abs(L[3*i + 1]) <= self.obj_mu * L[3*i])
            #constraints.append(cp.abs(L[3*i + 2]) <= self.obj_mu * L[3*i])
            self.cone_constraints += [cp.abs(self.L[1::3]) <= self.obj_mu * self.L[::3]]
            self.cone_constraints += [cp.abs(self.L[2::3]) <= self.obj_mu * self.L[::3]]
        else:
            self.cone_constraints.append(cp.SOC(self.obj_mu * self.L[::3],
                                           (self.L[2::3] + self.L[1::3])[None]))
        # constraints.append(L[::3] >= 0)
        self.constraints.append(f == np.zeros(f.shape))
  
        self.prob = cp.Problem(cp.Minimize(cost), self.cone_constraints + self.constraints)
        self.relaxed_prob = cp.Problem(cp.Minimize(cost), self.constraints)

        self.policy = CvxpyLayer(self.prob,
                [self.G, self.W, self.R_w_2_o, self.target_n_cp], [self.L])
        self.relaxed_policy = CvxpyLayer(self.relaxed_prob,
                [self.G, self.W, self.R_w_2_o, self.target_n_cp], [self.L])

        return

    def get_obj_position(self):
        cube_state = self.cube.get_state()
        obs_pos = np.asarray(cube_state[0])
        obs_rot = np.asarray(cube_state[1])
        return np.concatenate([obs_pos, obs_rot])

    def step(self, action):
        forces = self.get_balance_contact_forces(action)
        if forces is None:
            forces = np.zeros(9)
            rew = self.INFEASIBLE_PENALTY
            o, r, d, i = super(ContactForceWrenchCubeEnv, self).step(forces) 
            i['infeasible'] = True
            return o, rew, d, i
        else:
            o, r, d, i = super(ContactForceWrenchCubeEnv, self).step(forces)
            i['infeasible'] = False
            return o, r, d, i

    def solve_fop(self, G_t, des_wrench_t, R_w_2_o_t, relaxed=False):
        if relaxed and self.use_relaxed:
            policy = self.relaxed_policy
        else:
            policy = self.policy

        try:
            balance_force, = policy(G_t, des_wrench_t, R_w_2_o_t,
                                    self.target_n_t)
        except SolverError as e:
            balance_force = None
        return balance_force

    def get_balance_contact_forces(self, des_wrench, obj_pose=None):
        if obj_pose is None:
            obj_pose = self.get_obj_position()

        cp_list = self.get_cp_of_list(self.cp_params)

        # Get world to object rotation matrix
        quat_o_2_w = obj_pose[3:]
        R_w_2_o = Rotation.from_quat(quat_o_2_w).as_matrix().T
        R_w_2_o = block_diag(R_w_2_o, R_w_2_o)
        R_w_2_o_t = torch.from_numpy(R_w_2_o.astype('float32'))

        # Get grasp matrix
        G = self.get_grasp_matrix(cp_list, obj_pose)

        if self.step_count == 1 and self.debug:
            gymlogger.debug("Grasp matrix:\n{}".format(np.round(G,4)))
            import pdb; pdb.set_trace()
        G_t = torch.from_numpy(G.astype('float32'))
  
        # External wrench on object from gravity
        # des_wrench = des_wrench.squeeze().unsqueeze(-1)
        des_wrench_t = torch.from_numpy(des_wrench.astype('float32'))
  
        balance_force = self.solve_fop(G_t, des_wrench_t, R_w_2_o_t)
 
        if balance_force is None:
            if self.debug:
                gymlogger.debug('Did not solve contact force opt for '
                      'desired wrench: {}, contact points: {}'.format(
                        des_wrench, self.cp_params))
            balance_force = self.solve_fop(G_t, des_wrench_t, R_w_2_o_t,
                                           relaxed=True)
            if balance_force is None:
                if self.debug:
                    import pdb; pdb.set_trace()
                return balance_force

        balance_f = balance_force.detach().numpy()
        # gymlogger.debug("Balance force test: {}".format(G @ balance_f))
        return np.squeeze(balance_f)

    def get_grasp_matrix(self, cp_list, obj_pose):
        GT_list = []
        fnum = len(cp_list)
        H = self._get_H_matrix(fnum)
        for cp_of in cp_list:
            if cp_of is not None:
                GT_i = self._get_grasp_matrix_single_cp(cp_of, obj_pose)
                GT_list.append(GT_i)
            else:
                GT_list.append(np.zeros((3,3)))
        GT_full = np.concatenate(GT_list)
        GT = H @ GT_full
        return GT.T

    def _get_grasp_matrix_single_cp(self, cp_of, obj_pose):
        P = self._get_P_matrix(cp_of)
        # quat_o_2_w = obj_pose[3:]
        quat_cp_2_o = cp_of.quat_of

        # Orientation of cp frame w.r.t. world frame
        # quat_cp_2_w = quat_o_2_w * quat_cp_2_o
        # R_cp_2_w = Rotation.from_quat(quat_o_2_w) * Rotation.from_quat(quat_cp_2_o)
        # R is rotation matrix from contact frame i to world frame
        R_cp_2_o = Rotation.from_quat(quat_cp_2_o).as_matrix()
        # R = R_cp_2_w.as_matrix()
        R_bar_o = np.zeros((6,6))
        R_bar_o[0:3,0:3] = R_cp_2_o
        R_bar_o[3:6,3:6] = R_cp_2_o

        # R_bar_of = R_bar_o.T @ P.T
        G = P @ R_bar_o
        return G.T
  
    def _get_P_matrix(self, cp_of):
        cp_pos_of = cp_of.pos_of # Position of contact point in object frame

        S = np.array([
                     [0, -cp_pos_of[2], cp_pos_of[1]],
                     [cp_pos_of[2], 0, -cp_pos_of[0]],
                     [-cp_pos_of[1], cp_pos_of[0], 0]
                     ])

        P = np.eye(6)
        P[3:6,0:3] = S
        return P

    def _get_H_matrix(self, fnum):
        l_i = 3
        obj_dof = 6
        H_i = np.array([
                      [1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      ])
        H = np.zeros((l_i*fnum,obj_dof*fnum))
        for i in range(fnum):
          H[i*l_i:i*l_i+l_i, i*obj_dof:i*obj_dof+obj_dof] = H_i
        return H


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
        self._pybullet_client_id = self.platform.simfinger._pybullet_client_id
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
            robot_action = Action(torque=gym_action, position=np.zeros(9))
        elif self.action_type == ActionType.POSITION:
            robot_action = Action(
                position=gym_action
            )
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = Action(
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


def get_theta_z_wf(goal_rot, actual_rot):
    y_axis = [0, 1, 0]

    actual_direction_vector = actual_rot.apply(y_axis)

    goal_direction_vector = goal_rot.apply(y_axis)
    N = np.array([0, 0, 1]) # normal vector of ground plane
    proj = goal_direction_vector - goal_direction_vector.dot(N) * N
    goal_direction_vector = proj / np.linalg.norm(proj) # normalize projection

    orientation_error = np.arccos(
	goal_direction_vector.dot(actual_direction_vector)
    )

    return orientation_error


cube_env = CubeEnv
wrench_env = ContactForceWrenchCubeEnv
contact_env = ContactForceCubeEnv
real_env = RealRobotCubeEnv
