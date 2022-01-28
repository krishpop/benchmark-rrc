import os.path as osp
import numpy as np
import enum

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from rrc.env import controller_utils as c_utils
from rrc.env.pinocchio_utils import PinocchioUtils
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.tasks.move_cube import Pose


class TrajMode(enum.Enum):
    RESET = enum.auto()
    PRE_TRAJ_LOWER = enum.auto()
    PRE_TRAJ_REACH = enum.auto()
    REPOSE = enum.auto()
    ROTATE_X = enum.auto()
    ROTATE_Z = enum.auto()
    REPOSITION = enum.auto()
    RL_WRENCH = enum.auto()
    RELEASE = enum.auto()


class ImpedanceControllerPolicy:
    USE_FILTERED_POSE = True

    KP = [200, 200, 400, 200, 200, 400, 200, 200, 400]
    KV = [0.7, 0.7, 0.8, 0.7, 0.7, 0.8, 0.7, 0.7, 0.8]

    KP_OBJ = [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ]

    KV_OBJ = [
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
        0.001,
    ]

    # Re-orientation constants
    MIN_Z_ERROR = np.pi / 4
    MAX_Z_TRIES = 2
    Z_INCR = np.pi / 4

    FT_RADIUS = 0.0075

    def __init__(
        self,
        action_space=None,
        initial_pose=None,
        goal_pose=None,
        debug_waypoints=False,
        difficulty=None,
        save_path=None,
        load_dir=False,
        start_mode=TrajMode.RESET,
        ycb=False,
    ):
        if difficulty == 4:
            self.difficulty = 4
        else:
            self.difficulty = difficulty
        self.action_space = action_space
        self.debug_waypoints = debug_waypoints
        self.set_init_goal(initial_pose, goal_pose)
        self.init_face = None
        self.goal_face = None
        self.platform = None
        self.load_dir = load_dir
        self.ycb = ycb
        if self.ycb:
            self.FT_RADIUS = 0.0085

        self.start_mode = start_mode
        self.initialize_logging(save_path)
        self.load_model(load_dir)

    def initialize_logging(self, save_path=None):
        # CSV logging file path # need leading / for singularity image
        if osp.exists("/output"):
            self.csv_filepath = "/output/control_policy_data.csv"
            self.grasp_trajopt_filepath = "/output/grasp_trajopt_data"
            self.lift_trajopt_filepath = "/output/lift_trajopt_data"
            self.control_policy_log_filepath = "/output/control_policy_log"
        elif save_path:
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if proc_id is not None:
                time_str = time_str + "-" + str(proc_id())
            save_path = osp.join("./output", save_path, time_str)
            if not osp.exists(save_path):
                os.makedirs(save_path)
            self.csv_filepath = osp.join(save_path, "control_policy_data.csv")
            self.grasp_trajopt_filepath = osp.join(save_path, "grasp_trajopt_data")
            self.lift_trajopt_filepath = osp.join(save_path, "lift_trajopt_data")
            self.control_policy_log_filepath = osp.join(save_path, "control_policy_log")
        else:
            self.csv_filepath = None
            self.grasp_trajopt_filepath = None
            self.lift_trajopt_filepath = None
            self.control_policy_log_filepath = None

        # Lists for logging data
        # Rows correspond to step_count / timestamp
        self.l_step_count = []  # step_count
        self.l_timestamp = []  # time
        self.l_desired_ft_pos = []  # fingertip positions - desired
        self.l_actual_ft_pos = (
            []
        )  # fingertip positions - actual (computed from observation)
        self.l_desired_ft_vel = []  # fingertip velocities - desired
        self.l_desired_obj_pose = []  # object position - desired
        self.l_desired_obj_vel = []  # object position - desired
        self.l_observed_obj_pose = []  # object position - observed
        self.l_observed_filt_obj_pose = []  # object position - observed
        self.l_observed_obj_vel = []  # object velocity - observed
        self.l_desired_ft_force = []  # fingerip forces - desired
        self.l_desired_torque = []

        # Logs for debugging object pose feedback controller
        self.DEBUG = True
        self.l_dquat = []  # quaternion derivatives
        self.l_desired_obj_w = []

    def save_log(self):
        """Store logs in npz file"""
        if self.control_policy_log_filepath:
            np.savez(
                self.control_policy_log_filepath,
                step_count=np.asarray(self.l_step_count),
                timestamp=np.asarray(self.l_timestamp),
                desired_ft_pos=np.asarray(self.l_desired_ft_pos),
                desired_ft_vel=np.asarray(self.l_desired_ft_vel),
                actual_ft_pos=np.asarray(self.l_actual_ft_pos),
                desired_obj_pose=np.squeeze(np.asarray(self.l_desired_obj_pose)),
                desired_obj_vel=np.squeeze(np.asarray(self.l_desired_obj_vel)),
                observed_obj_pose=np.squeeze(np.asarray(self.l_observed_obj_pose)),
                observed_filt_obj_pose=np.squeeze(
                    np.asarray(self.l_observed_filt_obj_pose)
                ),
                observed_obj_vel=np.squeeze(np.asarray(self.l_observed_obj_vel)),
                desired_ft_force=np.squeeze(np.asarray(self.l_desired_ft_force)),
                desired_torque=np.squeeze(np.asarray(self.l_desired_torque)),
                dquat=np.squeeze(np.asarray(self.l_dquat)),
                desired_obj_w=np.squeeze(np.asarray(self.l_desired_obj_w)),
            )

    def reset_policy(self, observation, platform=None):
        if platform:
            self.platform = platform
        self.custom_pinocchio_utils = PinocchioUtils(
            self.platform.simfinger.finger_urdf_path,
            self.platform.simfinger.tip_link_names,
        )

        # Define nlp for finger traj opt
        nGrid = 30
        dt = 0.04
        self.finger_nlp = c_utils.define_static_object_opt(nGrid, dt)
        self.release_nlp = c_utils.define_static_object_opt(15, dt)

        # init_position = np.array([0.0, 0.9, -1.7, 0.0, 0.9, -1.7, 0.0, 0.9, -1.7])
        # Get joint positions
        current_position = observation['observation']['position']
        self.init_ft_pos = self.get_fingertip_pos_wf(current_position)
        self.init_ft_pos = np.asarray(self.init_ft_pos).flatten()

        # Previous object pose and time (for estimating object velocity)
        self.prev_obj_pose = get_pose_from_observation(observation)
        if osp.exists("/output"):
            self.prev_step_time = observation["observation"]["cam0_timestamp"]
        else:
            self.prev_step_time = observation["observation"]["cam0_timestamp"] / 1000
        self.prev_vel = np.zeros(6)
        self.filt_vel = np.zeros(6)

        self.filtered_obj_pose = get_pose_from_observation(observation)

        self.ft_pos_traj = np.tile(self.init_ft_pos, (10000, 1))
        self.ft_vel_traj = np.zeros((10000, 9))
        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None
        self.mode = self.start_mode  # TrajMode.RESET
        self.plan_trajectory(observation)

        # Counters
        self.step_count = 0  # Number of times predict() is called

        # Re-orientation try counters
        self.z_tries = 0

    def set_init_goal(self, initial_pose, goal_pose, flip=False):
        self.goal_pose = goal_pose
        self.x0 = np.concatenate([initial_pose.position, initial_pose.orientation])[
            None
        ]
        # init_goal_dist = np.linalg.norm(goal_pose.position - initial_pose.position)
        # print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}, '
        #      f'dist: {init_goal_dist}')
        # print(f'init orientation: {initial_pose.orientation}, goal orientation: {goal_pose.orientation}')

        """Get contact point parameters for either lifting"""

    def set_cp_params(self, observation):
        """Get contact point parameters for either lifting"""
        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        self.cp_params = c_utils.get_lifting_cp_params(obj_pose)

    def get_repose_mode_and_bounds(self, observation):
        """Set repose goal"""

        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        obj_pose = c_utils.get_aligned_pose(obj_pose)
        x0 = np.concatenate([obj_pose.position, obj_pose.orientation])[None]

        cur_R = Rotation.from_quat(obj_pose.orientation)

        # set object goal pose
        x_goal = x0.copy()
        x_goal[0, :3] = self.goal_pose.position

        # Get z error between goal and current object orientation
        theta_z = c_utils.get_y_axis_delta(obj_pose, self.goal_pose)
        # print("THETA_Z delta: {}".format(theta_z))

        # Set repose mode to ROTATE_z, ROTATE_X, or REPOSITION
        if self.load_dir:
            # Object should already be grasped after env.reset()
            mode = TrajMode.RL_WRENCH
        elif self.difficulty != 4:
            mode = TrajMode.REPOSITION
        else:
            if self.z_tries < self.MAX_Z_TRIES and np.abs(theta_z) > self.MIN_Z_ERROR:
                mode = TrajMode.ROTATE_Z
            else:
                mode = TrajMode.REPOSITION

        # TODO add rotation goal decomposition here
        if mode == TrajMode.ROTATE_Z:
            theta_z = np.clip(theta_z, -self.Z_INCR, self.Z_INCR)
            z_R = Rotation.from_euler("z", theta_z)

            new_R = z_R * cur_R
            x_goal[0, -4:] = new_R.as_quat()
            x_goal[0, 0] = 0
            x_goal[0, 1] = 0
            x_goal[0, 2] = c_utils.OBJ_SIZE[0] / 2

            # x_goal[0, -4:] = self.goal_pose.orientation

            self.z_tries += 1

        return mode, x0, x_goal

    def set_traj_repose_object(self, observation, x0, x_goal, nGrid=50, dt=0.01):
        """Run trajectory optimization to move object given fixed contact points"""
        self.traj_waypoint_counter = 0
        qnum = 3

        # print("Compute repose traj for MODE {}".format(self.mode))
        # print("Traj lift x0: {}".format(repr(x0)))
        # print("Traj lift x_goal: {}".format(repr(x_goal)))

        # Get current joint positions
        current_position, _ = observation['position']
        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        self.x_soln, self.dx_soln, l_wf_soln = c_utils.run_fixed_cp_traj_opt(
            self.cp_params,
            current_position,
            self.custom_pinocchio_utils,
            x0,
            x_goal,
            nGrid,
            dt,
            npz_filepath=self.lift_trajopt_filepath,
        )

        ft_pos = np.zeros((nGrid, 9))
        ft_vel = np.zeros((nGrid, 9))

        free_finger_id = None
        for i, cp in enumerate(self.cp_params):
            if cp is None:
                free_finger_id = i
                break

        for t_i in range(nGrid):
            # Set fingertip goal positions and velocities from x_soln, dx_soln
            next_cube_pos_wf = self.x_soln[t_i, 0:3]
            next_cube_quat_wf = self.x_soln[t_i, 3:]

            ft_pos_list = c_utils.get_cp_pos_wf_from_cp_params(
                self.cp_params, next_cube_pos_wf, next_cube_quat_wf
            )

            # Hold free_finger at current ft position
            if free_finger_id is not None:
                ft_pos_list[free_finger_id] = current_ft_pos[free_finger_id]
            ft_pos[t_i, :] = np.asarray(ft_pos_list).flatten()

            # Fingertip velocities
            ft_vel_arr = np.tile(self.dx_soln[t_i, 0:3], 3)
            if free_finger_id is not None:
                ft_vel_arr[
                    free_finger_id * qnum : free_finger_id * qnum + qnum
                ] = np.zeros(qnum)
            ft_vel[t_i, :] = ft_vel_arr

        # Add 0 forces for free_fingertip to l_wf
        l_wf = np.zeros((nGrid, 9))
        i = 0
        for f_i in range(3):
            if f_i == free_finger_id:
                continue
            l_wf[:, f_i * qnum : f_i * qnum + qnum] = l_wf_soln[
                :, i * qnum : i * qnum + qnum
            ]
            i += 1

        # Number of interpolation points
        interp_n = 26

        # Linearly interpolate between each position waypoint (row) and force waypoint
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid - 1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        # itp_vel = interp1d(row_ind_in, ft_vel, axis=0)
        itp_lwf = interp1d(row_ind_in, l_wf, axis=0)
        self.ft_pos_traj = itp_pos(row_coord_out)
        # self.ft_vel_traj = itp_vel(row_coord_out)
        self.l_wf_traj = itp_lwf(row_coord_out)

        # Linearly interpolate between each object pose
        # TODO: Does it make sense to linearly interpolate quaternions?
        itp_x_soln = interp1d(row_ind_in, self.x_soln, axis=0)
        self.x_traj = itp_x_soln(row_coord_out)

        # Zero-order hold for velocity waypoints
        self.ft_vel_traj = np.repeat(ft_vel, repeats=interp_n + 1, axis=0)[
            :-interp_n, :
        ]
        self.dx_traj = np.repeat(self.dx_soln, repeats=interp_n + 1, axis=0)[
            :-interp_n, :
        ]

    def set_traj_ft_init_pos(self, observation):
        """Set traj to raise fingers back to init pos"""
        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)
        obj_pose = c_utils.get_aligned_pose(obj_pose)

        # Get joint positions
        current_position, _ = get_robot_position_velocity(observation)
        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None

        # Raise fingers back to init position
        self.ft_pos_traj, self.ft_vel_traj = self.run_finger_traj_opt(
            current_position, obj_pose, self.init_ft_pos, self.release_nlp
        )

        # TODO
        """Set trajectory to retract fingers away from object"""

    def set_traj_to_object(self, observation):
        """Run trajectory optimization to move fingers to contact points on object"""
        self.traj_waypoint_counter = 0
        # First, set cp_params based on mode
        self.set_cp_params(observation)

        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        obj_pose = c_utils.get_aligned_pose(obj_pose)

        # Get joint positions
        current_position, _ = get_robot_position_velocity(observation)

        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        # Get list of desired fingertip positions
        if self.ycb:
            cp_wf_list = c_utils.get_cp_pos_wf_from_cp_params(
                self.cp_params,
                obj_pose.position,
                obj_pose.orientation,
                use_obj_size_offset=False,
            )
        else:
            cp_wf_list = c_utils.get_cp_pos_wf_from_cp_params(
                self.cp_params,
                obj_pose.position,
                obj_pose.orientation,
                use_obj_size_offset=False,
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
                temp = R @ np.array([0, 0, self.FT_RADIUS])
                new_pos = np.array(cp_wf_list[i]) + temp[:3]
                cp_wf_list[i] = new_pos

        ft_goal = np.asarray(cp_wf_list).flatten()
        self.ft_pos_traj, self.ft_vel_traj = self.run_finger_traj_opt(
            current_position, obj_pose, ft_goal, self.finger_nlp
        )
        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None

    def set_traj_lower_finger(self, observation):
        """Run traj opt to lower fingers to ground level"""
        self.traj_waypoint_counter = 0
        # First, set cp_params based on mode
        self.set_cp_params(observation)

        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)
        obj_pose = c_utils.get_aligned_pose(obj_pose)

        # Get joint positions
        current_position, _ = get_robot_position_velocity(observation)

        # Get current fingertip positions
        current_ft_pos = self.get_fingertip_pos_wf(current_position)

        ft_goal = c_utils.get_pre_grasp_ft_goal(
            obj_pose, current_ft_pos, self.cp_params
        )

        self.ft_pos_traj, self.ft_vel_traj = self.run_finger_traj_opt(
            current_position, obj_pose, ft_goal, self.release_nlp
        )
        self.l_wf_traj = None
        self.x_traj = None
        self.dx_traj = None

    def run_finger_traj_opt(self, current_position, obj_pose, ft_goal, nlp):
        """
        Run trajectory optimization for fingers, given fingertip goal positions
            Parameters:
                ft_goal: (9,) array of fingertip x,y,z goal positions in world frame
        """
        nGrid = nlp.nGrid
        self.traj_waypoint_counter = 0

        ft_pos, ft_vel = c_utils.get_finger_waypoints(
            nlp,
            ft_goal,
            current_position,
            obj_pose,
            npz_filepath=self.grasp_trajopt_filepath,
        )

        # Number of interpolation points
        interp_n = 26

        # Linearly interpolate between each waypoint (row)
        # Initial row indices
        row_ind_in = np.arange(nGrid)
        # Output row coordinates
        row_coord_out = np.linspace(0, nGrid - 1, interp_n * (nGrid - 1) + nGrid)
        # scipy.interpolate.interp1d instance
        itp_pos = interp1d(row_ind_in, ft_pos, axis=0)
        # itp_vel = interp1d(row_ind_in, ft_vel, axis=0)
        ft_pos_traj = itp_pos(row_coord_out)

        # Zero-order hold for velocity waypoints
        ft_vel_traj = np.repeat(ft_vel, repeats=interp_n + 1, axis=0)[:-interp_n, :]

        return ft_pos_traj, ft_vel_traj

    def log_to_buffers(
        self,
        ft_pos_goal_list,
        ft_vel_goal_list,
        cur_ft_pos,
        obj_pose,
        obj_vel,
        torque,
        ft_des_force_wf=None,
    ):
        # LOGGING
        self.l_step_count.append(self.step_count)
        self.l_timestamp.append(time.time())
        self.l_desired_ft_pos.append(np.asarray(ft_pos_goal_list).flatten())
        self.l_desired_ft_vel.append(np.asarray(ft_vel_goal_list).flatten())
        self.l_actual_ft_pos.append(cur_ft_pos)
        self.l_observed_obj_pose.append(
            np.concatenate((obj_pose.position, obj_pose.orientation))
        )
        self.l_observed_filt_obj_pose.append(
            np.concatenate(
                (self.filtered_obj_pose.position, self.filtered_obj_pose.orientation)
            )
        )
        self.l_observed_obj_vel.append(obj_vel)
        self.l_desired_torque.append(np.asarray(torque))

        if self.x_traj is None:
            # Nan if there is no obj traj (during grasping)
            self.l_desired_obj_pose.append(np.ones(7) * np.nan)
        else:
            self.l_desired_obj_pose.append(self.x_traj[self.traj_waypoint_counter, :])
        if self.dx_traj is None:
            # Nan if there is no obj traj (during grasping)
            self.l_desired_obj_vel.append(np.ones(6) * np.nan)
        else:
            self.l_desired_obj_vel.append(self.dx_traj[self.traj_waypoint_counter, :])
        if ft_des_force_wf is None:
            # Nan if no desired ft forces (during grasping)
            self.l_desired_ft_force.append(np.ones(9) * np.nan)
        else:
            self.l_desired_ft_force.append(ft_des_force_wf)
        return

    def plan_trajectory(self, observation):
        """Replans trajectory according to TrajMode, and sets self.traj_waypoint_counter"""
        # Get object pose
        if self.USE_FILTERED_POSE:
            obj_pose = self.filtered_obj_pose
        else:
            obj_pose = get_pose_from_observation(observation)

        if self.mode == TrajMode.RESET:
            self.set_traj_lower_finger(observation)
            self.mode = TrajMode.PRE_TRAJ_LOWER
        elif self.mode == TrajMode.PRE_TRAJ_LOWER:
            self.set_traj_to_object(observation)
            self.mode = TrajMode.PRE_TRAJ_REACH
        elif self.mode == TrajMode.PRE_TRAJ_REACH:
            self.mode, x0, x_goal = self.get_repose_mode_and_bounds(observation)
            if self.mode in [TrajMode.RL_WRENCH, TrajMode.REPOSITION]:
                self.set_traj_repose_object(observation, x0, x_goal, nGrid=50, dt=0.08)
            else:
                self.set_traj_repose_object(observation, x0, x_goal, nGrid=20, dt=0.08)
        elif self.mode == TrajMode.ROTATE_X or self.mode == TrajMode.ROTATE_Z:
            Get z error between goal and current object orientation
            theta_z = self.get_theta_z_wf(obj_pose)
            print("THETA_Z: {}".format(theta_z))
            if np.abs(theta_z) < self.MIN_Z_ERROR or self.z_tries > self.MAX_Z_TRIES:
               self.mode, x0, x_goal = self.get_repose_mode_and_bounds(observation)
               self.set_traj_repose_object(observation, x0, x_goal, nGrid=50, dt=0.08)
            else:
                self.set_traj_release_object(observation)
                self.mode = TrajMode.RELEASE
        elif self.mode == TrajMode.RELEASE:
            self.set_traj_ft_init_pos(observation)
            self.mode = TrajMode.RESET
        elif self.mode == TrajMode.REPOSITION:
            print(
                "ERROR: plan_trajectory() should not be called in TrajMode.REPOSITION"
            )

        self.traj_waypoint_counter = 0
        return

    # TODO: What about when object observations are noisy???
    def get_obj_vel(self, cur_obj_pose, cur_step_time):
        dt = cur_step_time - self.prev_step_time

        if dt == 0:
            return self.filt_vel

        obj_vel_position = (cur_obj_pose.position - self.prev_obj_pose.position) / dt

        # TODO: verify that we are getting angular velocities from quaternions correctly
        # cur_R = Rotation.from_quat(cur_obj_pose.orientation)
        # prev_R = Rotation.from_quat(self.prev_obj_pose.orientation)
        # delta_R = cur_R * prev_R.inv()
        # obj_vel_quat = delta_R.as_quat() / dt
        obj_vel_quat = (cur_obj_pose.orientation - self.prev_obj_pose.orientation) / dt
        M = c_utils.get_dquat_to_dtheta_matrix(
            self.prev_obj_pose.orientation
        )  # from Paul Mitiguy dynamics notes
        obj_vel_theta = 2 * M @ obj_vel_quat
        # obj_vel_theta = np.zeros(obj_vel_theta.shape)

        cur_vel = np.concatenate((obj_vel_position, obj_vel_theta))

        # Set previous obj_pose and step_time to current values
        self.prev_obj_pose = cur_obj_pose
        self.prev_step_time = cur_step_time
        self.prev_vel = cur_vel

        # filter the velocity
        theta = 0.1
        filt_vel = (1 - theta) * self.filt_vel + theta * cur_vel
        self.filt_vel = filt_vel.copy()

        # Log obj_vel_quat for debugging
        if self.DEBUG:
            self.l_dquat.append(obj_vel_quat)

        return filt_vel
        return cur_vel

    def predict(self, full_observation, residual_ft_force=None):
        if self.step_count == 0:
            init_pose = get_pose_from_observation(full_observation, goal=False)
            goal_pose = get_pose_from_observation(full_observation, goal=True)
            if self.difficulty == 4:
                self.flipping = flip_needed(init_pose, goal_pose)
            self.set_init_goal(init_pose, goal_pose)

        self.step_count += 1
        observation = full_observation["observation"]
        current_position, current_velocity = (
            observation["position"],
            observation["velocity"],
        )

        # Get object pose
        obj_pose = get_pose_from_observation(full_observation)
        # Filter object pose
        self.set_filtered_pose_from_observation(full_observation)

        # Estimate object velocity based on previous and current object pose
        # TODO: this might cause an issue if observed object poses are the same across steps?
        if osp.exists("/output"):
            timestamp = observation["cam0_timestamp"]
        else:
            timestamp = observation["cam0_timestamp"] / 1000
        # print("Cam0_timestamp: {}".format(timestamp))
        obj_vel = self.get_obj_vel(self.filtered_obj_pose, timestamp)

        # Get current fingertip position
        cur_ft_pos = self.get_fingertip_pos_wf(current_position)
        cur_ft_pos = np.asarray(cur_ft_pos).flatten()

        if self.traj_waypoint_counter == self.ft_pos_traj.shape[0]:
            # TODO: currently will redo the last waypoint after reaching end of trajectory
            self.plan_trajectory(full_observation)

        ft_pos_goal_list = []
        ft_vel_goal_list = []
        # If object is grasped, transform cp_wf to ft_wf
        if self.mode in [
            TrajMode.REPOSITION,
            TrajMode.ROTATE_X,
            TrajMode.ROTATE_Z,
            TrajMode.RL_WRENCH,
        ]:
            R_list = c_utils.get_ft_R(current_position)

        for f_i in range(3):
            new_pos = self.ft_pos_traj[
                self.traj_waypoint_counter, f_i * 3 : f_i * 3 + 3
            ]
            new_vel = self.ft_vel_traj[
                self.traj_waypoint_counter, f_i * 3 : f_i * 3 + 3
            ]

            if self.mode in [
                TrajMode.REPOSITION,
                TrajMode.ROTATE_X,
                TrajMode.ROTATE_Z,
                TrajMode.RL_WRENCH,
            ]:
                R = R_list[f_i]
                temp = R @ np.array([0, 0, self.FT_RADIUS])
                new_pos = np.array(new_pos) + temp[:3]
                new_pos = new_pos.tolist()

            ft_pos_goal_list.append(new_pos)
            ft_vel_goal_list.append(new_vel)

        if self.mode in [TrajMode.REPOSITION, TrajMode.ROTATE_X, TrajMode.ROTATE_Z]:
            ft_des_force_wf = self.l_wf_traj[self.traj_waypoint_counter, :]
        else:
            ft_des_force_wf = None

        if ft_des_force_wf is not None and residual_ft_force is not None:
            ft_des_force_wf += residual_ft_force
        elif residual_ft_force is not None:
            ft_des_force_wf = residual_ft_force

        # Compute torque with impedance controller, and clip
        torque = c_utils.impedance_controller(
            ft_pos_goal_list,
            ft_vel_goal_list,
            current_position,
            current_velocity,
            self.custom_pinocchio_utils,
            tip_forces_wf=ft_des_force_wf,
            Kp=self.KP,
            Kv=self.KV,
        )
        torque = np.clip(torque, self.action_space.low, self.action_space.high)

        self.log_to_buffers(
            ft_pos_goal_list,
            ft_vel_goal_list,
            cur_ft_pos,
            obj_pose,
            obj_vel,
            torque,
            ft_des_force_wf,
        )
        # always increment traj_waypoint_counter UNLESS in repose mode and have reached final waypoint
        if not (
            self.mode in [TrajMode.RL_WRENCH, TrajMode.REPOSITION]
            and self.traj_waypoint_counter == self.ft_pos_traj.shape[0] - 1
        ):
            self.traj_waypoint_counter += 1
        return torque

    def get_fingertip_pos_wf(self, current_q):
        """Get fingertip positions in world frame given current joint q"""
        fingertip_pos_wf = self.custom_pinocchio_utils.forward_kinematics(current_q)
        return fingertip_pos_wf

    def set_filtered_pose_from_observation(self, observation, theta=0.01):
        new_pose = get_pose_from_observation(observation)
        f_p = (1 - theta) * self.filtered_obj_pose.position + theta * new_pose.position
        f_o = new_pose.orientation
        # f_o = (1-theta) * self.filtered_obj_pose.orientation + theta * new_pose.orientation

        filt_pose = move_cube.Pose(position=f_p, orientation=f_o)
        self.filtered_obj_pose = filt_pose
        return filt_pose


def get_pose_from_observation(observation, goal=False):
    k = "desired_goal" if goal else "achieved_goal"
    return Pose.from_dict(observation[k])


def flip_needed(init_pose, goal_pose):
    return c_utils.get_closest_ground_face(
        init_pose
    ) != c_utils.get_closest_ground_face(goal_pose)
