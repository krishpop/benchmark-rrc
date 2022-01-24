import enum

import numpy as np
from rrc.env.fixed_contact_system import (
    ContactPoint,
    FixedContactPointOpt,
    StaticObjectOpt,
)
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from trifinger_simulation.tasks import move_cube

# Here, hard code the base position of the fingers (as angle on the arena)
r = 0.15
theta_0 = 90
theta_1 = 310
theta_2 = 200
# theta_2 = 3.66519 # 210 degrees
# CUBOID_SIZE in trifinger_simulation surf_2021 branch is set to cube dims
CUBOID_SIZE = move_cube._CUBOID_SIZE
CUBOID_HALF_SIZE = move_cube._CUBOID_HALF_SIZE
CUBE_SIZE = move_cube._CUBE_SIZE
CUBE_HALF_SIZE = move_cube._CUBE_SIZE[0] / 2 + 0.001
OBJ_SIZE = CUBOID_SIZE
OBJ_HALF_SIZE = CUBOID_HALF_SIZE
OBJ_MASS = 0.08  # 8 grams

FINGER_BASE_POSITIONS = [
    np.array(
        [[np.cos(theta_0 * (np.pi / 180)) * r, np.sin(theta_0 * (np.pi / 180)) * r, 0]]
    ),
    np.array(
        [[np.cos(theta_1 * (np.pi / 180)) * r, np.sin(theta_1 * (np.pi / 180)) * r, 0]]
    ),
    np.array(
        [[np.cos(theta_2 * (np.pi / 180)) * r, np.sin(theta_2 * (np.pi / 180)) * r, 0]]
    ),
]
BASE_ANGLE_DEGREES = [0, -120, -240]


class PolicyMode(enum.Enum):
    RESET = enum.auto()
    TRAJ_OPT = enum.auto()
    IMPEDANCE = enum.auto()
    RL_PUSH = enum.auto()
    RESIDUAL = enum.auto()


# Information about object faces given face_id
OBJ_FACES_INFO = {
    1: {
        "center_param": np.array([0.0, -1.0, 0.0]),
        "face_down_default_quat": np.array([0.707, 0, 0, 0.707]),
        "adjacent_faces": [6, 4, 3, 5],
        "opposite_face": 2,
        "up_axis": np.array([0.0, 1.0, 0.0]),  # UP axis when this face is ground face
    },
    2: {
        "center_param": np.array([0.0, 1.0, 0.0]),
        "face_down_default_quat": np.array([-0.707, 0, 0, 0.707]),
        "adjacent_faces": [6, 4, 3, 5],
        "opposite_face": 1,
        "up_axis": np.array([0.0, -1.0, 0.0]),
    },
    3: {
        "center_param": np.array([1.0, 0.0, 0.0]),
        "face_down_default_quat": np.array([0, 0.707, 0, 0.707]),
        "adjacent_faces": [1, 2, 4, 6],
        "opposite_face": 5,
        "up_axis": np.array([-1.0, 0.0, 0.0]),
    },
    4: {
        "center_param": np.array([0.0, 0.0, 1.0]),
        "face_down_default_quat": np.array([0, 1, 0, 0]),
        "adjacent_faces": [1, 2, 3, 5],
        "opposite_face": 6,
        "up_axis": np.array([0.0, 0.0, -1.0]),
    },
    5: {
        "center_param": np.array([-1.0, 0.0, 0.0]),
        "face_down_default_quat": np.array([0, -0.707, 0, 0.707]),
        "adjacent_faces": [1, 2, 4, 6],
        "opposite_face": 3,
        "up_axis": np.array([1.0, 0.0, 0.0]),
    },
    6: {
        "center_param": np.array([0.0, 0.0, -1.0]),
        "face_down_default_quat": np.array([0, 0, 0, 1]),
        "adjacent_faces": [1, 2, 3, 5],
        "opposite_face": 4,
        "up_axis": np.array([0.0, 0.0, 1.0]),
    },
}

"""
Compute joint torques to move fingertips to desired locations
Inputs:
tip_pos_desired_list: List of desired fingertip positions for each finger
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""


def impedance_controller(
    tip_pos_desired_list,
    tip_vel_desired_list,
    q_current,
    dq_current,
    custom_pinocchio_utils,
    tip_forces_wf=None,
    Kp=(25, 25, 25, 25, 25, 25, 25, 25, 25),
    Kv=(1, 1, 1, 1, 1, 1, 1, 1, 1),
    grav=-9.81,
):
    torque = 0
    for finger_id in range(3):
        # Get contact forces for single finger
        if tip_forces_wf is None:
            f_wf = None
        else:
            f_wf = np.expand_dims(
                np.array(tip_forces_wf[finger_id * 3 : finger_id * 3 + 3]), 1
            )

        finger_torque = impedance_controller_single_finger(
            finger_id,
            tip_pos_desired_list[finger_id],
            tip_vel_desired_list[finger_id],
            q_current,
            dq_current,
            custom_pinocchio_utils,
            tip_force_wf=f_wf,
            Kp=Kp,
            Kv=Kv,
            grav=grav,
        )
        torque += finger_torque
    return torque


"""
Compute joint torques to move fingertip to desired location
Inputs:
finger_id: Finger 0, 1, or 2
tip_desired: Desired fingertip pose **ORIENTATION??**
    for orientation: transform fingertip reference frame to world frame (take
        into account object orientation)
    for now, just track position
q_current: Current joint angles
dq_current: Current joint velocities
tip_forces_wf: fingertip forces in world frame
tol: tolerance for determining when fingers have reached goal
"""


def impedance_controller_single_finger(
    finger_id,
    tip_pos_desired,
    tip_vel_desired,
    q_current,
    dq_current,
    custom_pinocchio_utils,
    tip_force_wf=None,
    Kp=(25, 25, 25, 25, 25, 25, 25, 25, 25),
    Kv=(1, 1, 1, 1, 1, 1, 1, 1, 1),
    grav=-9.81,
):

    Kp_x = Kp[finger_id * 3 + 0]
    Kp_y = Kp[finger_id * 3 + 1]
    Kp_z = Kp[finger_id * 3 + 2]
    Kp = np.diag([Kp_x, Kp_y, Kp_z])

    Kv_x = Kv[finger_id * 3 + 0]
    Kv_y = Kv[finger_id * 3 + 1]
    Kv_z = Kv[finger_id * 3 + 2]
    Kv = np.diag([Kv_x, Kv_y, Kv_z])

    # Compute current fingertip position
    x_current = custom_pinocchio_utils.forward_kinematics(q_current)[finger_id]

    delta_x = np.expand_dims(np.array(tip_pos_desired) - np.array(x_current), 1)
    # print("Current x: {}".format(x_current))
    # print("Desired x: {}".format(tip_desired))
    # print("Delta: {}".format(delta_x))

    # Get full Jacobian for finger
    Ji = custom_pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
    # Just take first 3 rows, which correspond to linear velocities of fingertip
    Ji = Ji[:3, :]

    # Get g matrix for gravity compensation
    _, g = custom_pinocchio_utils.get_lambda_and_g_matrix(
        finger_id, q_current, Ji, grav
    )

    # Get current fingertip velocity
    dx_current = Ji @ np.expand_dims(np.array(dq_current), 1)

    delta_dx = np.expand_dims(np.array(tip_vel_desired), 1) - np.array(dx_current)

    if tip_force_wf is not None:
        torque = (
            np.squeeze(Ji.T @ (Kp @ delta_x + Kv @ delta_dx) + Ji.T @ tip_force_wf) + g
        )
    else:
        torque = np.squeeze(Ji.T @ (Kp @ delta_x + Kv @ delta_dx)) + g

    # print("Finger {} delta".format(finger_id))
    # print(np.linalg.norm(delta_x))
    return torque


"""
Compute contact point position in world frame
Inputs:
cp_param: Contact point param [px, py, pz]
cube: Block object, which contains object shape info
"""


def get_cp_pos_wf_from_cp_param(
    cp_param, cube_pos_wf, cube_quat_wf, obj_half_size=OBJ_HALF_SIZE
):
    cp = get_cp_of_from_cp_param(cp_param, obj_half_size)

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(cp.pos_of) + translation


"""
Get contact point positions in world frame from cp_params
"""


def get_cp_pos_wf_from_cp_params(
    cp_params, cube_pos, cube_quat, obj_half_size=OBJ_HALF_SIZE, **kwargs
):
    # Get contact points in wf
    fingertip_goal_list = []
    for i in range(len(cp_params)):
        # for i in range(cp_params.shape[0]):
        fingertip_goal_list.append(
            get_cp_pos_wf_from_cp_param(
                cp_params[i], cube_pos, cube_quat, obj_half_size
            )
        )
    return fingertip_goal_list


"""
Compute contact point position in object frame
Inputs:
cp_param: Contact point param [px, py, pz]
"""


def get_cp_of_from_cp_param(cp_param, obj_half_size=OBJ_HALF_SIZE):
    if isinstance(obj_half_size, float):
        obj_shape = (obj_half_size, obj_half_size, obj_half_size)
    else:
        obj_shape = obj_half_size
    cp_of = []
    # Get cp position in OF
    for i in range(3):
        cp_of.append(-obj_shape[i] + (cp_param[i] + 1) * obj_shape[i])

    cp_of = np.asarray(cp_of)

    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        quat = (0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2)
    elif y_param == 1:
        quat = (0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2)
    elif x_param == 1:
        quat = (0, 0, 1, 0)
    elif z_param == 1:
        quat = (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2)
    elif x_param == -1:
        quat = (0, 0, 0, 1)
    elif z_param == -1:
        quat = (0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2)

    cp = ContactPoint(cp_of, quat)
    return cp


"""
Get face id on cube, given cp_param
cp_param: [x,y,z]
"""


def get_face_from_cp_param(cp_param):
    x_param = cp_param[0]
    y_param = cp_param[1]
    z_param = cp_param[2]
    # For now, just hard code quat
    if y_param == -1:
        face = 1
    elif y_param == 1:
        face = 2
    elif x_param == 1:
        face = 3
    elif z_param == 1:
        face = 4
    elif x_param == -1:
        face = 5
    elif z_param == -1:
        face = 6

    return face


"""
Trasform point p from world frame to object frame, given object pose
"""


def get_wf_from_of(p, obj_pose):
    cube_pos_wf = obj_pose.position
    cube_quat_wf = obj_pose.orientation

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    return rotation.apply(p) + translation


"""
Trasform point p from object frame to world frame, given object pose
"""


def get_of_from_wf(p, obj_pose):
    cube_pos_wf = obj_pose.position
    cube_quat_wf = obj_pose.orientation

    rotation = Rotation.from_quat(cube_quat_wf)
    translation = np.asarray(cube_pos_wf)

    rotation_inv = rotation.inv()
    translation_inv = -rotation_inv.apply(translation)

    return rotation_inv.apply(p) + translation_inv


##############################################################################
# Lift mode functions
##############################################################################
"""
Run trajectory optimization
current_position: current joint positions of robot
x0: object initial position for traj opt
x_goal: object goal position for traj opt
nGrid: number of grid points
dt: delta t
"""


def run_fixed_cp_traj_opt(
    cp_params,
    current_position,
    custom_pinocchio_utils,
    x0,
    x_goal,
    nGrid,
    dt,
    npz_filepath=None,
):

    cp_params_on_obj = []
    for cp in cp_params:
        if cp is not None:
            cp_params_on_obj.append(cp)
    fnum = len(cp_params_on_obj)

    # Formulate and solve optimization problem
    opt_problem = FixedContactPointOpt(
        nGrid=nGrid,  # Number of timesteps
        dt=dt,  # Length of each timestep (seconds)
        fnum=fnum,
        cp_params=cp_params_on_obj,
        x0=x0,
        x_goal=x_goal,
        obj_shape=OBJ_SIZE,
        obj_mass=OBJ_MASS,
        npz_filepath=npz_filepath,
    )

    x_soln = np.array(opt_problem.x_soln)
    dx_soln = np.array(opt_problem.dx_soln)
    l_wf_soln = np.array(opt_problem.l_wf_soln)

    return x_soln, dx_soln, l_wf_soln


"""
Get initial contact points on cube
Assign closest cube face to each finger
Since we are lifting object, don't worry about wf z-axis, just care about wf xy-plane
"""


def get_lifting_cp_params(obj_pose):
    # face that is touching the ground
    ground_face = get_closest_ground_face(obj_pose)

    # Transform finger base positions to object frame
    base_pos_list_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, obj_pose)
        base_pos_list_of.append(f_of)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)

    xy_distances = np.zeros(
        (3, 2)
    )  # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(base_pos_list_of):
        point_in_plane = np.array(
            [f_of[0, x_ind], f_of[0, y_ind]]
        )  # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0, 0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0, 0]), point_in_plane)

        xy_distances[f_i, 0] = x_dist
        xy_distances[f_i, 1] = y_dist

    # Do the face assignment - greedy approach (assigned closest fingers first)
    free_faces = OBJ_FACES_INFO[ground_face][
        "adjacent_faces"
    ].copy()  # List of face ids that haven't been assigned yet
    assigned_faces = np.zeros(3)
    for i in range(3):
        # Find indices max element in array
        max_ind = np.unravel_index(np.argmax(xy_distances), xy_distances.shape)
        curr_finger_id = max_ind[0]
        furthest_axis = max_ind[1]

        # print("current finger {}".format(curr_finger_id))
        # Do the assignment
        x_dist = xy_distances[curr_finger_id, 0]
        y_dist = xy_distances[curr_finger_id, 1]
        if furthest_axis == 0:  # distance to x axis is greater than to y axis
            if base_pos_list_of[curr_finger_id][0, y_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
        else:
            if base_pos_list_of[curr_finger_id][0, x_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5
        # print("first choice face: {}".format(face))

        # Handle faces that may already be assigned
        if face not in free_faces:
            alternate_axis = abs(furthest_axis - 1)
            if alternate_axis == 0:
                if base_pos_list_of[curr_finger_id][0, y_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
            else:
                if base_pos_list_of[curr_finger_id][0, x_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5
            # print("second choice face: {}".format(face))

        # If backup face isn't free, assign random face from free_faces
        if face not in free_faces:
            # print("random")
            # print(xy_distances[curr_finger_id, :])
            face = free_faces[0]
        assigned_faces[curr_finger_id] = face

        # Replace row with -np.inf so we can assign other fingers
        xy_distances[curr_finger_id, :] = -np.inf
        # Remove face from free_faces
        free_faces.remove(face)
    # print(assigned_faces)
    # Set contact point params
    cp_params = []
    for i in range(3):
        face = assigned_faces[i]
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        # print(i)
        # print(param)
        cp_params.append(param)
    # print("assigning cp params for lifting")
    # print(cp_params)
    return cp_params


"""
For a specified finger f_i and list of available faces, get closest face
"""


def assign_faces_to_fingers(obj_pose, finger_id_list, free_faces):

    ground_face = get_closest_ground_face(obj_pose)

    # Find distance from x axis and y axis, and store in xy_distances
    # Need some additional logic to prevent multiple fingers from being assigned to same face
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(ground_face)

    # Transform finger base positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, obj_pose)
        finger_base_of.append(f_of)

    xy_distances = np.zeros((3, 2))  # Rows: fingers, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array(
            [f_of[0, x_ind], f_of[0, y_ind]]
        )  # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0, 0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0, 0]), point_in_plane)

        xy_distances[f_i, 0] = x_dist
        xy_distances[f_i, 1] = y_dist

    assignments = {}
    for i in range(3):
        max_ind = np.unravel_index(np.nanargmax(xy_distances), xy_distances.shape)
        f_i = max_ind[0]
        if f_i not in finger_id_list:
            xy_distances[f_i, :] = np.nan
            continue
        furthest_axis = max_ind[1]
        x_dist = xy_distances[f_i, 0]
        y_dist = xy_distances[f_i, 1]
        if furthest_axis == 0:  # distance to x axis is greater than to y axis
            if finger_base_of[f_i][0, y_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
        else:
            if finger_base_of[f_i][0, x_ind] > 0:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
            else:
                face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5

        # Get alternate closest face
        if face not in free_faces:
            alternate_axis = abs(furthest_axis - 1)
            if alternate_axis == 0:
                if finger_base_of[f_i][0, y_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][1]  # 2
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][0]  # 1
            else:
                if finger_base_of[f_i][0, x_ind] > 0:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][2]  # 3
                else:
                    face = OBJ_FACES_INFO[ground_face]["adjacent_faces"][3]  # 5

        # Assign first face in free_faces
        if face not in free_faces:
            face = free_faces[0]

        assignments[f_i] = face

        xy_distances[f_i, :] = np.nan

        if face in free_faces:
            free_faces.remove(face)

    return assignments


def get_pre_grasp_ft_goal(obj_pose, fingertips_current_wf, cp_params):
    ft_goal = np.zeros(9)
    incr = 0.04

    # Get list of desired fingertip positions
    cp_wf_list = get_cp_pos_wf_from_cp_params(
        cp_params, obj_pose.position, obj_pose.orientation
    )

    for f_i in range(3):
        f_wf = cp_wf_list[f_i]
        if cp_params[f_i] is None:
            f_new_wf = fingertips_current_wf[f_i]
        else:
            # Get face that finger is on
            face = get_face_from_cp_param(cp_params[f_i])
            f_of = get_of_from_wf(f_wf, obj_pose)

            # Release object
            f_of = f_of - incr * OBJ_FACES_INFO[face]["up_axis"]

            # Convert back to wf
            f_new_wf = get_wf_from_of(f_of, obj_pose)

        ft_goal[3 * f_i : 3 * f_i + 3] = f_new_wf
    return ft_goal


"""
Set up traj opt for fingers and static object
"""


def define_static_object_opt(nGrid, dt):
    cube_shape = (move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH)
    problem = StaticObjectOpt(
        nGrid=nGrid,
        dt=dt,
        obj_shape=cube_shape,
    )
    return problem


"""
Solve traj opt to get finger waypoints
"""


def get_finger_waypoints(nlp, ft_goal, q_cur, obj_pose, npz_filepath=None):
    nlp.solve_nlp(ft_goal, q_cur, obj_pose=obj_pose, npz_filepath=npz_filepath)
    ft_pos = nlp.ft_pos_soln
    ft_vel = nlp.ft_vel_soln
    return ft_pos, ft_vel


"""
Get waypoints to initial contact point on object
For now, we assume that contact points are always in the center of cube face
Return waypoints in world frame
Inputs:
cp_param: target contact point param
fingertip_pos: fingertip start position in world frame
"""


def get_waypoints_to_cp_param(
    obj_pose, fingertip_pos, cp_param, obj_half_size=OBJ_HALF_SIZE
):
    # Get ground face
    ground_face = get_closest_ground_face(obj_pose)
    # Transform finger tip positions to object frame
    fingertip_pos_of = np.squeeze(get_of_from_wf(fingertip_pos, obj_pose))

    waypoints = []
    if cp_param is not None:
        # Transform cp_param to object frame
        cp = get_cp_of_from_cp_param(cp_param, obj_half_size=obj_half_size)
        cp_pos_of = cp.pos_of

        tol = 0.05

        # Get the non-zero cp_param dimension (to determine which face the contact point is on)
        # This works because we assume z is always 0, and either x or y is 0
        non_zero_dim = np.argmax(abs(cp_param))
        zero_dim = abs(1 - non_zero_dim)

        # Work with absolute values, and then correct sign at the end
        w = np.expand_dims(fingertip_pos_of, 0)
        w[0, :] = (
            0.07 * OBJ_FACES_INFO[ground_face]["up_axis"]
        )  # Bring fingers lower, to avoid links colliding with each other
        if abs(fingertip_pos_of[non_zero_dim]) < abs(cp_pos_of[non_zero_dim]) + tol:
            w[0, non_zero_dim] = cp_param[non_zero_dim] * (
                abs(cp_pos_of[non_zero_dim]) + tol
            )  # fix sign
        if abs(fingertip_pos_of[zero_dim]) < abs(cp_pos_of[zero_dim]) + tol:
            w[0, zero_dim] = cp_param[zero_dim] * (
                abs(cp_pos_of[zero_dim]) + tol
            )  # fix sign
        # print(w)
        waypoints.append(w.copy())

        # Align zero_dim
        w[0, zero_dim] = 0
        waypoints.append(w.copy())

        # w[0,non_zero_dim] = cp_pos_of[non_zero_dim]
        # w[0,2] = 0
        # waypoints.append(w.copy())
        waypoints.append(cp_pos_of)
    else:
        w = np.expand_dims(fingertip_pos_of, 0)
        waypoints.append(w.copy())
        waypoints.append(w.copy())
        waypoints.append(w.copy())

    # Transform waypoints from object frame to world frame
    waypoints_wf = []
    # waypoints_wf.append(fingertip_pos)
    for wp in waypoints:
        wp_wf = np.squeeze(get_wf_from_of(wp, obj_pose))
        # If world frame z coord in less than 0, clip this to 0.01
        if wp_wf[2] <= 0:
            wp_wf[2] = 0.01
        waypoints_wf.append(wp_wf)

    # return waypoints_wf
    # Add intermediate waypoints
    interp_num = 10
    waypoints_final = []
    for i in range(len(waypoints_wf) - 1):
        curr_w = waypoints_wf[i]
        next_w = waypoints_wf[i + 1]

        interp_pts = np.linspace(curr_w, next_w, interp_num)
        for r in range(interp_num):
            waypoints_final.append(interp_pts[r])

    # waypoints_final.pop(-1)

    return waypoints_final


##############################################################################
# Lift mode functions
##############################################################################

"""
Determine face that is closest to ground
"""


def get_closest_ground_face(obj_pose):
    min_z = np.inf
    min_face = None
    for i in range(1, 7):
        c = OBJ_FACES_INFO[i]["center_param"].copy()
        c_wf = get_wf_from_of(c, obj_pose)
        if c_wf[2] < min_z:
            min_z = c_wf[2]
            min_face = i

    return min_face


"""
Get flipping contact points
"""


def get_flipping_cp_params(
    init_pose,
    goal_pose,
    obj_half_size=OBJ_HALF_SIZE,
):
    # Get goal face
    init_face = get_closest_ground_face(init_pose)
    # print("Init face: {}".format(init_face))
    # Get goal face
    goal_face = get_closest_ground_face(goal_pose)
    # print("Goal face: {}".format(goal_face))

    if goal_face not in OBJ_FACES_INFO[init_face]["adjacent_faces"]:
        # print("Goal face not adjacent to initial face")
        goal_face = OBJ_FACES_INFO[init_face]["adjacent_faces"][0]
        # print("Intermmediate goal face: {}".format(goal_face))

    # Common adjacent faces to init_face and goal_face
    common_adjacent_faces = list(
        set(OBJ_FACES_INFO[init_face]["adjacent_faces"]).intersection(
            OBJ_FACES_INFO[goal_face]["adjacent_faces"]
        )
    )

    opposite_goal_face = OBJ_FACES_INFO[goal_face]["opposite_face"]

    # print("place fingers on faces {}, towards face {}".format(common_adjacent_faces, opposite_goal_face))

    # Find closest fingers to each of the common_adjacent_faces
    # Transform finger tip positions to object frame
    finger_base_of = []
    for f_wf in FINGER_BASE_POSITIONS:
        f_of = get_of_from_wf(f_wf, init_pose)
        # f_of = np.squeeze(get_of_from_wf(f_wf, init_pose))
        finger_base_of.append(f_of)

    # Object frame axis corresponding to plane parallel to ground plane
    x_ind, y_ind = __get_parallel_ground_plane_xy(init_face)
    # Find distance from x axis and y axis, and store in xy_distances
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    xy_distances = np.zeros(
        (3, 2)
    )  # Row corresponds to a finger, columns are x and y axis distances
    for f_i, f_of in enumerate(finger_base_of):
        point_in_plane = np.array(
            [f_of[0, x_ind], f_of[0, y_ind]]
        )  # Ignore dimension of point that's not in the plane
        x_dist = __get_distance_from_pt_2_line(x_axis, np.array([0, 0]), point_in_plane)
        y_dist = __get_distance_from_pt_2_line(y_axis, np.array([0, 0]), point_in_plane)

        xy_distances[f_i, 0] = np.sign(f_of[0, y_ind]) * x_dist
        xy_distances[f_i, 1] = np.sign(f_of[0, x_ind]) * y_dist

    finger_assignments = {}
    for face in common_adjacent_faces:
        face_ind = OBJ_FACES_INFO[init_face]["adjacent_faces"].index(face)
        if face_ind in [2, 3]:
            # Check y_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][x_ind] < 0:
                # Want most negative value
                f_i = np.nanargmin(xy_distances[:, 1])
            else:
                # Want most positive value
                f_i = np.nanargmax(xy_distances[:, 1])
        else:
            # Check x_ind column for finger that is furthest away
            if OBJ_FACES_INFO[face]["center_param"][y_ind] < 0:
                f_i = np.nanargmin(xy_distances[:, 0])
            else:
                f_i = np.nanargmax(xy_distances[:, 0])
        finger_assignments[face] = f_i
        xy_distances[f_i, :] = np.nan

    cp_params = [None, None, None]
    full_size = obj_half_size * 2
    if not isinstance(full_size, float):
        height_param = -full_size[2]
        width_param = full_size[1]
    else:
        full_size
        height_param = -0.65  # Always want cps to be at this height
        width_param = 0.65
    for face in common_adjacent_faces:
        param = OBJ_FACES_INFO[face]["center_param"].copy()
        param += (
            OBJ_FACES_INFO[OBJ_FACES_INFO[init_face]["opposite_face"]]["center_param"]
            * height_param
        )
        param += OBJ_FACES_INFO[opposite_goal_face]["center_param"] * width_param
        cp_params[finger_assignments[face]] = param
        # cp_params.append(param)
    # print("Assignments: {}".format(finger_assignments))
    return cp_params, init_face, goal_face


"""
Get next waypoint for flipping
"""


def get_flipping_waypoint(
    obj_pose,
    init_face,
    goal_face,
    fingertips_current_wf,
    fingertips_init_wf,
    cp_params,
):
    # Get goal face
    # goal_face = get_closest_ground_face(goal_pose)
    # print("Goal face: {}".format(goal_face))
    # print("ground face: {}".format(get_closest_ground_face(obj_pose)))

    ground_face = get_closest_ground_face(obj_pose)
    # if (get_closest_ground_face(obj_pose) == goal_face):
    #  # Move fingers away from object
    #  return fingertips_init_wf

    # Transform current fingertip positions to of
    fingertips_new_wf = []

    incr = 0.01
    for f_i in range(3):
        f_wf = fingertips_current_wf[f_i]
        if cp_params[f_i] is None:
            f_new_wf = fingertips_init_wf[f_i]
        else:
            # Get face that finger is one
            face = get_face_from_cp_param(cp_params[f_i])
            f_of = get_of_from_wf(f_wf, obj_pose)

            if ground_face == goal_face:
                # Release object
                f_new_of = f_of - 0.01 * OBJ_FACES_INFO[face]["up_axis"]
                if obj_pose.position[2] <= 0.034:  # TODO: HARDCODED
                    flip_done = True
                    # print("FLIP SUCCESSFUL!")
                else:
                    flip_done = False
            elif ground_face != init_face:
                # Ground face does not match goal force or init face, give up
                f_new_of = f_of - 0.01 * OBJ_FACES_INFO[face]["up_axis"]
                if obj_pose.position[2] <= 0.034:  # TODO: HARDCODED
                    flip_done = True
                else:
                    flip_done = False
            else:
                # Increment up_axis of f_of
                f_new_of = f_of + incr * OBJ_FACES_INFO[ground_face]["up_axis"]
                flip_done = False

            # Convert back to wf
            f_new_wf = get_wf_from_of(f_new_of, obj_pose)

        fingertips_new_wf.append(f_new_wf)

    # print(fingertips_current_wf)
    # print(fingertips_new_wf)
    # fingertips_new_wf[2] = fingertips_init_wf[2]

    return fingertips_new_wf, flip_done


##############################################################################
# Flip mode functions
##############################################################################

"""
Determine face that is closest to ground
"""


def get_closest_ground_face(obj_pose):
    min_z = np.inf
    min_face = None
    for i in range(1, 7):
        c = OBJ_FACES_INFO[i]["center_param"].copy()
        c_wf = get_wf_from_of(c, obj_pose)
        if c_wf[2] < min_z:
            min_z = c_wf[2]
            min_face = i

    return min_face


##############################################################################
# Private functions
##############################################################################

"""
Given a ground face id, get the axes that are parallel to the floor
"""


def __get_parallel_ground_plane_xy(ground_face):
    if ground_face in [1, 2]:
        x_ind = 0
        y_ind = 2
    if ground_face in [3, 5]:
        x_ind = 2
        y_ind = 1
    if ground_face in [4, 6]:
        x_ind = 0
        y_ind = 1
    return x_ind, y_ind


"""
Get distance from point to line (in 2D)
Inputs:
a, b: points on line
p: standalone point, for which we want to compute its distance to line
"""


def __get_distance_from_pt_2_line(a, b, p):
    a = np.squeeze(a)
    b = np.squeeze(b)
    p = np.squeeze(p)

    ba = b - a
    ap = a - p
    c = ba * (np.dot(ap, ba) / np.dot(ba, ba))
    d = ap - c

    return np.sqrt(np.dot(d, d))


"""
Get grasp matrix
Input:
x: object pose [px, py, pz, qw, qx, qy, qz]
"""


def __get_grasp_matrix(x, cp_list):
    fnum = len(cp_list)
    obj_dof = 6

    # Contact model force selection matrix
    l_i = 3
    H_i = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ]
    )
    H = np.zeros((l_i * fnum, obj_dof * fnum))
    for i in range(fnum):
        H[i * l_i : i * l_i + l_i, i * obj_dof : i * obj_dof + obj_dof] = H_i

    # Transformation matrix from object frame to world frame
    quat_o_2_w = [x[3], x[4], x[5], x[6]]

    G_list = []

    # Calculate G_i (grasp matrix for each finger)
    for c in cp_list:
        cp_pos_of = c.pos_of  # Position of contact point in object frame
        quat_cp_2_o = (
            c.quat_of
        )  # Orientation of contact point frame w.r.t. object frame

        S = np.array(
            [
                [0, -cp_pos_of[2], cp_pos_of[1]],
                [cp_pos_of[2], 0, -cp_pos_of[0]],
                [-cp_pos_of[1], cp_pos_of[0], 0],
            ]
        )

        P_i = np.eye(6)
        P_i[3:6, 0:3] = S

        # Orientation of cp frame w.r.t. world frame
        # quat_cp_2_w = quat_o_2_w * quat_cp_2_o
        R_cp_2_w = Rotation.from_quat(quat_o_2_w) * Rotation.from_quat(quat_cp_2_o)
        # R_i is rotation matrix from contact frame i to world frame
        R_i = R_cp_2_w.as_matrix()
        R_i_bar = np.zeros((6, 6))
        R_i_bar[0:3, 0:3] = R_i
        R_i_bar[3:6, 3:6] = R_i

        G_iT = R_i_bar.T @ P_i.T
        G_list.append(G_iT)

    GT_full = np.concatenate(G_list)
    GT = H @ GT_full
    # print(GT.T)
    return GT.T


"""
Get matrix to convert dquat (4x1 vector) to angular velocities (3x1 vector)
"""


def get_dquat_to_dtheta_matrix(quat):
    qx = quat[0]
    qy = quat[1]
    qz = quat[2]
    qw = quat[3]

    M = np.array(
        [
            [-qx, -qy, -qz],
            [qw, qz, -qy],
            [-qz, qw, qx],
            [qy, -qx, qw],
        ]
    )

    return M.T


def get_ft_R(q):
    R_list = []
    for f_i, angle in enumerate(BASE_ANGLE_DEGREES):
        theta = angle * (np.pi / 180)
        q1 = q[3 * f_i + 0]
        q2 = q[3 * f_i + 1]
        q3 = q[3 * f_i + 2]
        R = np.array(
            [
                [
                    np.cos(q1) * np.cos(theta),
                    (
                        np.sin(q1) * np.sin(q2) * np.cos(theta)
                        - np.sin(theta) * np.cos(q2)
                    )
                    * np.cos(q3)
                    + (
                        np.sin(q1) * np.cos(q2) * np.cos(theta)
                        + np.sin(q2) * np.sin(theta)
                    )
                    * np.sin(q3),
                    -(
                        np.sin(q1) * np.sin(q2) * np.cos(theta)
                        - np.sin(theta) * np.cos(q2)
                    )
                    * np.sin(q3)
                    + (
                        np.sin(q1) * np.cos(q2) * np.cos(theta)
                        + np.sin(q2) * np.sin(theta)
                    )
                    * np.cos(q3),
                ],
                [
                    np.sin(theta) * np.cos(q1),
                    (
                        np.sin(q1) * np.sin(q2) * np.sin(theta)
                        + np.cos(q2) * np.cos(theta)
                    )
                    * np.cos(q3)
                    + (
                        np.sin(q1) * np.sin(theta) * np.cos(q2)
                        - np.sin(q2) * np.cos(theta)
                    )
                    * np.sin(q3),
                    -(
                        np.sin(q1) * np.sin(q2) * np.sin(theta)
                        + np.cos(q2) * np.cos(theta)
                    )
                    * np.sin(q3)
                    + (
                        np.sin(q1) * np.sin(theta) * np.cos(q2)
                        - np.sin(q2) * np.cos(theta)
                    )
                    * np.cos(q3),
                ],
                [
                    -np.sin(q1),
                    np.sin(q2) * np.cos(q1) * np.cos(q3)
                    + np.sin(q3) * np.cos(q1) * np.cos(q2),
                    -np.sin(q2) * np.sin(q3) * np.cos(q1)
                    + np.cos(q1) * np.cos(q2) * np.cos(q3),
                ],
            ]
        )
        R_list.append(R)
    return R_list


"""
Get orientation that is parallel to ground, with specified ground face down
"""


def get_ground_aligned_orientation(obj_pose):
    z_axis = [0, 0, 1]
    ground_face = get_closest_ground_face(obj_pose)
    actual_rot = Rotation.from_quat(obj_pose.orientation)

    # print("GROUND FACE: {}".format(ground_face))

    pose_up_vector = actual_rot.apply(OBJ_FACES_INFO[ground_face]["up_axis"])
    # print(pose_up_vector)

    orientation_error = np.arccos(pose_up_vector.dot(z_axis))

    # Rotate by orientation error
    align_rot = Rotation.from_euler("y", orientation_error)
    new_rot = align_rot.inv() * actual_rot

    # Check new error, if larger, rotate the other way
    pose_up_vector = new_rot.apply(OBJ_FACES_INFO[ground_face]["up_axis"])
    new_orientation_error = np.arccos(pose_up_vector.dot(z_axis))

    if new_orientation_error > orientation_error:
        align_rot = Rotation.from_euler("y", -orientation_error)
        new_rot = align_rot.inv() * actual_rot

    return new_rot.as_quat()


def get_aligned_pose(obj_pose):
    # print("Observed pose:")
    # print(obj_pose.position, obj_pose.orientation)

    # Clip obj z coord to half width of cube
    clipped_pos = obj_pose.position.copy()
    clipped_pos[2] = 0.01  # TODO hardcoded
    aligned_quat = get_ground_aligned_orientation(obj_pose)

    obj_pose.position = clipped_pos
    obj_pose.orientation = aligned_quat

    # print("Aligned pose:")
    # print(obj_pose.position, obj_pose.orientation)

    return obj_pose


def get_y_axis_delta(obj_pose, goal_pose):
    ground_face = get_closest_ground_face(obj_pose)
    goal_rot = Rotation.from_quat(goal_pose.orientation)
    actual_rot = Rotation.from_quat(obj_pose.orientation)

    y_axis = [0, 1, 0]

    actual_direction_vector = actual_rot.apply(y_axis)

    goal_direction_vector = goal_rot.apply(y_axis)
    N = np.array([0, 0, 1])  # normal vector of ground plane
    proj = goal_direction_vector - goal_direction_vector.dot(N) * N
    goal_direction_vector = proj / np.linalg.norm(proj)  # normalize projection

    # Always in [0, pi] range
    orientation_error = np.arccos(goal_direction_vector.dot(actual_direction_vector))

    # Determine direction of rotation
    if ground_face in [5]:  # TODO ???
        direction = -1
    else:
        direction = 1
    rot = Rotation.from_euler("z", direction * orientation_error)
    new_rot = rot * actual_rot

    # Check new error, if larger, rotate the other way
    new_direction_vector = new_rot.apply(y_axis)
    new_orientation_error = np.arccos(goal_direction_vector.dot(new_direction_vector))

    if new_orientation_error < orientation_error:
        return direction * orientation_error
    else:
        return -1 * direction * orientation_error
