"""Place reward functions here.

These will be passed as an arguement to the training env, allowing us to
easily try out new reward functions.
"""


import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation
from trifinger_simulation.tasks import move_cube
from trifinger_simulation.tasks.move_cube import _max_height, _min_height

###############################
# Competition Reward Functions
###############################


def competition_reward(previous_observation, observation, info):
    return -move_cube.evaluate_state(
        move_cube.Pose.from_dict(observation["desired_goal"]),
        move_cube.Pose.from_dict(observation["achieved_goal"]),
        info["difficulty"],
    )


# For backward compatibility
task1_competition_reward = competition_reward
task2_competition_reward = competition_reward
task3_competition_reward = competition_reward
task4_competition_reward = competition_reward

##############################
# Training Reward functions
##############################


def _lgsk_kernel(x, scale: float = 50.0):
    """Defines logistic kernel function to bound input to [-0.25, 0)
    Ref: https://arxiv.org/abs/1901.08652 (page 15)
    Args:
        x: distance.
        scale:  kernel function.
    Returns:
        Output tensor computed using kernel.
    """
    scaled = x * scale
    return 1.0 / (np.exp(scaled) + 2 + np.exp(-scaled))


def _tip_distance_to_cube(observation):
    # calculate first reward term
    pose = observation["achieved_goal"]
    return np.linalg.norm(
        observation["observation"]["tip_positions"] - pose["position"]
    )


def _action_reg(observation):
    v = observation["observation"]["velocity"]
    t = observation["observation"]["torque"]
    velocity_reg = v.dot(v)
    torque_reg = t.dot(t)
    return 0.1 * velocity_reg + torque_reg


def _tip_slippage(previous_observation, observation):
    pose = observation["achieved_goal"]
    prev_pose = previous_observation["achieved_goal"]
    obj_rot = Rotation.from_quat(pose["orientation"])
    prev_obj_rot = Rotation.from_quat(prev_pose["orientation"])
    relative_tip_pos = obj_rot.apply(
        observation["observation"]["tip_positions"]
        - observation["achieved_goal"]["position"]
    )
    prev_relative_tip_pos = prev_obj_rot.apply(
        previous_observation["observation"]["tip_positions"]
        - previous_observation["achieved_goal"]["position"]
    )
    return -np.linalg.norm(relative_tip_pos - prev_relative_tip_pos)


def _corner_error(observation):
    goal_pose = move_cube.Pose.from_dict(observation["desired_goal"])
    actual_pose = move_cube.Pose.from_dict(observation["achieved_goal"])
    goal_corners = move_cube.get_cube_corner_positions(goal_pose)
    actual_corners = move_cube.get_cube_corner_positions(actual_pose)
    orientation_errors = np.linalg.norm(goal_corners - actual_corners, axis=1)
    return orientation_errors


def training_reward(previous_observation, observation, info):
    shaping = _tip_distance_to_cube(previous_observation) - _tip_distance_to_cube(
        observation
    )
    r = competition_reward(previous_observation, observation, info)
    reg = _action_reg(observation)
    slippage = _tip_slippage(previous_observation, observation)
    return r - 0.1 * reg + 500 * shaping + 300 * slippage


def training_reward1(previous_observation, observation, info):
    r = competition_reward(previous_observation, observation, info)
    action = observation["observation"]["action"]
    ac_reg = 0.1 * np.linalg.norm(action)
    vel_reg = 0.01 * np.linalg.norm(observation["observation"]["velocity"])
    return r - ac_reg - vel_reg


def training_reward2(previous_observation, observation, info):
    import dm_control.utils.rewards as dmr

    ori_err = _orientation_error(observation)
    pos_err = _position_error(observation)
    pos_err_xy, pos_err_z = _xy_position_error(observation), _z_position_error(
        observation
    )
    xy_reward = dmr.tolerance(
        pos_err_xy, bounds=(0, 0.025), margin=0.075, sigmoid="long_tail"
    )
    z_reward = dmr.tolerance(
        pos_err_z, bounds=(0, 0.025), margin=0.075, sigmoid="long_tail"
    )
    r = xy_reward + z_reward
    if r > 1.0:
        r += dmr.tolerance(ori_err, bounds=(0, 0.025), margin=0.05)
    # r = pos_err <= 0.05
    # r += ori_err <= 0.05
    action = observation["observation"]["action"]
    # ac_reg = .05 * np.linalg.norm(action)
    # vel_reg = .05 * np.linalg.norm(observation['observation']['velocity'])
    return r  # - ac_reg - vel_reg


def training_reward3(previous_observation, observation, info):
    ori_shaping = _orientation_error(previous_observation) - _orientation_error(
        observation
    )
    return training_reward2(previous_observation, observation, info) + 50 * ori_shaping


def training_reward4(previous_observation, observation, info):
    pos_shaping = _position_error(previous_observation) - _position_error(observation)
    return training_reward3(previous_observation, observation, info) + 50 * pos_shaping


def training_reward5(previous_observation, observation, info):
    dist = _corner_error(observation)
    return sum([_lgsk_kernel(d) for d in dist])


def training_reward_s2r2(previous_observation, observation, info):
    finger_reach_object_weight = -250  # info["finger_reach_object_weight"]
    finger_movement_weight = -0.5  # info["finger_movement_weight"]
    object_dist_weight = 2000  # info["object_dist_weight"]
    dt = 0.004
    fingertip_vel = (
        observation["observation"]["tip_positions"]
        - previous_observation["observation"]["tip_positions"]
    ) / dt
    finger_movement_penalty = finger_movement_weight * np.square(fingertip_vel).sum()
    curr_norms = _tip_distance_to_cube(observation)
    prev_norms = _tip_distance_to_cube(previous_observation)
    if True:  # info["steps"] < 1e6:
        finger_reach_object_reward = finger_reach_object_weight * (
            curr_norms - prev_norms
        )
    else:
        finger_reach_object_reward = 0.0
    keypoints_kernel_sum = training_reward5(previous_observation, observation, info)
    pose_reward = object_dist_weight * dt * keypoints_kernel_sum
    return finger_reach_object_reward + finger_movement_penalty + pose_reward


def corner_shaped_reward(previous_observation, observation, info):
    corner_rew = training_reward5(previous_observation, observation, info)
    _tip_dist = _tip_distance_to_cube(observation)
    tip_dist = stats.norm.pdf(_tip_dist * (1 / 0.08))
    return tip_dist + corner_rew


train = training_reward
train1 = training_reward1
train2 = training_reward2
train3 = training_reward3
train4 = training_reward4
train5 = training_reward5
train6 = training_reward_s2r2
corner_shaped = corner_shaped_reward
competition = competition_reward


def gaussian_reward(previous_observation, observation, info):
    r = competition_reward(previous_observation, observation, info)
    return stats.norm.pdf(7 * r)


def gaussian_training_reward(previous_observation, observation, info):
    """gaussian reward with additional reward engineering"""
    r = gaussian_reward(previous_observation, observation, info)

    # Large tip forces are around 0.5. 0.05 means no force is sensed at the tips
    tip_force = np.sum(observation["observation"]["tip_force"])

    # NOTE: _act_reg
    # smaller is better
    # a rough rule of thumb: 1.1 or above means a 'large' action
    _act_reg = _action_reg(observation)
    act_reg = stats.norm.pdf(_act_reg * (1 / 1.0))

    # NOTE: _slippage
    # smaller is better
    # a rough rule of thumb: 0.0018 or above means slip
    _slippage = -1 * _tip_slippage(previous_observation, observation)
    slippage = stats.norm.pdf(_slippage * (1 / 0.0018))

    # NOTE: _tip_dist
    # smaller is better
    # a rough rule of thumb: 0.07 ~ 0.08  while the obj is stably grasped
    _tip_dist = _tip_distance_to_cube(observation)
    tip_dist = stats.norm.pdf(_tip_dist * (1 / 0.08))

    reward = r + 0.04 * (act_reg + slippage + tip_dist + tip_force)
    # print('==== reward ====')
    # print(f'comp: {r}')
    # print(f'act-reg original: {_act_reg}')
    # print(f'act-reg: {act_reg}')
    # print(f'tip-dist original: {_tip_dist}')
    # print(f'tip-dist: {tip_dist}')
    # print(f'slip original: {_slippage}')
    # print(f'slip: {slippage}')
    # print(f'tip_force: {tip_force}')
    # print('shaping', reward - r)

    return reward


def _orientation_error(observation):
    goal_rot = Rotation.from_quat(observation["desired_goal"]["orientation"])
    actual_rot = Rotation.from_quat(observation["achieved_goal"]["orientation"])
    error_rot = goal_rot.inv() * actual_rot
    return error_rot.magnitude() / np.pi


def _xy_position_error(observation):
    range_xy_dist = 0.195 * 2
    xy_dist = np.linalg.norm(
        observation["desired_goal"]["position"][:2]
        - observation["achieved_goal"]["position"][:2]
    )
    return xy_dist / range_xy_dist


def _z_position_error(observation):
    range_z_dist = _max_height

    z_dist = abs(
        observation["desired_goal"]["position"][2]
        - observation["achieved_goal"]["position"][2]
    )
    return z_dist / range_z_dist


def _position_error(observation):
    pos_err = (_xy_position_error(observation) + _z_position_error(observation)) / 2
    return pos_err


def match_orientation_reward(previous_observation, observation, info):
    shaping = _tip_distance_to_cube(previous_observation) - _tip_distance_to_cube(
        observation
    )
    return -_orientation_error(observation) + 500 * shaping


def match_orientation_reward_shaped(previous_observation, observation, info):
    shaping = _tip_distance_to_cube(previous_observation) - _tip_distance_to_cube(
        observation
    )
    ori_shaping = _orientation_error(previous_observation) - _orientation_error(
        observation
    )
    return 500 * shaping + 100 * ori_shaping
