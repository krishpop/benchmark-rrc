import os
import os.path as osp
import json

from trifinger_simulation.tasks import move_cube
from rrc.env import make_env, cube_env, initializers, termination_fns
from rrc_iprl_package.control.control_policy import (
    CuboidImpedanceControllerPolicy,
    ImpedanceControllerPolicy,
)
import numpy as np

from rrc_iprl_package.control.controller_utils_cube import OBJ_HALF_SIZE

FRAMESKIP = 1
MAX_STEPS = 120 * 1000 / 4
EP_LEN = None


def main(difficulty, goal_pose_json, object_shape, path=None):
    with open(goal_pose_json, "r") as f:
        goal_pose = move_cube.goal_from_json(f.read())
    object_shape = "cube"
    ep_len = EP_LEN or MAX_STEPS
    if difficulty == 4:
        term_fn = termination_fns.stay_close_to_goal_level_4
    else:
        term_fn = termination_fns.stay_close_to_goal

    if osp.exists(path):
        i = 0
        while osp.exists(path + str(i)):
            i += 1
        path = path + str(i)
    os.makedirs(path)

    env = cube_env.RobotWrenchCubeEnv(
        goal_pose.to_dict(),
        difficulty,
        initializer=None,
        frameskip=FRAMESKIP,
        episode_length=ep_len,
        termination_fn=term_fn,
        visualization=False,
        path=path,
        return_timestamp=True,
        object_shape=object_shape,
        use_actual_cp=True,
        use_impedance=True,
        use_traj_opt=False,
    )
    observation = env.reset()
    initial_pose = move_cube.Pose.from_dict(observation["achieved_goal"])
    goal = goal_pose.to_dict()
    goal["position"] = goal["position"] - np.array([0, 0, OBJ_HALF_SIZE])
    move_cube.Pose.from_dict(observation["achieved_goal"]),
    move_cube.Pose.from_dict(observation["desired_goal"]),
    policy = ImpedanceControllerPolicy(
        env.action_space,
        initial_pose,
        goal,
        difficulty=difficulty,
        save_path=path,
        ycb=(object_shape == "ycb"),
    )
    policy.set_init_goal(
        move_cube.Pose.from_dict(observation["achieved_goal"]),
        move_cube.Pose.from_dict(observation["desired_goal"]),
    )
    # policy.reset_policy(observation, env.platform)

    accumulated_reward = env.ep_reward
    is_done = False
    old_mode = policy.mode
    steps_so_far = 0
    try:
        while not is_done:
            if MAX_STEPS is not None and steps_so_far == MAX_STEPS:
                break
            action = policy.predict(observation)
            observation, reward, is_done, info = env.step(action)
            if old_mode != policy.mode:
                print("mode changed: {} to {}".format(old_mode, policy.mode))
                old_mode = policy.mode
            # print("reward:", reward)
            accumulated_reward += reward
            steps_so_far += env.frameskip
    except Exception as e:
        print("Error encounted: {}. Saving logs and exiting".format(e))
        env.save_custom_logs()
        policy.save_log()
        raise e

    print("Acc reward:", accumulated_reward)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cube_goal_json",
        "--g",
        default="./goals/goal3-2.json",
        type=str,
    )
    parser.add_argument("--logdir", default="./goal-log", type=str)
    parser.add_argument("--object_shape", "--s", default="cube", type=str)
    args = parser.parse_args()
    difficulty = int(osp.split(args.cube_goal_json)[1][4])

    main(difficulty, args.cube_goal_json, args.object_shape)
