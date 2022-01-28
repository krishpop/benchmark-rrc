"""Demo on how to run the robot using the Gym environment

This demo creates a RealRobotCubeEnv environment and runs one episode using a
dummy policy which uses random actions.
"""
import json
import os
import os.path as osp

import numpy as np
from rrc.env import cube_env, initializers, termination_fns
from rrc_iprl_package.control.control_policy import (
    CuboidImpedanceControllerPolicy,
    ImpedanceControllerPolicy,
)
from trifinger_simulation.tasks import move_cube

FRAMESKIP = 1
MAX_STEPS = 120 * 1000
EP_LEN = None


class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main(difficulty, goal_pose_json, object_shape):
    # the difficulty level and the goal pose (as JSON string) are passed as
    # arguments
    if os.path.exists(goal_pose_json):
        with open(goal_pose_json) as f:
            goal = json.load(f)["goal"]
    else:
        goal = json.loads(goal_pose_json)
    goal = {k: np.asarray(goal[k]) for k in goal}
    initial_pose = move_cube.sample_goal(-1)
    if args.object_shape == "cuboid":
        initial_pose.position[2] = move_cube._CUBOID_WIDTH / 2
    else:
        initial_pose.position[2] = move_cube._CUBE_WIDTH / 2
    theta = 0
    initial_pose.orientation = np.array([0, 0, np.sin(theta / 2), np.cos(theta / 2)])
    initializer = initializers.DumbInitializer(
        difficulty, goal, initial_pose.to_dict(), object_shape=object_shape
    )
    if difficulty == 4:
        term_fn = termination_fns.stay_close_to_goal_level_4
    else:
        term_fn = termination_fns.stay_close_to_goal

    ep_len = EP_LEN or MAX_STEPS
    path = "./output" if osp.exists("./output") else None
    env = cube_env.RobotCubeEnv(
        goal,
        difficulty,
        action_type=cube_env.ActionType.TORQUE,
        initializer=initializer,
        frameskip=FRAMESKIP,
        episode_length=ep_len,
        termination_fn=term_fn,
        sim=True,
        visualization=True,
        path=path,
        return_timestamp=True,
        object_shape=object_shape,
    )

    goal_pose = move_cube.Pose.from_dict(goal)
    if object_shape == "cuboid":
        policy = CuboidImpedanceControllerPolicy(
            action_space=env.action_space,
            initial_pose=initial_pose,
            goal_pose=goal_pose,
            debug_waypoints=False,
            difficulty=difficulty,
        )
    else:
        policy = ImpedanceControllerPolicy(
            env.action_space,
            initial_pose,
            goal_pose,
            difficulty=difficulty,
            save_path=path,
            ycb=(object_shape == "ycb"),
        )

    observation = env.reset()
    policy.set_init_goal(
        move_cube.Pose.from_dict(observation["achieved_goal"]),
        move_cube.Pose.from_dict(observation["desired_goal"]),
    )
    policy.reset_policy(observation, env.platform)

    accumulated_reward = 0
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

    env.save_custom_logs()
    # Save control_policy_log
    policy.save_log()

    print("------")
    print("Accumulated Reward: {:.3f}".format(accumulated_reward))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", "--d", default=2, type=int)
    parser.add_argument(
        "--cube_goal_json",
        "--g",
        default="../rrc_package/goals/goal20.json",
        type=str,
    )
    parser.add_argument("--object_shape", "--s", default="cube", type=str)
    args = parser.parse_args()

    main(args.difficulty, args.cube_goal_json, args.object_shape)
