#!/usr/bin/env python3
"""Run a single episode with a controller in simulation."""
import argparse
import numpy as np
import os
import os.path as osp

from rrc.combined_code import create_state_machine
from rrc.env.make_env import make_env
from rrc.mp.utils import set_seed
from trifinger_simulation.tasks import move_cube


def _init_env(
    goal_pose_dict,
    difficulty,
    episode_length,
    method,
    path,
    object_shape,
    visualize=False,
):
    eval_config = {
        "action_space": "torque_and_position",
        "frameskip": 3,
        "reward_fn": "competition_reward",
        "termination_fn": "no_termination",
        "initializer": "centered_init",
        "monitor": False,
        "visualization": visualize,
        "object_shape": object_shape,
        "sim": True,
        "path": path,
        "rank": 0,
        "episode_length": int(episode_length),
        "real": True,
    }

    set_seed(0)
    env = make_env(goal_pose_dict, difficulty, **eval_config)
    return env


def main():
    parser = argparse.ArgumentParser("args")
    parser.add_argument(
        "method",
        type=str,
        help="The method to run. One of 'mp-pg', 'cic-cg', 'cpc-tg'",
        default="mp-pg",
    )
    parser.add_argument(
        "object", default="cube", help="object to use. One of 'cube', 'cuboid', 'ycb'"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true",
        help="add to use residual policies. Only compatible with difficulties 3 and 4.",
    )
    parser.add_argument(
        "--bo",
        default=False,
        action="store_true",
        help="add to use BO optimized parameters.",
    )
    parser.add_argument("--goal_list", type=str)
    parser.add_argument("--goal_json", default="./goals/goal1-0.json")
    parser.add_argument("--episode_length", "--ep_len", default=2000)
    parser.add_argument("--v", "--visualize", action="store_true")
    args = parser.parse_args()
    if args.goal_list and osp.exists(args.goal_list):
        goal_list = [osp.join(args.goal_list, g) for g in os.listdir(args.goal_list)]
    else:
        goal_list = [args.goal_json]

    with open(goal_list[0], "r") as f:
        goal_pose = move_cube.goal_from_json(f.read())
    goal_pose_dict = {
        "position": goal_pose.position.tolist(),
        "orientation": goal_pose.orientation.tolist(),
    }
    goal_id = osp.split(args.goal_json)[1].split(".")[0]
    path = osp.join("output", args.method, goal_id)
    if not osp.exists(path):
        os.makedirs(path)
    difficulty = int(osp.split(args.goal_json)[1][4])
    env = _init_env(
        goal_pose_dict,
        difficulty,
        args.episode_length,
        args.method,
        path=path,
        object_shape=args.object,
        visualize=args.v,
    )
    for goal in goal_list:
        with open(goal, "r") as f:
            goal_pose = move_cube.goal_from_json(f.read())
        goal_pose_dict = goal_pose.to_dict()
        goal_id = osp.split(goal)[1].split(".")[0]
        difficulty = int(osp.split(goal)[1][4])
        path = osp.join("output", args.method, goal_id)
        print("Saving to path:", path)
        if not osp.exists(path):
            os.makedirs(path)
        else:
            continue
        state_machine = create_state_machine(
            difficulty, args.method, env, args.residual, args.bo
        )
        env.goal = goal_pose_dict
        env.path = path
        env.difficulty = difficulty
        run_state_machine(env, state_machine)


def run_state_machine(env, state_machine, ep_len=5000):
    #####################
    # Run state machine
    #####################
    obs = env.reset()
    state_machine.reset()

    done = False
    rs = []
    step_count = 0
    # Ep Len 5000
    while not done and step_count < ep_len:
        action = state_machine(obs)
        obs, r, done, _ = env.step(action)
        step_count += 1
        rs.append(r)
    print("total reward:", sum(rs))
    env.save_custom_logs()
    np.savez("{}/rewards.npz".format(env.path), rews=np.array(rs))


if __name__ == "__main__":
    main()
