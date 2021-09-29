#!/usr/bin/env python3
"""Run a single episode with a controller in simulation."""
import argparse

from rrc.combined_code import create_state_machine
from rrc.env.make_env import make_env
from rrc.mp.utils import set_seed
from trifinger_simulation.tasks import move_cube


def _init_env(goal_pose_dict, difficulty, episode_length, method):
    eval_config = {
        'action_space': 'torque_and_position',
        'frameskip': 3,
        'reward_fn': 'competition_reward',
        'termination_fn': 'no_termination',
        'initializer': 'centered_init',
        'monitor': False,
        'visualization': True,
        'sim': True,
        'path': '/logdir/{}'.format(method),
        'rank': 0,
        'episode_length': int(episode_length)
    }

    set_seed(0)
    env = make_env(goal_pose_dict, difficulty, **eval_config)
    return env


def main():
    parser = argparse.ArgumentParser('args')
    parser.add_argument('difficulty', type=int)
    parser.add_argument('method', type=str, help="The method to run. One of 'mp-pg', 'cic-cg', 'cpc-tg'")
    parser.add_argument('--residual', default=False, action='store_true',
                        help="add to use residual policies. Only compatible with difficulties 3 and 4.")
    parser.add_argument('--bo', default=False, action='store_true',
                        help="add to use BO optimized parameters.")
    parser.add_argument('--episode_length', '--ep_len', default=1000)
    args = parser.parse_args()
    goal_pose = move_cube.sample_goal(args.difficulty)
    goal_pose_dict = {
        'position': goal_pose.position.tolist(),
        'orientation': goal_pose.orientation.tolist()
    }

    env = _init_env(goal_pose_dict, args.difficulty, args.episode_length, args.method)
    state_machine = create_state_machine(args.difficulty, args.method, env,
                                         args.residual, args.bo)

    #####################
    # Run state machine
    #####################
    obs = env.reset()
    state_machine.reset()

    done = False
    rs = []
    while not done:
        action = state_machine(obs)
        obs, r, done, _ = env.step(action)
        rs.append(r)
    np.savez('/logdir/rewards.npz', rews=np.array(rs))


if __name__ == "__main__":
    main()
