import os
import os.path as osp

import numpy as np
from rrc.env import cube_env, initializers, wrappers
from rrc_iprl_package.envs import cube_env as iprl_cube_env


def test_cube_env(ac, logdir):
    path = osp.join(osp.split(osp.abspath(__file__))[0], logdir)
    if not osp.exists(path):
        os.makedirs(path)
    print("logging to {}".format(path))
    env = cube_env.RobotWrenchCubeEnv(
        dict(position=[0, 0, 0.05], orientation=[0, 0, 0, 1]),
        1,
        visualization=False,
        initializer=initializers.dumb_init(1),
        debug=True,
        use_traj_opt=True,
        path=path,
        episode_length=2000,
    )
    env = wrappers.PyBulletClearGUIWrapper(env)
    obs = env.reset()
    d = False
    while not d:
        # q = obs["observation"]["position"]
        # dq = obs["observation"]["velocity"]
        obs, r, d, i = env.step(ac)
        # env.step(env.action_space.sample())
        # env.execute_simple_traj(env.step_count * np.pi / 10, q, dq)


def test_iprl_cube_env():
    env = iprl_cube_env.RealRobotCubeEnv()
    return env


def main(logdir):
    # ac = np.array([0, 0, 1.0, 0, 0, 0])
    ac = np.zeros(6)
    test_cube_env(ac, logdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="output/")

    args = parser.parse_args()
    main(args.logdir)
