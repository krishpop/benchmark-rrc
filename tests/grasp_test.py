import os
import os.path as osp

import numpy as np
import pybullet as p
from rrc.env import cube_env, initializers, wrappers
from rrc.mp.const import CUBE_HALF_WIDTH, CUBE_WIDTH
from rrc_iprl_package.envs import cube_env as iprl_cube_env


def test_cube_env(
    env,
    ki,
    int_freq,
    policy,
    pd,
    pd_kwargs,
    logdir,
    visualization,
    use_actual_cp,
    gravity,
    benchmark,
):
    if logdir:
        path = osp.join(osp.split(osp.abspath(__file__))[0], logdir)
        if not osp.exists(path):
            os.makedirs(path)
        print("logging to {}".format(path))
    else:
        path = None
    if pd:
        default_goal = dict(
            position=np.array([-CUBE_WIDTH * 0.6, -CUBE_WIDTH * 0.6, CUBE_HALF_WIDTH]),
            orientation=np.array([0, 0, 0, 1]),
        )
        if gravity == 0.0:
            print("Setting initial cube position to 1.5*cube_half_width")
            default_init = dict(
                position=np.array(
                    [-CUBE_WIDTH * 0.0, -CUBE_WIDTH * 0.0, 3 * CUBE_HALF_WIDTH]
                ),
                orientation=np.array([0, 0, 0, 1]),
            )
        else:
            default_init = None
        init = initializers.dumb_init(
            1, default_goal=default_goal, default_initial_state=default_init
        )
    else:
        init = initializers.dumb_init(1)
    if env == "real":
        env = cube_env.RobotWrenchCubeEnv(
            dict(position=[0, 0, 0.05], orientation=[0, 0, 0, 1]),
            1,
            ki=ki,
            gravity=gravity,
            integral_control_freq=int_freq,
            visualization=visualization,
            initializer=init,
            debug=True,
            use_traj_opt=True,
            path=path,
            force_factor=1.0,
            torque_factor=0.1,
            episode_length=1000,
            use_actual_cp=use_actual_cp,
            use_benchmark_controller=benchmark,
        )
    elif env == "hog":
        env = cube_env.ContactForceWrenchCubeEnv(
            dict(position=[0, 0, 0.05], orientation=[0, 0, 0, 1]),
            1,
            visualization=visualization,
            initializer=init,
            debug=True,
            force_factor=1,
            torque_factor=0.1,
            episode_length=1000,
            path=path,
        )

    if pd:
        env = wrappers.ResidualPDWrapper(env, **pd_kwargs)
    env = wrappers.PyBulletClearGUIWrapper(env)
    obs = env.reset()
    if gravity == 0.0:
        env.gravity = -9.81
        p.setGravity(
            0, 0, -9.81, physicsClientId=env.platform.simfinger._pybullet_client_id
        )
    d = False
    while not d:
        # q = obs["observation"]["position"]
        # dq = obs["observation"]["velocity"]
        ac = policy(obs)
        obs, r, d, i = env.step(ac)
        # env.step(env.action_space.sample())
        # env.execute_simple_traj(env.step_count * np.pi / 10, q, dq)


def test_iprl_cube_env():
    env = iprl_cube_env.RobotCubeEnv()
    return env


def main(
    ki,
    int_freq,
    logdir,
    policy,
    env,
    pd,
    pd_kwargs,
    visualization,
    use_actual_cp,
    gravity,
    benchmark,
):
    if policy == "lift":
        ac = np.array([0, 0, 0.75, 0, 0, 0])
        pol = lambda obs: ac
    elif policy == "up":
        ac = np.array([0, 1.0, 0, 0, 0, 0])
        pol = lambda obs: ac
    elif policy == "down":
        ac = np.array([0, -1.0, 0, 0, 0, 0])
        pol = lambda obs: ac
    elif policy == "left":
        ac = np.array([1.0, 0, 0, 0, 0, 0])
        pol = lambda obs: ac
    elif policy == "right":
        ac = np.array([-1.0, 0, 0, 0, 0, 0])
        pol = lambda obs: ac
    if policy == "hold" or pd:
        ac = np.zeros(6)
        pol = lambda obs: ac
    elif policy == "random":
        pol = lambda obs: np.random.sample(6) * 0.2
    test_cube_env(
        env,
        ki,
        int_freq,
        pol,
        pd,
        pd_kwargs,
        logdir,
        visualization,
        use_actual_cp,
        gravity,
        benchmark,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="lift")
    parser.add_argument("--env", default="real")
    parser.add_argument("--logdir", default="output/")
    parser.add_argument("--visualization", "--v", action="store_true")
    parser.add_argument("--ki", default=0.1, type=float)
    parser.add_argument("--int_freq", default=10, type=int)
    parser.add_argument("--pd", action="store_true")
    parser.add_argument("--kp", default=10.0, type=float)
    parser.add_argument("--kd", default=1.0, type=float)
    parser.add_argument("--use_cp", action="store_true")
    parser.add_argument("--gravity", default=-9.81, type=float)
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    pd_kwargs = dict(Kp=args.kp, Kd=args.kd)
    main(
        args.ki,
        args.int_freq,
        args.logdir,
        args.policy,
        args.env,
        args.pd,
        pd_kwargs,
        args.visualization,
        args.use_cp,
        args.gravity,
        args.benchmark,
    )
