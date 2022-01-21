import os
import os.path as osp
import shelve
import numpy as np
import pybullet as p
from rrc.env import initializers, wrappers, viz_utils

# from rrc.env import cube_env_old as cube_env
from rrc.env import cube_env
from rrc.mp.const import CUBE_HALF_WIDTH, CUBE_WIDTH


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
    impedance,
    traj_opt,
    object_shape="cube",
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
        default_init = None
        init = initializers.dumb_init(
            1, default_goal=default_goal, default_initial_state=default_init
        )
    else:
        init = initializers.fixed_init(2)
    if env == "real":
        env = cube_env.RobotWrenchCubeEnv(
            dict(position=[0, 0, 0.05], orientation=[0, 0, 0, 1]),
            2,
            ki=ki,
            integral_control_freq=int_freq,
            visualization=visualization,
            initializer=init,
            debug=True,
            use_traj_opt=traj_opt,
            path=path,
            force_factor=1.0,
            torque_factor=0.1,
            episode_length=1000,
            use_actual_cp=use_actual_cp,
            use_impedance=impedance,
            object_shape=object_shape,
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
    d = False
    dxs, ddxs, ft_vels, ft_poss = [], [], [], []
    while not d:
        # q = obs["observation"]["position"]
        # dq = obs["observation"]["velocity"]
        ac = policy(obs).copy()
        obs, r, d, i = env.step(ac)
        delta_x = ac[:3]
        q_curr = env.prev_observation["observation"]["position"]
        dq_curr = env.prev_observation["observation"]["velocity"]
        current_ft_pos = env.prev_observation["observation"]["tip_positions"]
        des_wrench = ac * np.concatenate([env.force_factor, [env.torque_factor] * 3])
        ft_pos, ft_vel, delta_dx, _ = env.get_ft_pos_vel_goals(
            q_curr, dq_curr, current_ft_pos, des_wrench
        )
        dxs.append(delta_x)
        ddxs.append(delta_dx)
        ft_vels.append(ft_vel)
        ft_poss.append(ft_pos)
        # env.step(env.action_space.sample())
        # env.execute_simple_traj(env.step_count * np.pi / 10, q, dq)
    ft_wf_traj = np.asarray(ft_poss).reshape(-1, 9)
    dx_traj = np.asarray(ddxs)
    ft_vel_traj = np.asarray(ft_vels)
    x_traj = np.asarray(dxs)
    np.savez(
        "test_ki_{}{}.npz".format(ki, int_freq),
        ft_wf_traj=ft_wf_traj,
        dx_traj=dx_traj,
        ft_vel_traj=ft_vel_traj,
        x_traj=x_traj,
    )
    env.save_custom_logs()
    data = extract_data(
        osp.join(env.path, "custom_data"), keys=["des_tip_forces", "obs_tip_forces"]
    )
    viz_utils.plot_3f_des_obs_data(data["des_tip_forces"], data["obs_tip_forces"])
    # d0 = np.load(osp.join(osp.split(osp.abspath(__file__))[0], "test.npz"))
    # viz_utils.plot_3f_des_obs_data(d0["ft_wf_traj"], ft_wf_traj, title="ft_pos_wf")


def extract_data(filepath, keys):
    with shelve.open(filepath) as f:
        data = {k: [np.array(d["data"]).flatten() for d in f[k]] for k in keys}
    return data


def test_iprl_cube_env():
    from rrc_iprl_package.envs.cube_env import RobotCubeEnv

    env = RobotCubeEnv()
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
    impedance,
    traj_opt,
    object_shape,
):
    if policy == "lift":
        ac = np.array([0, 0, 0.9, 0, 0, 0])
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
        impedance,
        traj_opt,
        object_shape,
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
    parser.add_argument("--impedance", action="store_true")
    parser.add_argument("--traj_opt", action="store_true")
    parser.add_argument("--object", default="cube")
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
        args.impedance,
        args.traj_opt,
        object_shape=args.object,
    )
