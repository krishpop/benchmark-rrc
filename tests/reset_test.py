import argparse
import numpy as np

from rrc.env import make_env
from trifinger_simulation.tasks.move_cube import Pose


def main(n_episodes=40, visualization=True, use_traj_opt=False):
    env = make_env.env_fn_generator(
        diff=1,
        env_cls="robot_wrench_env",
        visualization=visualization,
        flatten_goal=False,
        use_traj_opt=use_traj_opt,
    )()
    dists = []
    grasp_succ = []
    obj_poses = []
    for _i in range(n_episodes):
        obs = env.reset()
        obj_pose = Pose.from_dict(env.prev_observation["achieved_goal"])
        cp_list = env.get_cp_wf_list(env.cp_params, obj_pose)
        dist = env.prev_observation["observation"]["tip_positions"] - np.asarray(
            [cp[0] for cp in cp_list]
        )
        dists.append(dist)
        print("Distance to desired contact points:\n", dist)
        print("L2 distance:\n", np.linalg.norm(dist))
        # grasp_succ = np.linalg.norm(dist) < 0.05
        grasp_success = int(input("Grasp success: "))
        grasp_succ.append(grasp_success)
        obj_poses.append(obs["achieved_goal"])
    print(np.mean(grasp_succ))
    np.savez(
        open(
            "grasp_success.npz", grasp_succ=grasp_succ, obj_poses=obj_poses, dists=dists
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", action="store_true")
    parser.add_argument("--n", default=40)
    parser.add_argument("--to", action="store_true")
    args = parser.parse_args()
    main(args.n, args.v, args.to)
