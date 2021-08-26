import time

import numpy as np
from rrc.env import cube_env, initializers, wrappers
from rrc_iprl_package.control import controller_utils_cube as c_utils
from scipy.spatial.transform import Rotation
from trifinger_simulation.tasks.move_cube import Pose


def main():
    start_ori = Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat()
    start_ori = (
        0,
        0,
        0,
        1,
    )

    camera_info = [
        1024,
        768,
        (
            0.9997805953025818,
            -0.02073860913515091,
            0.002914566546678543,
            0.0,
            0.020942410454154015,
            0.9900512099266052,
            -0.13914000988006592,
            0.0,
            -0.0,
            0.13917051255702972,
            0.9902684688568115,
            0.0,
            2.852175384759903e-09,
            -0.012565813958644867,
            -0.3000878095626831,
            1.0,
        ),
        (
            0.7499999403953552,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            -0.02000020071864128,
            0.0,
        ),
        (0.0, 0.0, 1.0),
        (-0.002914566546678543, 0.13914000988006592, -0.9902684688568115),
        (26660.818359375, 558.4642944335938, -0.0),
        (-414.7721862792969, 19801.025390625, 2783.41064453125),
        1.2000000476837158,
        -82.00016021728516,
        1.2000000476837158,
        (-0.0028834545519202948, 0.1376546025276184, -0.8894059062004089),
    ]

    camera_kw = [
        "width",
        "height",
        "viewMatrix",
        "projectionMatrix",
        "cameraUp",
        "cameraForward",
        "horizontal",
        "vertical",
        "yaw",
        "pitch",
        "dist",
        "target",
    ]

    info_kw = [
        ("cameraYaw", "yaw"),
        ("cameraDistance", "dist"),
        ("cameraPitch", "pitch"),
        ("cameraTargetPosition", "target"),
    ]

    def cam_info_to_kwargs(info):
        cam_info = {kw: val for kw, val in zip(camera_kw, info)}
        cam_kw = {kw: cam_info[kw2] for kw, kw2 in info_kw}
        return cam_kw

    def get_quat_ori(degrees):
        return Rotation.from_euler("xyz", degrees, degrees=True).as_quat()

    def get_face_from_obs(obs):
        cube_pose = Pose.from_dict(obs["achieved_goal"])
        face = c_utils.get_closest_ground_face(cube_pose)
        return face

    def get_cp_params_from_obs(obs):
        cube_pose = Pose.from_dict(obs["achieved_goal"])
        return c_utils.get_lifting_cp_params(cube_pose)

    cam_kwargs = cam_info_to_kwargs(camera_info)

    env = wrappers.PyBulletClearGUIWrapper(
        cube_env.ContactForceWrenchCubeEnv(
            None,
            1,
            visualization=True,
            use_relaxed=False,
            force_factor=0.5,
            torque_factor=0.1,
            object_frame=True,
            initializer=initializers.fixed_init(
                1,
                default_initial_state=dict(
                    position=np.array([0, 0, 0.0325]), orientation=np.array(start_ori)
                ),
            ),
        )
    )

    x_ac = np.array([1, 0, 0, 0, 0, 0])
    xy_ac = np.array([0, 1, 0, 0, 0, 0])
    y_ac = np.array([0, 0, 1, 0, 0, 0])
    z_ac = np.array([0, 0, 0.5, 0, 0, 0])
    acs = [
        x_ac,
        -x_ac,
        -x_ac,
        xy_ac,
        -xy_ac,
        -xy_ac,
        y_ac,
        -y_ac,
        -y_ac,
        z_ac,
        -z_ac / 2,
    ]

    get_action = lambda i: acs[i // 50 % len(acs)]
    get_action = lambda i: np.clip(
        env.action_space.sample() + np.array([0, 0, 0.5, 0, 0, 0]), -1, 1
    )

    obs = env.reset()  # camera_kwargs=cam_kwargs)
    last_face = get_face_from_obs(obs)
    d = False
    i = 0

    while not d:
        ac = get_action(i)
        if i % 50 == 0:
            print(f"Wrench: {ac}")
        i += 1
        obs, r, d, info = env.step(ac)
        new_face = get_face_from_obs(obs)
        if last_face != new_face:
            print(f"Flipped! last face:{last_face}, new face: {new_face}")
            last_face = new_face
            env.unwrapped.cp_params = get_cp_params_from_obs(obs)
        if info["infeasible"] > 0:
            print(info["infeasible"])


if __name__ == "__main__":
    main()
