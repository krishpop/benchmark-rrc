import enum
from collections import namedtuple
from copy import copy

import numpy as np
import pybullet as p

default_cam_kwargs = dict(
    render_width=256, render_height=192, cam_dist=0.65, cam_yaw=0.0, cam_pitch=-45.0
)


class LinearSchedule:
    def __init__(self, start=0.0, end=0.0, n_steps=None):
        self.initial_value = start
        self.final_value = end
        self.n_steps = n_steps
        self.current_step = 0.0

    def __call__(self, progress_remaining=None):
        if progress_remaining is None:
            progress_remaining = max(0, 1 - self.current_step / self.n_steps)
            self.current_step += 1
        return (
            progress_remaining * self.initial_value
            + (1 - progress_remaining) * self.final_value
        )


def render_frame(env, **cam_kwargs):
    pid = env.platform.simfinger._pybullet_client_id
    ck = copy(default_cam_kwargs)
    ck.update(cam_kwargs)
    aspect = float(ck.get("render_width")) / ck.get("render_height")
    base_pos = [0, 0, 0]
    if pid >= 0:
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=ck.get("cam_dist"),
            yaw=ck.get("cam_yaw"),
            pitch=ck.get("cam_pitch"),
            roll=0,
            upAxisIndex=2,
            physicsClientId=pid,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=aspect,
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=pid,
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=ck.get("render_width"),
            height=ck.get("render_height"),
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=pid,
        )

        p.configureDebugVisualizer(
            p.COV_ENABLE_SINGLE_STEP_RENDERING,
            1,
            physicsClientId=pid,
        )
    else:
        px = np.array(
            [[[255, 255, 255, 255]] * ck.get("render_width")] * ck.get("render_height"),
            dtype=np.uint8,
        )

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(
        np.array(px), (ck.get("render_height"), ck.get("render_width"), -1)
    )
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


class PolicyMode(enum.Enum):
    RESET = enum.auto()
    TRAJ_OPT = enum.auto()
    IMPEDANCE = enum.auto()
    RL_PUSH = enum.auto()
    RESIDUAL = enum.auto()


ContactResult = namedtuple(
    "ContactResult",
    [
        "contactFlag",
        "bodyUniqueIdA",
        "bodyUniqueIdB",
        "linkIndexA",
        "linkIndexB",
        "positionOnA",
        "positionOnB",
        "contactNormalOnB",
        "contactDistance",
        "normalForce",
        "lateralFriction1",
        "lateralFrictionDir1",
        "lateralFriction2",
        "lateralFrictionDir2",
    ],
)
