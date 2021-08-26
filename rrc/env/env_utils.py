from collections import namedtuple
from copy import copy

import numpy as np
import pybullet as p

default_cam_kwargs = dict(
    render_width=256, render_height=192, cam_dist=0.65, cam_yaw=0.0, cam_pitch=-45.0
)


class LinearSchedule:
    def __init__(self, n_steps=100, start=0.0, end=-9.81, rounding=True):
        self.current = start
        self.step_size = (end - start) / n_steps
        self.n_steps = n_steps
        self.current_step = 0.0
        self.rounding = rounding

    def __call__(self):
        self.current_step += 1
        if self.current_step <= self.n_steps:
            self.current += self.step_size
        if self.rounding:
            self.current = round(self.current, 4)
        return self.current


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
