import numpy as np
from scipy.spatial.transform import Rotation 
from rrc.env import cube_env, wrappers, initializers
import time

start_ori = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_quat()
start_ori = (0,0,0,1,)

def get_quat_ori(degrees):
    return Rotation.from_euler('xyz', degrees, degrees=True).as_quat()


camera_info = [1024, 768, (0.9997805953025818, -0.02073860913515091, 0.002914566546678543, 0.0, 0.020942410454154015, 0.9900512099266052, -0.13914000988006592, 0.0, -0.0, 0.13917051255702972, 0.9902684688568115, 0.0, 2.852175384759903e-09, -0.012565813958644867, -0.3000878095626831, 1.0), (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (-0.002914566546678543, 0.13914000988006592, -0.9902684688568115), (26660.818359375, 558.4642944335938, -0.0), (-414.7721862792969, 19801.025390625, 2783.41064453125), 1.2000000476837158, -82.00016021728516, 1.2000000476837158, (-0.0028834545519202948, 0.1376546025276184, -0.8894059062004089)]

camera_kw = ['width', 'height', 'viewMatrix', 'projectionMatrix',
             'cameraUp', 'cameraForward', 'horizontal', 'vertical', 'yaw', 'pitch',
             'dist', 'target']

info_kw = [('cameraYaw', 'yaw'), ('cameraDistance', 'dist'), 
     ('cameraPitch', 'pitch'), ('cameraTargetPosition', 'target')]

def cam_info_to_kwargs(info):
    cam_info =  {kw:val for kw, val in zip(camera_kw, info)}
    cam_kw = {kw: cam_info[kw2] for kw, kw2 in info_kw}
    return cam_kw
    

cam_kwargs = cam_info_to_kwargs(camera_info)

env = wrappers.PyBulletClearGUIWrapper(
        cube_env.ContactForceWrenchCubeEnv(None, 1, visualization=True,
            use_relaxed = False,
            initializer=initializers.fixed_init(1, default_initial_state=dict(
                position=np.array([0,0,0.0325]), 
                orientation=np.array(start_ori)))))

scale = 1
x_ac = np.array([.4,0,0,0,0,0])*scale
xy_ac = np.array([.4,0.4,0,0,0,0])*scale
y_ac = np.array([0,.4,0,0,0,0])*scale
z_ac = np.array([0,0,.5,0,0,0])
acs = [x_ac, -x_ac, -x_ac, xy_ac, -xy_ac, -xy_ac, y_ac, -y_ac, -y_ac, z_ac, -z_ac/2]

env.reset(camera_kwargs = cam_kwargs)
d = False
i = 0
while not d:
    ac = acs[i//50 % len(acs)]
    if i % 50 == 0:
        print(f"Wrench: {ac}")

    i += 1
    o,r,d,_ = env.step(ac)
