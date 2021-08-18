import numpy as np
from rrc.env import cube_env, wrappers

env = cube_env.ContactForceWrenchCubeEnv(None, 1, visualization=True, debug=True)
env = wrappers.PyBulletClearGUIWrapper(env)

env.reset()
ac = np.array([0, 1, 0, 0, 0, 0]) * .1
d = False
while not d:
    o, r, d, i = env.step(ac)
