from rrc.env import cube_env, wrappers, initializers
import numpy as np
import time


viz, debug = True, True

env = cube_env.ContactForceCubeEnv(None, 1, visualization=viz, debug=debug,
            initializer=initializers.fixed_init(difficulty=1,default_initial_state=dict(
                position=np.array([0,0,0.0325]),
                orientation=np.array([0,0,0,1]))))

env =  wrappers.PyBulletClearGUIWrapper(env)
center_ac = np.array([.5,0,0,0,0,0,0,0,0])
left_ac = np.array([0,0,0,.5,0,0,0,0,0])
right_ac = np.array([0,0,0,0,0,0,.5,0,0])
acs = [center_ac, left_ac, right_ac]
env.reset()
d = False
step = 0 
while not d:
    ac = acs[(step // 50) % 3]
    if step % 50 == 0:
        print(f"action: {ac}")
    o,r,d,i = env.step(ac)
    step += 1
