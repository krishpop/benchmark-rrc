import numpy as np
from rrc.env import initializers, make_env, wrappers

init = initializers.dumb_init(1)
env = make_env.env_fn_generator(
    # cube_goal_pose=dict(position=[0, 0, 0.05], orientation=[0, 0, 0, 1]),
    env_cls="real_env",
    diff=1,
    visualization=True,
    initializer=init,
    debug=True,
    path=None,
    episode_length=2000,
    action_scale=1,
)()
env = wrappers.PyBulletClearGUIWrapper(env)

env.reset()
d = False
while not d:
    o, r, d, i = env.step(env.action_space.sample())
    if d:
        print({k: i[k] for k in i if "err" in k})
