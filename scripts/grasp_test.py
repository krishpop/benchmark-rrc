from rrc.env import cube_env, wrappers, initializers

env = cube_env.RobotWrenchCubeEnv(
        dict(position=[0,0,.05],orientation=[0,0,0,1]), 
        1, visualization=True, initializer=initializers.dumb_init(1),
        debug=True)
env = wrappers.PyBulletClearGUIWrapper(env)
env.reset()
