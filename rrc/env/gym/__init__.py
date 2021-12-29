from gym.envs.registration import register
from rrc.env import make_env

register(
    id="RobotContactCubeEnv-v0",
    entry_point=make_env.env_fn_generator(
        diff=1, env_cls="robot_contact_env", monitor=False
    ),
)
