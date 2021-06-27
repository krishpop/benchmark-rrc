from .cube_env import RealRobotCubeEnv, ActionType
import rrc.env.wrappers as wrappers
from gym.wrappers import Monitor


def get_initializer(name):
    from rrc.env import initializers
    if name is None:
        return None
    if hasattr(initializers, name):
        return getattr(initializers, name)
    else:
        raise ValueError(f"Can't find initializer: {name}")


def get_reward_fn(name):
    from rrc.env import reward_fns
    if name is None:
        return reward_fns.competition_reward
    if hasattr(reward_fns, name):
        return getattr(reward_fns, name)
    else:
        raise ValueError(f"Can't find reward function: {name}")


def get_termination_fn(name):
    from rrc.env import termination_fns
    if name is None:
        return None
    if hasattr(termination_fns, name):
        return getattr(termination_fns, name)
    elif hasattr(termination_fns, "generate_" + name):
        return getattr(termination_fns, "generate_" + name)()
    else:
        raise ValueError(f"Can't find termination function: {name}")


def make_env(cube_goal_pose, goal_difficulty, action_space, frameskip=1,
             sim=False, visualization=False, reward_fn=None,
             termination_fn=None, initializer=None, episode_length=119000,
             rank=0, monitor=False, path=None):
    reward_fn = get_reward_fn(reward_fn)
    initializer = get_initializer(initializer)(goal_difficulty)
    termination_fn = get_termination_fn(termination_fn)
    if action_space not in ['torque', 'position', 'torque_and_position', 'position_and_torque']:
        raise ValueError(f"Unknown action space: {action_space}.")
    if action_space == 'torque':
        action_type = ActionType.TORQUE
    elif action_space in ['torque_and_position', 'position_and_torque']:
        action_type = ActionType.TORQUE_AND_POSITION
    else:
        action_type = ActionType.POSITION
    env = RealRobotCubeEnv(cube_goal_pose,
                           goal_difficulty,
                           action_type=action_type,
                           frameskip=frameskip,
                           sim=sim,
                           visualization=visualization,
                           reward_fn=reward_fn,
                           termination_fn=termination_fn,
                           initializer=initializer,
                           episode_length=episode_length,
                           path=path)
    env.seed(seed=rank)
    env.action_space.seed(seed=rank)
    env = wrappers.NewToOldObsWrapper(env)
    env = wrappers.AdaptiveActionSpaceWrapper(env)
    if not sim:
        env = wrappers.TimingWrapper(env, 0.001)
    if visualization:
        env = wrappers.PyBulletClearGUIWrapper(env)
    if monitor:
        env = Monitor(wrappers.RenderWrapper(env), path, force=True)
    return env


def env_fn_generator(diff=3, episode_length=500, relative_goal=True,
                     reward_fn=None, termination_fn=False, save_mp4=False,
                     save_dir='', save_freq=10, initializer=None, **env_kwargs):
    reward_fn = get_reward_fn(reward_fn)

    goal = None
    if initializer is None:
        initializer = initializers.centered_init(diff)
    elif initializer =='center':
        initializer = initializers.centered_init
    elif initializer == 'train':
        initializer = initializers.training_init
    elif initializer == 'fixed':
        from trifinger_simulation.tasks.move_cube import Pose
        import json
        goal = Pose.from_json(json.load(open('goal.json', 'r'))).to_dict()
        initializer = initializers.fixed_g_init(diff, goal)
    else:
        initializer = get_initializer(initializer)(diff)

    termination_fn = get_termination_fn(termination_fn)

    def env_fn():
        env = cube_env.CubeEnv(goal, diff,
                initializer=initializer,
                episode_length=episode_length,
                relative_goal=relative_goal,
                reward_fn=reward_fn,
                force_factor=1.,
                torque_factor=.1,
                termination_fn=termination_fn,
                **env_kwargs)
        if save_mp4:
            env = MonitorPyBullet(env, save_dir, save_freq)
        env = FlattenGoalObs(env, ['desired_goal', 'achieved_goal', 'observation'])
        return Monitor(env, info_keywords=('ori_err', 'pos_err'))
    return env_fn


