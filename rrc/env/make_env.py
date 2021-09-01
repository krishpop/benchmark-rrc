import functools
import json
import os.path as osp

import numpy as np
import rrc.env.wrappers as wrappers
from gym.wrappers import Monitor
from rrc.env import cube_env, initializers
from trifinger_simulation.tasks.move_cube import Pose

from .cube_env import ActionType, RealRobotCubeEnv


def get_env_cls(name):
    if name is None:
        return cube_env.CubeEnv
    if hasattr(cube_env, name):
        return getattr(cube_env, name)
    else:
        raise ValueError(f"Can't find env_cls: {name}")


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


def make_env(
    cube_goal_pose,
    goal_difficulty,
    action_space,
    frameskip=1,
    sim=False,
    visualization=False,
    reward_fn=None,
    termination_fn=None,
    initializer=None,
    episode_length=119000,
    rank=0,
    monitor=False,
    path=None,
):
    reward_fn = get_reward_fn(reward_fn)
    initializer = get_initializer(initializer)(goal_difficulty)
    termination_fn = get_termination_fn(termination_fn)
    if action_space not in [
        "torque",
        "position",
        "torque_and_position",
        "position_and_torque",
    ]:
        raise ValueError(f"Unknown action space: {action_space}.")
    if action_space == "torque":
        action_type = ActionType.TORQUE
    elif action_space in ["torque_and_position", "position_and_torque"]:
        action_type = ActionType.TORQUE_AND_POSITION
    else:
        action_type = ActionType.POSITION
    env = RealRobotCubeEnv(
        cube_goal_pose,
        goal_difficulty,
        action_type=action_type,
        frameskip=frameskip,
        sim=sim,
        visualization=visualization,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        initializer=initializer,
        episode_length=episode_length,
        path=path,
    )
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


def make_env_cls(
    diff=3,
    initializer="training_init",
    episode_length=500,
    reward_fn=None,
    termination_fn=False,
    **env_kwargs,
):
    if reward_fn is None:
        reward_fn = get_reward_fn("train4")
    else:
        reward_fn = get_reward_fn(reward_fn)

    if termination_fn:
        if diff < 4:
            termination_fn = get_termination_fn("stay_close_to_goal")
        else:
            termination_fn = get_termination_fn("stay_close_to_goal_level_4")
    else:
        termination_fn = None

    if initializer is None:
        initializer = initializers.centered_init
    elif initializer == "fixed":
        goal_fp = osp.join(osp.split(__file__)[0], "goal.json")
        goal = Pose.from_json(json.load(open(goal_fp, "r"))).to_dict()
        initializer = initializers.fixed_g_init(diff, goal)
    else:
        initializer = get_initializer(initializer)(diff)

    env_cls = functools.partial(
        cube_env.CubeEnv,
        cube_goal_pose=None,
        goal_difficulty=diff,
        initializer=initializer,
        episode_length=episode_length,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        force_factor=1.0,
        torque_factor=0.1,
        **env_kwargs,
    )
    return env_cls


def env_fn_generator(
    diff=3,
    episode_length=500,
    reward_fn=None,
    termination_fn=None,
    save_mp4=False,
    save_dir="",
    save_freq=1,
    initializer=None,
    residual=False,
    env_cls=None,
    flatten_goal=True,
    scale=None,
    action_type=None,
    **env_kwargs,
):
    reward_fn = get_reward_fn(reward_fn)

    goal = None
    if initializer is None:
        initializer = initializers.centered_init(diff)
    elif initializer == "fixed":
        goal_fp = osp.join(osp.split(__file__)[0], "goal.json")
        goal = Pose.from_json(json.load(open(goal_fp, "r"))).to_dict()
        initializer = initializers.fixed_g_init(diff, goal)
    else:
        initializer = get_initializer(initializer)(diff)

    if termination_fn is not None:
        termination_fn = get_termination_fn(termination_fn)

    info_keywords = ("ori_err", "pos_err")
    if env_cls in ["real_env", "wrench_robot_env"]:
        info_keywords = ("ori_err", "pos_err", "corner_err", "tot_tip_pos_err")
    if env_cls == "real_env":
        if action_type is not None:
            if action_type not in [
                "torque",
                "position",
                "torque_and_position",
                "position_and_torque",
            ]:
                raise ValueError(f"Unknown action space: {action_type}.")
            if action_type == "torque":
                env_kwargs["action_type"] = ActionType.TORQUE
            elif action_type in ["torque_and_position", "position_and_torque"]:
                env_kwargs["action_type"] = ActionType.TORQUE_AND_POSITION
            else:
                env_kwargs["action_type"] = ActionType.POSITION
        env_kwargs["sim"] = True
        if "object_frame" in env_kwargs:
            env_kwargs.pop("object_frame")
    else:
        # TODO (fix): hard-coding force and torque factor
        if scale is not None and len(scale) == 6:
            force_factor, torque_factor = np.asarray(scale[:3]), np.asarray(scale[3:])
        else:
            force_factor, torque_factor = scale or (0.5, 0.1)
        if residual:
            r_force_factor, r_torque_factor = force_factor, torque_factor
            force_factor, torque_factor = 1.0, 1.0
        env_kwargs["torque_factor"] = torque_factor
        env_kwargs["force_factor"] = force_factor

    env_cls = get_env_cls(env_cls)

    def env_fn():

        env = env_cls(
            goal,
            diff,
            initializer=initializer,
            episode_length=episode_length,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            **env_kwargs,
        )
        if residual:
            env = wrappers.ResidualPDWrapper(
                env, force_factor=r_force_factor, torque_factor=r_torque_factor
            )
        if save_mp4:
            env = wrappers.MonitorPyBulletWrapper(env, save_dir, save_freq)
        elif env_kwargs.get("visualization", False):
            env = wrappers.PyBulletClearGUIWrapper(env)
        if flatten_goal:
            env = wrappers.FlattenGoalObs(
                env, ["desired_goal", "achieved_goal", "observation"]
            )
        return wrappers.Monitor(env, info_keywords=info_keywords)

    return env_fn
