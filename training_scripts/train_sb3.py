import os
import os.path as osp
import time
from collections import deque
from copy import deepcopy
from typing import Any, Dict

import gym
import numpy as np
import pybullet as p
import torch.nn as nn
from gym.spaces import flatten_space
from rrc.env import make_env
from rrc.env.env_utils import LinearSchedule
from rrc.env.reward_fns import *
from rrc.env.termination_fns import (stay_close_to_goal,
                                     stay_close_to_goal_level_4)
from stable_baselines3 import SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.preprocessing import (get_flattened_obs_dim,
                                                    is_image_space)
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wandb


class WandbEvalCallback(EvalCallback):
    def _on_step(self):
        vids = self.eval_env.envs[0].videos[:]
        ret = super(WandbEvalCallback, self)._on_step()
        vids = [v for v in self.eval_env.envs[0].videos if v not in vids]
        for v in vids:
            if osp.exists(v):
                wandb.log({"EvalRollout": wandb.Video(v)})
        return ret


class LogEpInfoEvalCallback(EvalCallback):
    def __init__(self, **eval_callback_kwargs):
        super(LogEpInfoEvalCallback, self).__init__(**eval_callback_kwargs)
        self._ep_info_buffer = []

    def _log_success_callback(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        super(LogEpInfoEvalCallback, self)._log_success_callback(locals_, globals_)
        info = locals_["info"]
        info = {info[k] for k in info if k not in ["r", "l", "t"]}
        if info:
            self._ep_info_buffer.append(info)

    def _on_step(self) -> bool:
        eval_step = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

        super(LogEpInfoEvalCallback, self)._on_step()

        if eval_step and len(self._ep_info_buffer) > 0:
            for k in self._ep_info_buffer[-1].keys():
                collected_k = [i[k] for i in self._ep_info_buffer[-self.eval_freq :]]
                mean_k, std_k = np.mean(collected_k), np.std(collected_k)
                if self.verbose > 0:
                    print(f"Episode {k}: {mean_k:.2f} +/- {std_k:.2f}")
                # Add to current Logger
                self.logger.record(f"eval/mean_{k}", float(mean_reward))
        return True


class LogEpInfoCallback(BaseCallback):
    def __init__(self, verbose: int = 0, log_freq: int = 5):
        super(LogEpInfoCallback, self).__init__(verbose)
        self.ep_count = 0
        self.last_ep_count = 0
        self.log_freq = log_freq

    def _on_rollout_end(self):
        self.ep_count += 1

    def init_callback(self, model: "base_class.BaseAlgorithm"):
        super(LogEpInfoCallback, self).init_callback(model)
        self.ep_info_buffer = model.ep_info_buffer

    def _on_step(self):
        num_rollouts = self.ep_count - self.last_ep_count
        if len(self.ep_info_buffer) > 0 and num_rollouts >= self.log_freq:
            ep_keys = self.ep_info_buffer[-1].keys()
            last_k_eps = list(self.ep_info_buffer)[-num_rollouts:]
            for k in ep_keys:
                if k not in ["r", "l", "t"]:
                    self.logger.record(
                        "rollout/ep_{}_mean".format(k),
                        safe_mean([info[k] for info in last_k_eps]),
                    )
            self.last_ep_count = self.ep_count
        return True


class HERCombinedExtractor(CombinedExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_output_dim: int = 256,
        observation_keys: list = [],
    ):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key in observation_keys:
            subspace = observation_space.spaces[key]
            if is_image_space(subspace):
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size


def make_exp_dir(method="SAC", env_str="sparse_push"):
    exp_root = "./data"
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "HER-{}_{}".format(method, env_str)
    exp_dir = osp.join(exp_root, exp_name, hms_time)
    os.makedirs(exp_dir)
    return exp_dir


def train_save_model(
    model,
    eval_env,
    exp_dir,
    n_steps=1e5,
    reset_num_timesteps=False,
    render_env=None,
    callbacks=[],
):
    if render_env:
        render_callback = WandbEvalCallback(
            render_env, eval_freq=10000, n_eval_episodes=1
        )
        callbacks.append(render_callback)
    else:
        render_callback = None
    if len(callbacks) > 1:
        callback = CallbackList(callbacks)
    else:
        callback = None if len(callbacks) == 0 else callbacks[0]

    model.learn(
        n_steps,
        eval_env=eval_env,
        n_eval_episodes=5,
        eval_freq=10000,
        reset_num_timesteps=reset_num_timesteps,
        eval_log_path=exp_dir,
        callback=callback,
    )
    # Save the trained agent
    model.save(osp.join(exp_dir, "{:.0e}-steps".format(model.num_timesteps)))
    return model


def make_model(
    ep_len,
    lr,
    env=None,
    use_goal=True,
    use_sde=False,
    log_std_init=-3,
    load_path=None,
    residual=False,
    her=True,
):
    if use_goal:
        obs_keys = ["desired_goal", "achieved_goal", "observation"]
    else:
        obs_keys = ["observation"]

    policy_kwargs = dict(log_std_init=log_std_init)
    if her:
        policy_kwargs.update(
            dict(
                features_extractor_class=HERCombinedExtractor,
                features_extractor_kwargs=dict(observation_keys=obs_keys),
            )
        )
    if use_sde:
        sde_kwargs = dict(use_sde=True, use_sde_at_warmup=False, sde_sample_freq=64)
    else:
        sde_kwargs = {}

    if her:
        rb_kwargs = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            online_sampling=False,
            max_episode_length=ep_len,
        )
    else:
        rb_kwargs = {}
    replay_buffer_class = HerReplayBuffer if her else None
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=replay_buffer_class,
        # Parameters for HER
        replay_buffer_kwargs=rb_kwargs,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=int(1e6),
        learning_starts=20000,
        learning_rate=lr,
        gamma=0.99,
        batch_size=256,
        residual=residual,
        **sde_kwargs,
    )
    if load_path is not None:
        if osp.isdir(load_path):
            load_path = osp.join(load_path, "best_model.zip")
        model = model.load(load_path, env)
    return model


def main(
    n_steps,
    diff,
    ep_len,
    lr,
    norm_env={},
    dry_run=False,
    render=False,
    use_goal=True,
    use_sde=True,
    load_path=None,
    residual=False,
    seed=None,
    reward_fn=None,
    **make_env_kwargs,
):
    exp_root = "./data"
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = "HER-SAC_rrc-diff{}".format(diff)
    exp_dir = osp.join(exp_root, exp_name, hms_time)
    os.makedirs(exp_dir)
    if not dry_run:
        wandb.config.update({"exp_dir": exp_dir})

    env_fn = make_env.env_fn_generator(
        diff,
        episode_length=ep_len,
        reward_fn=reward_fn,
        residual=residual,
        **make_env_kwargs,
    )
    env = DummyVecEnv([env_fn])
    if norm_env:
        env = VecNormalize(env, **norm_env)

    model = make_model(
        ep_len, lr, env, use_goal, load_path=load_path, residual=residual
    )
    model.set_random_seed(seed)
    if load_path:
        env.reset()

    log_kinds = ["stdout", "wandb"] if not dry_run else ["stdout"]
    logger = configure(exp_dir, log_kinds)
    model.set_logger(logger)

    if render:
        save_dir = osp.join(exp_dir, "videos")
        render_env_fn = make_env.env_fn_generator(
            diff,
            episode_length=ep_len,
            visualization=True,
            save_mp4=True,
            save_dir=save_dir,
            reward_fn="competition",
            residual=residual,
            **make_env_kwargs,
        )
        render_env = DummyVecEnv([render_env_fn])
        if norm_env:
            render_env = VecNormalize(
                render_env, norm_obs=norm_env["norm_obs"], training=False
            )
    else:
        render_env = render_env_fn = None

    eval_env_fn = make_env.env_fn_generator(
        diff,
        episode_length=ep_len,
        reward_fn="competition",
        residual=residual,
        **make_env_kwargs,
    )
    eval_env = DummyVecEnv([eval_env_fn])
    if norm_env:
        eval_env = VecNormalize(eval_env, norm_obs=norm_env["norm_obs"], training=False)

    # Training model, with keyboard interrupting for saving
    callbacks = []
    callbacks.append(LogEpInfoCallback(log_freq=4))
    try:
        model = train_save_model(
            model,
            eval_env,
            exp_dir,
            n_steps,
            render_env=render_env,
            callbacks=callbacks,
        )
    except KeyboardInterrupt as e:
        print("Interrupted training, saving and exiting")
        model.save(osp.join(exp_dir, "{:.0e}-steps".format(model.num_timesteps)))
        raise e

    # Post training eval step
    mean_rew, std_rew = evaluate_policy(model, eval_env)
    wandb.log({"eval/mean_reward": mean_rew})
    print("Want to save Replay Buffer?")
    print("call `model.save_replay_buffer(exp_dir)`")
    import IPython

    IPython.embed()
    if norm_env:
        stats_path = osp.join(exp_dir, "vec_normalize.pkl")
        print("saving env stats to {}".format(stats_path))
        env.save(stats_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_steps", type=float, default=1e5)
    parser.add_argument("--diff", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--ep_len", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--normalize_obs", "--norm_obs", action="store_true")
    parser.add_argument("--normalize_reward", "--norm_reward", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--rew_fn", default="train5", type=str)
    parser.add_argument("--no_goal", action="store_true")
    parser.add_argument("--sde", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--contact", action="store_true")
    parser.add_argument("--env_cls", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--term_fn", type=str)
    parser.add_argument("--init", default="centered_init", type=str)
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--gravity", default="-9.81", type=str)
    parser.add_argument("--scale", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--object_frame", "--of", action="store_true")
    parser.add_argument("--action_type", default=None)
    args = parser.parse_args()

    norm_env = {}
    if args.normalize_obs or args.normalize_reward:
        norm_env = dict(norm_obs=args.normalize_obs, norm_reward=args.normalize_reward)
    if args.gravity:
        if args.gravity == "lin":
            args.gravity = LinearSchedule(n_steps=(args.n_steps - 20000) / args.ep_len)
        else:
            args.gravity = float(args.gravity)

    if args.env_cls is None and args.contact:
        args.env_cls = "contact_env"

    if args.scale:
        if len(args.scale.split(",")) > 1:
            args.scale = tuple([float(x) for x in args.scale.split(",")])
        else:
            args.scale = float(args.scale)

    run = None
    if not args.dry_run:
        run = wandb.init(project="cvxrl", name=args.name)  # sync_tensorboard=True)
        del args.name
        wandb.config.update(args)

    main(
        args.n_steps,
        args.diff,
        args.ep_len,
        args.lr,
        norm_env,
        dry_run=args.dry_run,
        render=args.render,
        reward_fn=args.rew_fn,
        use_goal=not (args.no_goal),
        use_sde=args.sde,
        termination_fn=args.term_fn,
        initializer=args.init,
        load_path=args.load_path,
        residual=args.residual,
        env_cls=args.env_cls,
        gravity=args.gravity,
        seed=args.seed,
        object_frame=args.object_frame,
        scale=args.scale,
        action_type=args.action_type,
    )
