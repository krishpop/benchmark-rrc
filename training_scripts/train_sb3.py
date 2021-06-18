import gym
import os
import os.path as osp
import time
import numpy as np
import torch.nn as nn
import wandb
import pybullet as p

from copy import deepcopy
from collections import deque
from gym import ObservationWrapper
from gym.spaces import flatten_space
from gym.wrappers import FilterObservation
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import HerReplayBuffer, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.logger import configure
from rrc.env import cube_env, initializers
from rrc.env.cube_env import training_reward1, training_reward2, training_reward3
from rrc.env.wrappers import MonitorPyBulletWrapper
from rrc.env.reward_fns import competition_reward, _orientation_error, gaussian_reward


class WandbEvalCallback(EvalCallback):

    def _on_step(self):
        vids = self.eval_env.envs[0].videos[:]
        ret = super(WandbEvalCallback, self)._on_step()
        vids = [v for v in self.eval_env.envs[0].videos if v not in vids]
        for v in vids:
            wandb.log({'EvalRollout': wandb.Video(v)})
        return ret


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
                if k not in ['r', 'l', 't']:
                    self.logger.record("rollout/ep_{}_mean".format(k),
                            safe_mean([info[k] for info in last_k_eps]))
            self.last_ep_count = self.ep_count
        return True


class FlattenGoalObs(ObservationWrapper):
    def __init__(self, env, observation_keys):
        super().__init__(env)
        obs_space = self.env.observation_space
        obs_dict = {k: flatten_space(obs_space[k]) for k in observation_keys}
        self.observation_space = gym.spaces.Dict(obs_dict)

    def observation(self, obs):
        n_obs = {}
        for k in self.observation_space.spaces:
            if isinstance(obs[k], dict):
                obs_list = [obs[k][k2] for k2 in self.env.observation_space[k]]
                n_obs[k] = np.concatenate(obs_list)
            else:
                n_obs[k] = obs[k]
        return n_obs


class HERCombinedExtractor(CombinedExtractor):
    """
    HERCombinedExtractor is a combined extractor which only extracts pre-specified observation_keys to include in
    the observation, while retaining them at the environment level so that they may still be stored in the replay buffer
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256, observation_keys: list = []):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key in observation_keys:
            subspace = observation_space.spaces[key]
            # The observation key is a vector, flatten it if needed
            extractors[key] = nn.Flatten()
            total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size


def env_fn_generator(diff=3, initializer=initializers.training_init,
                     episode_length=500, relative_goal=True, reward_fn=None,
                     save_mp4=False, save_dir='', save_freq=10, **env_kwargs):
    if reward_fn is None:
        reward_fn = training_reward3
    else:
        if reward_fn == 'train1':
            reward_fn = training_reward1
        elif reward_fn == 'train2':
            reward_fn = training_reward2
        elif reward_fn == 'train3':
            reward_fn = training_reward3
        elif reward_fn == 'competition':
            reward_fn = competition_reward

    def env_fn():
        env = cube_env.CubeEnv(None, diff,
                initializer=initializer,
                episode_length=episode_length,
                relative_goal=relative_goal,
                reward_fn=reward_fn,
                torque_factor=.1,
                **env_kwargs)
        if save_mp4:
            env = MonitorPyBullet(env, save_dir, save_freq)
        env = FlattenGoalObs(env, ['desired_goal', 'achieved_goal', 'observation'])
        return Monitor(env, info_keywords=('ori_err', 'pos_err'))
    return env_fn


def make_exp_dir(method='SAC', env_str='rrc', *args):
    exp_root = './data'
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    env_str = env_str + '-'.join(list(args))
    exp_name = 'HER-{}_{}'.format(method, env_str)
    exp_dir = osp.join(exp_root, exp_name, hms_time)
    return exp_dir


def train_save_model(model, eval_env, exp_dir, n_steps=1e5, reset_num_timesteps=False,
                     render_env=None, callbacks=[]):
    if render_env:
        render_callback = WandbEvalCallback(
                render_env, eval_freq=10000, n_eval_episodes=1)
        callbacks.append(render_callback)
    else:
        render_callback = None
    if len(callbacks) > 1:
        callback = CallbackList(callbacks)
    else:
        callback = None if len(callbacks) == 0 else callbacks[0]

    model.learn(n_steps, eval_env=eval_env, n_eval_episodes=5,
                eval_freq=10000, reset_num_timesteps=False,
                eval_log_path=exp_dir, callback=callback)
    # Save the trained agent
    model.save(osp.join(exp_dir, '{:.0e}-steps'.format(model.num_timesteps)))
    return model


def make_model(ep_len, lr, exp_dir=None, env=None, use_goal=True,
               use_sde=False):
    if use_goal:
        obs_keys = ['desired_goal', 'observation']
    else:
        obs_keys = ['observation']

    policy_kwargs = dict(
                    log_std_init=-3,
                    features_extractor_class=HERCombinedExtractor,
                    features_extractor_kwargs=dict(observation_keys=obs_keys))
    if use_sde:
        sde_kwargs = dict(
                use_sde=True,
                use_sde_at_warmup=True,
                sde_sample_freq=64)
    else:
        sde_kwargs = {}

    rb_kwargs = dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                    online_sampling=False,
                    max_episode_length=ep_len)

    model = SAC('MultiInputPolicy', env,
                replay_buffer_class=HerReplayBuffer,
                # Parameters for HER
                replay_buffer_kwargs=rb_kwargs,
                policy_kwargs=policy_kwargs,
                verbose=1, buffer_size=int(1e6),
                learning_starts=1500,
                learning_rate=lr,
                gamma=0.99, batch_size=256, **sde_kwargs)
    return model


def main(n_steps, diff, ep_len, lr, norm_env={}, dry_run=False,
         render=False, reward_fn=None, use_goal=True, use_sde=True):
    exp_dir = make_exp_dir(diff)
    os.makedirs(exp_dir)
    if not dry_run:
        wandb.config.update({'exp_dir': exp_dir})

    env_fn = env_fn_generator(diff, episode_length=ep_len, reward_fn=reward_fn)
    env = DummyVecEnv([env_fn])
    if norm_env:
        env = VecNormalize(env, **norm_env)

    model = make_model(ep_len, lr, exp_dir, env, use_goal)
    logger = configure(exp_dir, ['stdout', 'wandb'])
    model.set_logger(logger)

    if render:
        save_dir = osp.join(exp_dir, 'videos')
        render_env_fn = env_fn_generator(diff, episode_length=ep_len,
                visualization=True, save_mp4=True, save_dir=save_dir,
                reward_fn='competition')
        render_env = DummyVecEnv([render_env_fn])
        if norm_env:
            render_env = VecNormalize(render_env, norm_obs=norm_env['norm_obs'],
                                      training=False)
    else:
        render_env = render_env_fn = None

    eval_env_fn = env_fn_generator(diff, episode_length=ep_len,
                                   reward_fn='competition')
    eval_env = DummyVecEnv([eval_env_fn])
    if norm_env:
        eval_env = VecNormalize(eval_env, norm_obs=norm_env['norm_obs'],
                                training=False)

    # Training model, with keyboard interrupting for saving
    callbacks = []
    callbacks.append(LogEpInfoCallback(log_freq=4))
    try:
        model = train_save_model(model, eval_env, exp_dir, n_steps,
                                 render_env=render_env, callbacks=callbacks)
    except KeyboardInterrupt as e:
        print("Interrupted training, saving and exiting")
        model.save(osp.join(exp_dir, '{:.0e}-steps'.format(model.num_timesteps)))
        raise e

    # Post training eval step
    mean_rew, std_rew = evaluate_policy(model, eval_env)
    wandb.log({'eval/mean_reward':mean_rew})
    print('Want to save Replay Buffer?')
    print('call `model.save_replay_buffer(exp_dir)`')
    import IPython; IPython.embed()
    if norm_env:
        stats_path = osp.join(exp_dir, "vec_normalize.pkl")
        print('saving env stats to {}'.format(stats_path))
        env.save(stats_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=float, default=1e5)
    parser.add_argument('--diff', type=int, default=3, choices=[1,2,3,4])
    parser.add_argument('--ep_len', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--normalize_obs', '--norm_obs', action='store_true')
    parser.add_argument('--normalize_reward', '--norm_reward', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--rew_fn', default='train2', type=str)
    parser.add_argument('--no_goal', action='store_true')
    parser.add_argument('--no_sde', action='store_true')
    parser.add_argument('--name', type=str)
    args = parser.parse_args()

    norm_env = {}
    if args.normalize_obs or args.normalize_reward:
        norm_env = dict(norm_obs=args.normalize_obs, norm_reward=args.normalize_reward)

    if not args.dry_run:
        wandb.init(project='cvxrl', name=args.name)# sync_tensorboard=True)
        del args.name
        wandb.config.update(args)

    main(args.n_steps, args.diff, args.ep_len, args.lr, norm_env,
         dry_run=args.dry_run, render=args.render, reward_fn=args.rew_fn,
         use_goal=not(args.no_goal), use_sde=not(args.no_sde))
