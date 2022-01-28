import wandb
import gym
import torch as th
import torch.nn as nn
import os.path as osp
import numpy as np
from rrc.env import fop
from typing import Any, Dict, List, Optional, Type, Union
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import CombinedExtractor, NatureCNN
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import register_policy


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
                self.logger.record(f"eval/mean_{k}", float(mean_k))
        return True


class LogEpInfoCallback(BaseCallback):
    def __init__(self, verbose: int = 0, log_freq: int = 5):
        super(LogEpInfoCallback, self).__init__(verbose)
        self.ep_count = 0
        self.last_ep_count = 0
        self.log_freq = log_freq

    def _on_rollout_end(self):
        self.ep_count += 1

    def init_callback(self, model="base_class.BaseAlgorithm"):
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


class CvxPolicy(MultiInputPolicy):
    """
    Policy class (with both actor and critic) for SAC with CVX layer.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(CvxPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
        self.fop = fop.BatchForceOptProblem()

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        actions = self.actor(observation, deterministic)
        return self.fop(actions, observation["cp_list"])


register_policy("CvxPolicy", CvxPolicy)
