import wandb
import gym
import torch.nn as nn
import os.path as osp
import numpy as np
from typing import Any, Dict
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import CombinedExtractor, NatureCNN
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback


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
