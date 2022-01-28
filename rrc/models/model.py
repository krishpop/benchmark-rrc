import os.path as osp

from rrc.models.model_utils import HERCombinedExtractor

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def make_model(
    ep_len,
    lr,
    model_name="MultiInputPolicy",
    exp_dir=None,
    env=None,
    use_goal=True,
    use_sde=False,
    log_std_init=-3,
    load_path=None,
    residual=False,
    her=True,
    model_kwargs=dict(
        verbose=1,
        buffer_size=int(1e6),
        learning_starts=20000,
        gamma=0.99,
        batch_size=256,
    ),
    rb_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
        online_sampling=False,
    ),
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
        rb_kwargs["max_episode_length"] = ep_len

    replay_buffer_class = HerReplayBuffer if her else None
    model = SAC(
        model_name,
        env,
        # tensorboard_log=exp_dir,
        replay_buffer_class=replay_buffer_class,
        # Parameters for HER
        replay_buffer_kwargs=rb_kwargs,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        residual=residual,
        **sde_kwargs,
        **model_kwargs
    )
    if load_path is not None:
        if osp.isdir(load_path):
            load_path = osp.join(load_path, "best_model.zip")
        model = model.load(load_path, env)
    return model
