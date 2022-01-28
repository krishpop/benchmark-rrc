import os.path as osp
import wandb
import gym
from stable_baselines3 import SAC, HerReplayBuffer
from rrc.env import make_env, cube_env, initializers
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .training_scripts.training_utils import HERCombinedExtractor


def make_model(
    ep_len,
    lr,
    exp_dir=None,
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
        # tensorboard_log=exp_dir,
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
        **sde_kwargs
    )
    if load_path is not None:
        if osp.isdir(load_path):
            load_path = osp.join(load_path, "best_model.zip")
        model = model.load(load_path, env)
    return model


wandb_root = "/scr-ssd/ksrini/benchmark-rrc/training_scripts"
get_save_path = lambda run: "/".join(
    [wandb_root] + run.config["exp_dir"].split("/")[1:]
)


def make_sb3_env(run):
    # Create DummyVecEnv
    if run.config.get("contact"):
        env_cls = cube_env.ContactForceCubeEnv
    else:
        env_cls = cube_env.CubeEnv

    env_fn = make_env.env_fn_generator(
        diff=run.config["diff"],
        visualization=True,
        save_freq=1,
        initializer=run.config["init"],
        reward_fn=run.config["rew_fn"],
        residual=run.config.get("residual", False),
        env_cls=env_cls,
    )
    env = DummyVecEnv([env_fn])
    return env


def load_env_and_policy(run):
    # create env
    env = make_env(run)
    # Create model
    use_goal = not (run.config.get("no_goal"))
    use_sde = not (run.config.get("no_sde"))
    load_path = None  # osp.join(get_save_path(run), '2e+05-steps.zip')
    model = make_model(
        run.config.get("ep_len"),
        run.config["lr"],
        use_goal=use_goal,
        use_sde=use_sde,
        env=env,
        load_path=load_path,
    )
    return model, env


def run_hog_env(env, model, d=False):
    obs = env.reset()

    d = False
    rs = []
    acs = []
    x, x_vel = [], []

    while not d:
        ac, _ = model.predict(obs, deterministic=d)
        acs.append(ac)
        obs, r, d, i = env.step(ac)
        x.append(obs["position"])
        x_vel.append(obs["velocity"])
        rs.append(r)

    return acs, x, x_vel
