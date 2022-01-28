import os
import os.path as osp
import time

from rrc.env import make_env
from rrc.env.env_utils import LinearSchedule
from rrc.env.reward_fns import *
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rrc.models.model_utils import (
    WandbEvalCallback,
    LogEpInfoCallback,
    HERCombinedExtractor,
)

import wandb


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
    n_steps,
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
    gamma = 1 - (10 ** -np.floor(np.log10(ep_len)))
    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=replay_buffer_class,
        # Parameters for HER
        replay_buffer_kwargs=rb_kwargs,
        policy_kwargs=policy_kwargs,
        verbose=1,
        buffer_size=int(n_steps),
        learning_starts=10000,
        learning_rate=lr,
        gamma=gamma,
        batch_size=256,
        residual=residual,
        # ent_coef="auto",
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
    her=True,
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
    if her:
        exp_name = "HER-SAC_rrc-diff{}".format(diff)
    else:
        exp_name = "SAC_rrc-diff{}".format(diff)

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
        ep_len,
        lr,
        n_steps,
        env,
        use_goal,
        load_path=load_path,
        residual=residual,
        her=her,
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, type=str2bool, dest=name, const=True, nargs="?")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # Training args
    parser.add_argument("--n_steps", type=float, default=1e5)
    parser.add_argument("--lr", type=str, default="3e-4")
    parser.add_argument("--normalize_obs", "--norm_obs", action="store_true")
    parser.add_argument("--normalize_reward", "--norm_reward", action="store_true")

    # Model args
    parser.add_argument("--sde", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--load_path", type=str)
    add_bool_arg(parser, "her", default=True)

    # Env args
    parser.add_argument("--diff", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--ep_len", type=int, default=2500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--rew_fn", default="train5", type=str)
    parser.add_argument("--no_goal", action="store_true")
    parser.add_argument("--contact", action="store_true")
    parser.add_argument("--env_cls", type=str)
    parser.add_argument("--term_fn", type=str)
    parser.add_argument("--init", default="centered_init", type=str)
    parser.add_argument("--gravity", default="-9.81", type=str)
    parser.add_argument("--scale", type=str)
    parser.add_argument("--action_type", default=None)

    # Experiment args
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    norm_env = {}
    if args.normalize_obs or args.normalize_reward:
        norm_env = dict(norm_obs=args.normalize_obs, norm_reward=args.normalize_reward)
    if args.gravity:
        if args.gravity == "lin":
            args.gravity = LinearSchedule(
                0, -9.81, n_steps=(args.n_steps - 20000) / args.ep_len
            )
        else:
            args.gravity = float(args.gravity)

    if args.env_cls is None and args.contact:
        args.env_cls = "contact_env"

    if args.scale:
        if len(args.scale.split(",")) > 1:
            args.scale = tuple([float(x) for x in args.scale.split(",")])
        else:
            args.scale = float(args.scale)

    try:
        lr = float(args.lr)
    except ValueError:
        initial_lr = float(args.lr.lstrip("lin_"))
        lr = LinearSchedule(initial_lr)

    run = None
    if not args.dry_run:
        run = wandb.init(project="cvxrl", name=args.name)  # sync_tensorboard=True)
        del args.name
        wandb.config.update(args)

    main(
        args.n_steps,
        args.diff,
        args.ep_len,
        lr,
        her=args.her,
        norm_env=norm_env,
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
        scale=args.scale,
        action_type=args.action_type,
    )
