import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shelve
import pickle

from scipy.spatial.transform import Rotation

DEFAULT_OBSDES_LABELS = [
    "des normal",
    "des latfric1",
    "des latfric2",
    "obs normal",
    "obs latfric1",
    "obs latfric2",
]
DEFAULT_DATA_LABELS = ["normal", "latfric1", "latfric2"]


def plot_3f_des_obs_data(
    des_data, obs_data, line_labels=DEFAULT_OBSDES_LABELS, title="tip_forces"
):
    # Plots 3-Finger desired and observed data
    des_color, obs_color = ["b", "orange", "g"], [
        "cyan",
        "red",
        "olive",
    ]  # solid and dashed line colors
    des_style, obs_style = ["-"] * 3, ["--"] * 3
    f, ax = plt.subplots(1, 3, figsize=(16, 4))
    des_df = pd.DataFrame(des_data, columns=[f"d{i}" for i in range(9)])
    obs_df = pd.DataFrame(obs_data, columns=[f"d{i}" for i in range(9)])
    if len(des_df) != len(obs_df):
        des_df["steps"] = np.linspace(0, 1, len(des_df))
        obs_df["steps"] = np.linspace(0, 1, len(obs_df))
    else:
        des_df["steps"] = np.arange(0, len(des_df))
        obs_df["steps"] = np.arange(0, len(obs_df))
    for i in range(3):
        des_df.plot(
            y=[col for col in des_df.columns[3 * i : 3 * i + 3]],
            x="steps",
            ax=ax[i],
            title=f"Finger {i} {title}",
            legend=False,
            style=des_style,
            color=des_color,
        )
        lines = obs_df.plot(
            y=[col for col in obs_df.columns[3 * i : 3 * i + 3]],
            x="steps",
            ax=ax[i],
            title=f"Finger {i} {title}",
            legend=False,
            style=obs_style,
            color=obs_color,
        )
    plt.figlegend(
        lines.lines,
        line_labels,
        loc="lower center",
        borderaxespad=0.1,
        ncol=6,
        labelspacing=0.0,
        prop={"size": 13},
    )
    plt.show()


def plot_3f_data(data, line_labels=DEFAULT_DATA_LABELS, title="torques"):
    # Plots 3-Finger desired and observed data
    assert data.shape[1] == 9, "data needs to be 9D"
    color = ["b", "orange", "g"]  # solid and dashed line colors
    style = ["-"] * 3
    f, ax = plt.subplots(1, 3, figsize=(16, 4))
    df = pd.DataFrame(data, columns=[f"d{i}" for i in range(9)])
    for i in range(3):
        lines = df.plot(
            y=[col for col in df.columns[3 * i : 3 * i + 3]],
            ax=ax[i],
            title=f"Finger {i} {title}",
            legend=False,
            style=style,
            color=color,
        )
    plt.figlegend(
        lines.lines,
        line_labels,
        loc="lower center",
        borderaxespad=0.1,
        ncol=6,
        labelspacing=0.0,
        prop={"size": 13},
    )


def get_keys(output_dir):
    data = pickle.load(open(f"{output_dir}/observations.pkl", "rb"))[0]
    obs_keys = {k: [k2 for k2 in data[k].keys()] for k in data}
    data = shelve.open(f"{output_dir}/custom_data")
    return obs_keys, list(data.keys())


def load_custom_df(output_dir, labels=[], obs_labels=[], achieved=False):
    # Loads observations
    observations = pickle.load(open(f"{output_dir}/observations.pkl", "rb"))
    # Loads and formats custom logs for observed and desired tip forces
    custom_data = shelve.open(f"{output_dir}/custom_data")
    data = {}
    obs_keys, keys = get_keys(output_dir)
    obs_labels = obs_labels or [f"robot_{k}" for k in obs_keys["observation"]]
    if achieved:
        obs_labels += ["cube_position", "cube_orientation"]
    labels = labels or keys
    for label in labels:
        data[label] = [d.get("data") for d in custom_data.get(label)]
    observations = observations[-len(data[label]) :]
    for label in obs_labels:
        if "robot_" in label:
            data[label] = [
                obs["observation"][label[len("robot_") :]] for obs in observations
            ]
        else:
            data[label] = [
                obs["achieved_goal"][label[len("cube_") :]] for obs in observations
            ]
    # Makes all arrays the same length
    data = {k: data[k][-min([len(val) for k, val in data.items()]) :] for k in data}
    df = pd.DataFrame.from_dict(data)
    return df


def rotate_obs_force(obs_force, tip_dir):
    forces = []
    for f, td in zip(obs_force, tip_dir):
        v, w = td
        u = np.cross(v, w)
        R = np.vstack([u, v, w]).T
        forces.append(Rotation.from_matrix(R).inv().as_matrix() @ f)
    return forces


def flatten_data(d):
    """flatten_data() Flattens 2D data present in DataFrame"""
    return np.asarray(d).reshape(1, -1)


def get_data(df, key):
    return np.concatenate(df[key].apply(flatten_data).values)
