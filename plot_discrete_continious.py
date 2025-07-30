# -----------------------------------------------------------------------------
# plot_utils.py
# -----------------------------------------------------------------------------

"""Matplotlib utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


################################################################################
# continuous latents
################################################################################

def plot_smoothed_cumulative_latents(
    latents: np.ndarray,  # [T, H]
    footshock_binary: np.ndarray,  # [T]
    *,
    bin_width: float = 0.01,
    duration: float | None = None,
    smoothing_sec: float = 0.5,
):
    """Replicates the notebook visualisation in a single callable."""

    T, H = latents.shape
    duration = duration or T * bin_width
    times = np.linspace(0, duration, T)

    # where shocks happen
    shock_times = times[np.nonzero(footshock_binary.squeeze())]
    sigma_bins = smoothing_sec / bin_width

    for i in range(H):
        x = latents[:, i]
        x = x - x.mean()
        x_smooth = gaussian_filter1d(x, sigma=sigma_bins)
        cum = np.cumsum(x_smooth)

        plt.figure(figsize=(10, 4))
        plt.plot(times, cum, lw=1, label=f"Latent {i+1}")
        for t in shock_times:
            plt.axvline(t, color="red", ls="--", alpha=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("Cumulative value")
        plt.title(f"Latent {i+1} cumulative (σ={smoothing_sec}s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

################################################################################
# discrete states
################################################################################

def plot_discrete_states(
    z: np.ndarray,  # [T]
    *,
    footshock_binary: np.ndarray | None = None,
    bin_width: float = 0.01,
):
    """Plot inferred discrete states as coloured horizontal bars."""

    T = len(z)
    times = np.arange(T) * bin_width
    change = np.where(np.diff(z) != 0)[0]
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [T - 1]))

    K = z.max() + 1
    colors = sns.color_palette("husl", K)

    fig, ax = plt.subplots(figsize=(15, 3))
    for s, e in zip(starts, ends):
        state = z[s]
        ax.hlines(state, times[s], times[e], color=colors[state], linewidth=4, alpha=0.9)

    if footshock_binary is not None:
        for t in np.where(footshock_binary.squeeze())[0]:
            ax.axvline(t * bin_width, color="black", ls="--", lw=0.7, alpha=0.6)

    ax.set_yticks(np.arange(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("Time (s)")
    ax.set_title("Inferred discrete states")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# __init__.py – re‑export public symbols so users can do e.g.::
#
#     from srnn_helpers import NeuralDataset, train_srnn
# -----------------------------------------------------------------------------

# NOTE: keep this at the bottom so that imports above succeed first

# flake8: noqa
from importlib import import_module as _imp
from types import ModuleType as _ModuleType

_modules = {
    "dataset_utils": ["NeuralDataset", "make_dataloader"],
    "dist_patch": ["apply_patch"],
    "networks_local": [
        "InferenceNetwork",
        "GenerativeSRNN",
        "build_inference_network",
        "build_generative_srnn",
    ],
    "train_utils": ["train_srnn", "evaluate_srnn"],
    "plot_utils": [
        "plot_smoothed_cumulative_latents",
        "plot_discrete_states",
    ],
}


def __getattr__(name: str):
    for mod_name, symbols in _modules.items():
        if name in symbols:
            mod: _ModuleType = _imp(f"srnn_helpers.{mod_name}", globals(), locals(), symbols)
            return getattr(mod, name)
    raise AttributeError(name)


def __dir__():
    out = list(globals().keys())
    for symbols in _modules.values():
        out.extend(symbols)
    return sorted(out)
