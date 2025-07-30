"""
analysis_utils.py
=================
Utilities for deciding the dimensionality of the SRNN’s continuous latent
space *H* and for performing quick PCA projections of trained latents.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

__all__ = [
    "choose_latent_dim",
    "pca_latents",
]


def choose_latent_dim(
    rates: np.ndarray,                      # shape [N, T]
    *,
    var_threshold: float = 0.90,
    max_dim: int | None = None,
    show_plot: bool = True,
) -> int:
    """Return the smallest *H* whose cumulative explained variance ≥ *var_threshold*.

    Parameters
    ----------
    rates : np.ndarray
        Firing‑rate matrix, neurons × time.
    var_threshold : float, default **0.90**
        Target cumulative variance to capture.
    max_dim : int | None
        Limit the search to at most this many principal components.  If *None*,
        use ``min(N, T)``.
    show_plot : bool, default True
        Whether to display the scree plot and mark the chosen *H*.

    Returns
    -------
    int
        Chosen latent dimension *H*.
    """
    if rates.ndim != 2:
        raise ValueError("rates must be 2‑D [N, T]")

    N, T = rates.shape
    max_dim = max_dim or min(N, T)

    pca = PCA(n_components=max_dim, svd_solver="randomized")
    pca.fit(rates.T)                     # time as samples

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    H = int(np.searchsorted(cumvar, var_threshold) + 1)

    if show_plot:
        idx = np.arange(1, len(cumvar) + 1)
        plt.figure(figsize=(4, 3))
        plt.plot(idx, cumvar, marker="o", lw=1)
        plt.axhline(var_threshold, color="r", ls="--", alpha=0.6)
        plt.axvline(H, color="g", ls="--", alpha=0.6)
        plt.xlabel("# principal components")
        plt.ylabel("cumulative explained variance")
        plt.title(f"Choose H = {H} (≥ {var_threshold:.0%})")
        plt.tight_layout()
        plt.show()

    return H


def pca_latents(latents: np.ndarray, *, n_components: int = 2):
    """Flatten any latent tensor to 2‑D and project with PCA.

    Parameters
    ----------
    latents : np.ndarray
        Shape (..., H).  All but the last dimension are stacked.
    n_components : int, default 2
        Number of PCA dimensions to keep.

    Returns
    -------
    proj : np.ndarray
        Flattened latent points in PCA space, shape (T_total, n_components).
    pca : sklearn.decomposition.PCA
        The fitted PCA object (so you can call ``inverse_transform`` etc.).
    """
    if latents.ndim < 2:
        raise ValueError("latents must have at least 2 dimensions (…, H)")

    H = latents.shape[-1]
    flat = latents.reshape(-1, H)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(flat)
    return proj, pca
