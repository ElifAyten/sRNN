# srnn_helper.py
# ------------------------------------------------------------
# Minimal, tidy SRNN training utilities with:
# - time-aware train/test split on a single long session
# - warm-up init phase (entropy + usage regularizers)
# - optional warm-start from rSLDS (pi0 + transitions)
# - kappa sweep support
# - TBPTT support (truncate backprop through time over window chunks)
# - fully UNSUPERVISED option (ignore behavior/shock inputs during training)
# - prediction evaluation via post-hoc linear decoder h->Z
# - DCA-lite or PCA front-end, robust NaN handling
# - CLI that reads YAML, supports multiple rat IDs, and rSLDS path templates
# ------------------------------------------------------------

from __future__ import annotations
import os, json, math, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Sequence

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from tqdm import tqdm


# ================
# Config utilities
# ================
def _cfg_to_jsonable(cfg_obj):
    """
    Convert TrainConfig dataclass to JSON-safe dictionary.
    Converts Path objects to strings.
    """
    d = asdict(cfg_obj)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d

# =========================
# Defaults (tune to Drive)
# =========================

# YOU NEED TO CHANGE HERE ALSO 
DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/rSLDS")
DEFAULT_OUTPUTS_ROOT = Path("/content/drive/MyDrive/sRNN/sRNN-Model-Outputs")


def set_roots(data_root: str | Path = None, outputs_root: str | Path = None):
    """Override default roots at runtime."""
    global DEFAULT_DATA_ROOT, DEFAULT_OUTPUTS_ROOT
    if data_root is not None:
        DEFAULT_DATA_ROOT = Path(data_root)
    if outputs_root is not None:
        DEFAULT_OUTPUTS_ROOT = Path(outputs_root)
    print(f"[paths] DATA_ROOT={DEFAULT_DATA_ROOT}")
    print(f"[paths] OUTPUTS_ROOT={DEFAULT_OUTPUTS_ROOT}")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =========================
# Paths
# =========================

def h5_path_for_rat(rid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    """Expected HDF5 location (10 ms bins, behavior etc.)."""
    root = Path(data_root)
    return root / "Rat-Data-hdf5" / f"Rat{rid}" / "NpxFiringRate_Behavior_SBL_10msBINS_0smoothing.hdf5"


def base_model_dir(rid: int, outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    return Path(outputs_root) / f"Rat{rid}-Model-Outputs"


def run_dir(
    rid: int,
    suffix: str,
    K: int,
    seed: int,
    kappa: float,
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT,
) -> Path:
    name = f"models_Rat{rid}_{suffix}_K{K}_seed{seed}_kappa{kappa:g}"
    return base_model_dir(rid, outputs_root) / name


# =========================
# IO & preprocessing (HDF5-only)
# =========================

def build_footshock_regressor(
    t: np.ndarray,
    shock_times: Optional[np.ndarray],
    *,
    expand_sec: float = 0.0,
) -> np.ndarray:
    """
    Binary regressor with 1 around each shock time on a given time grid t (seconds).
    If expand_sec == 0, mark the single closest sample to each shock.
    If expand_sec > 0, mark all samples within ±expand_sec of each shock.
    """
    v = np.zeros_like(t, dtype=float)
    if shock_times is None or len(shock_times) == 0:
        return v[:, None]
    shocks = np.asarray(shock_times).ravel()
    if expand_sec > 0.0:
        for s in shocks:
            mask = np.abs(t - float(s)) <= float(expand_sec)
            v[mask] = 1.0
    else:
        idx = np.searchsorted(t, shocks)
        idx = np.clip(idx, 0, len(t) - 1)
        v[idx] = 1.0
    return v[:, None]


def _std_nan_robust(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd


def _downsample_10ms_to_target(
    time_10: np.ndarray,
    arr_10: np.ndarray,
    *,
    target_ms: float = 100.0,
    mode: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic downsample from ~10 ms bins to target_ms bins (default 100 ms).

    Parameters
    ----------
    time_10 : (T,) seconds
    arr_10  : (T, N) or (T,) array (will be treated as 2D)
    target_ms : desired bin size in ms
    mode : "mean" or "sum"

    Returns
    -------
    arr_ds : (T_ds, N) downsampled
    time_ds : (T_ds,) corresponding time (mean within each bin)
    """
    time_10 = np.asarray(time_10, float)
    arr_10 = np.asarray(arr_10, float)

    if arr_10.ndim == 1:
        arr_10 = arr_10[:, None]

    T, N = arr_10.shape
    if T != len(time_10):
        raise ValueError(f"Time/array length mismatch: T_time={len(time_10)}, T_arr={T}")

    if T < 2:
        raise ValueError("Not enough samples to infer native dt.")

    dt = float(np.median(np.diff(time_10)))
    native_ms = dt * 1000.0
    if native_ms <= 0:
        raise ValueError(f"Non-positive native_ms from dt={dt}")

    factor = int(round(target_ms / native_ms))
    if factor <= 0:
        raise ValueError(f"Bad factor computed from target_ms={target_ms}, native_ms={native_ms}")

    T_ds = T // factor
    if T_ds == 0:
        raise ValueError("Recording too short for requested target_ms.")

    cut = T_ds * factor
    arr_block = arr_10[:cut].reshape(T_ds, factor, N)
    time_block = time_10[:cut].reshape(T_ds, factor)

    if mode == "mean":
        arr_ds = np.nanmean(arr_block, axis=1)
    elif mode == "sum":
        arr_ds = np.nansum(arr_block, axis=1)
    else:
        raise ValueError(f"Unknown mode={mode}")

    time_ds = np.nanmean(time_block, axis=1)
    arr_ds = np.nan_to_num(arr_ds, nan=0.0, posinf=0.0, neginf=0.0)
    time_ds = np.nan_to_num(time_ds, nan=0.0, posinf=0.0, neginf=0.0)
    return arr_ds, time_ds


# =========================
# DCA-lite / PCA embedding
# =========================

def _time_lagged_projection(X_TN, d=8, lag=1, ridge=1e-6, symmetric=False):
    """
    DCA-lite: dominant predictive components from lag-1 covariance.
    Returns Z:(T,d) and V:(N,d).
    """
    X = np.asarray(X_TN, float)
    X -= X.mean(0, keepdims=True)
    T, N = X.shape
    if T <= lag:
        raise ValueError(f"lag={lag} requires T>{lag} but T={T}")
    C0 = (X.T @ X) / T + ridge * np.eye(N)
    Xp, Xf = X[:-lag], X[lag:]
    C1 = (Xp.T @ Xf) / (T - lag)

    if symmetric:
        M = np.linalg.solve(C0, 0.5 * (C1 + C1.T))
        M = 0.5 * (M + M.T)
        vals, vecs = eigh(M)
        V = vecs[:, np.argsort(vals)[-d:]]
        Q, _ = np.linalg.qr(V)
        V = Q
    else:
        w, Vc0 = eigh(C0)
        w = np.maximum(w, ridge)
        Wm12 = Vc0 @ np.diag(1.0 / np.sqrt(w)) @ Vc0.T
        M = Wm12 @ C1 @ Wm12
        M = 0.5 * (M + M.T)
        vals, vecs = eigh(M)
        V = Wm12 @ vecs[:, np.argsort(vals)[-d:]]
    Z = X @ V
    return Z.astype(np.float32), V.astype(np.float32)


def make_embedding(FR_sec, method="dca1", n_components=8, random_state=None):
    X = _std_nan_robust(FR_sec)
    if method == "pca":
        model = PCA(n_components=n_components, random_state=random_state).fit(X)
        Z = model.transform(X)
        meta = {"method": "pca", "n_components": int(Z.shape[1])}
        return Z.astype(np.float32), meta
    elif method in ("dca1", "dca1_sym"):
        symmetric = (method == "dca1_sym")
        Z, _ = _time_lagged_projection(X, d=n_components, lag=1, ridge=1e-6, symmetric=symmetric)
        return Z.astype(np.float32), {
            "method": method,
            "n_components": int(Z.shape[1]),
            "lag": 1,
        }
    else:
        raise ValueError(f"Unknown DR method: {method}")


# =========================
# Window dataset + split
# =========================

class NeuralWindows(Dataset):
    """Yields (window_x, window_u) with shape (W, d_in) and (W, 1)."""
    def __init__(self, Z_Td: np.ndarray, u_T1: np.ndarray, window: int = 100, stride: int = 1):
        X = np.asarray(Z_Td, np.float32)
        u = np.asarray(u_T1, np.float32)
        if u.ndim == 1:
            u = u[:, None]
        if len(X) != len(u):
            raise ValueError("Z and u lengths differ")
        self.X = torch.from_numpy(X)
        self.u = torch.from_numpy(u)
        self.W = int(window)
        self.stride = max(1, int(stride))
        if len(self.X) < self.W:
            raise ValueError(f"T={len(self.X)} < window={self.W}")

    def __len__(self):
        return 1 + (len(self.X) - self.W) // self.stride

    def __getitem__(self, idx):
        s = idx * self.stride
        e = s + self.W
        x = self.X[s:e, :]
        u = self.u[s:e, :]
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0)
        u = torch.nan_to_num(u, 0.0, 0.0, 0.0)
        return x, u


def make_time_split_indices(T: int, window: int, stride: int, test_frac: float) -> Tuple[List[int], List[int]]:
    """
    Time-aware split: last fraction of windows for test (no leakage).
    Returns lists of training and test window indices.
    """
    num_windows = 1 + (T - window) // stride
    cut_t = int(math.floor((1.0 - test_frac) * T))
    train_idx, test_idx = [], []
    for i in range(num_windows):
        s = i * stride
        e = s + window
        if e <= cut_t:
            train_idx.append(i)
        else:
            test_idx.append(i)
    if len(train_idx) == 0 or len(test_idx) == 0:
        # fallback to 80/20 by index if recording is short
        pivot = int(0.8 * num_windows)
        train_idx = list(range(pivot))
        test_idx = list(range(pivot, num_windows))
    return train_idx, test_idx


def subset_sampler(indices: List[int]) -> torch.utils.data.Sampler:
    return torch.utils.data.SubsetRandomSampler(indices)


# =========================
# Patch SRNN package (TBPTT + optional no-input transitions)
# =========================

def apply_srnn_patches():
    """
    Patches the upstream sRNN implementation:
    - robust inference forward (no NaNs, fixed init)
    - accepts optional h0 and returns final hidden (for TBPTT)
    - generative forward returning log-probs for transitions/dynamics
    - supports ignoring external inputs (purely recurrent)
    """
    try:
        from sRNN.networks import InferenceNetwork as _Inf, GenerativeSRNN as _Gen
    except Exception:
        from networks import InferenceNetwork as _Inf, GenerativeSRNN as _Gen

    def _infer_num_dirs(rnn: nn.RNNBase) -> int:
        return 2 if getattr(rnn, "bidirectional", False) else 1

    # NOTE: signature supports optional h0 and returns (h, q, h_last)
    def _inf_forward_stable(self, x, h0: Optional[torch.Tensor] = None):  # x: (B, T, D)
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0).contiguous()
        try:
            self.rnn.flatten_parameters()
        except Exception:
            pass
        B, T, D = x.shape
        num_dirs = _infer_num_dirs(self.rnn)
        num_layers = getattr(self.rnn, "num_layers", 1)
        hidden_size = getattr(self.rnn, "hidden_size", None)
        if hidden_size is None:
            hs, _ = self.rnn(x[:, :1, :])
            hidden_size = hs.size(-1)
        if h0 is None:
            h0 = torch.zeros(num_layers * num_dirs, B, hidden_size, device=x.device, dtype=x.dtype).contiguous()
        else:
            if h0.dim() != 3:
                raise ValueError("h0 must have shape (num_layers*num_dirs, B, hidden_size)")
        h_seq, h_last = self.rnn(x, h0)
        m = self.fc_mean(h_seq)
        lv = torch.clamp(torch.nan_to_num(self.fc_logvar(h_seq), 0.0, 10.0, -10.0), -8.0, 5.0)
        std = torch.exp(0.5 * lv) + 1e-6
        std = torch.nan_to_num(std, 1e-3, 1.0, 1e-3)
        q = torch.distributions.Independent(torch.distributions.Normal(m, std), 1)
        h = torch.nan_to_num(q.rsample(), 0.0, 0.0, 0.0).contiguous()
        return h, q, h_last

    def _gen_forward(self, x, u, h_samp):
        # Light-weight generator with per-state RNN on x to predict mean h
        B, T, N = x.shape
        K = self.K
        H = self.H
        device = x.device
        x = torch.nan_to_num(x, 0.0, 0.0, 0.0).contiguous()
        u = torch.nan_to_num(u, 0.0, 0.0, 0.0).contiguous()
        h_samp = torch.nan_to_num(h_samp, 0.0, 0.0, 0.0).contiguous()

        if not hasattr(self, "pi0"):
            self.pi0 = nn.Parameter(torch.zeros(K, device=device))
        p0 = F.log_softmax(self.pi0, dim=0).view(1, K).expand(B, K).contiguous()

        # Parametrized transitions A(h_{t-1}, u_t)
        if not hasattr(self, "trans_h"):
            self.trans_h = nn.Linear(H, K * K, bias=True).to(device)
        # create trans_u lazily only when used
        if getattr(self, "use_inputs", False) and not hasattr(self, "trans_u"):
            self.trans_u = nn.Linear(u.size(-1), K * K, bias=False).to(device)

        # K per-state simple RNNs to evolve h
        if not hasattr(self, "rnns") or len(self.rnns) != K:
            self.rnns = nn.ModuleList([nn.RNN(N, H, batch_first=True).to(device) for _ in range(K)])
        for r in self.rnns:
            try:
                r.flatten_parameters()
            except Exception:
                pass

        # Gaussian h likelihood around state-conditioned prediction
        if not hasattr(self, "std_h"):
            self.std_h = 1e-2
        const_h = float(-0.5 * (H * np.log(2 * np.pi * (self.std_h ** 2))))

        p_s = torch.full((B, T, K, K), -1e8, device=device)
        p_h = torch.zeros(B, T, K, device=device)
        eye = torch.eye(K, device=device).view(1, K, K).expand(B, K, K)
        p_s[:, 0] = torch.log(eye + 1e-12)
        p_h[:, 0, :] = const_h

        for t in range(1, T):
            Ah = self.trans_h(h_samp[:, t - 1, :])
            if getattr(self, "use_inputs", False):
                Au = self.trans_u(u[:, t, :])
                A = (Ah + Au).view(B, K, K)
            else:
                A = Ah.view(B, K, K)
            kappa_val = float(getattr(self, "kappa", 0.0))
            if kappa_val != 0.0:
                A = A + torch.eye(K, device=device).view(1, K, K) * kappa_val
            A = A - torch.logsumexp(A, dim=2, keepdim=True)
            p_s[:, t] = A

            x_step = x[:, t - 1:t, :].contiguous()
            h0 = h_samp[:, t - 1, :].contiguous().unsqueeze(0)  # (1,B,H)
            for k in range(K):
                out_k, _ = self.rnns[k](x_step, h0)
                mean_h = out_k[:, 0, :]
                diff = h_samp[:, t, :] - mean_h
                p_h[:, t, k] = const_h - (diff.pow(2).sum(dim=1) / (2 * (self.std_h ** 2)))

        p0 = torch.nan_to_num(p0, -1e-8, -1e8, -1e-8)
        p_s = torch.nan_to_num(p_s, -1e-8, -1e-8, -1e-8)
        p_h = torch.nan_to_num(p_h, -1e-8, -1e-8, -1e-8)
        return p0, p_s, p_h, None, None, None, None, None

    _Inf.forward = _inf_forward_stable
    _Gen.forward = _gen_forward
    return _Inf, _Gen


# =========================
# Config & rSLDS warm-start
# =========================

@dataclass
class TrainConfig:
    # data
    rat_id: int
    data_root: Path = DEFAULT_DATA_ROOT
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT
    subset_name: str = "responsive"
    h5_optional: bool = True

    # DR
    dr_method: str = "dca1"
    dr_n: int = 8
    dr_random_state: Optional[int] = 0

    # model
    K_states: int = 5
    latent_dim: int = 8
    kappa: float = 0.0
    use_inputs: bool = True           # if False, ignore external inputs entirely
  
    # NEW: use only behavior (pupil + speed) as observations
    behavior_only: bool = False

    # training
    num_iters: int = 2000
    warmup_epochs: int = 200
    window_size: int = 100
    stride: int = 1
    batch_size: int = 128
    lr: float = 1e-4
    seed: int = 0
    test_split: float = 0.2
    overwrite: bool = False
    verbose: bool = True
    tbptt_steps: Optional[int] = None

    # regularization
    lambda_entropy: float = 1.0e-3
    lambda_usage: float = 1.0e-2

    # resampling / behavior
    ms_per_sample: Optional[int] = None   # kept for metadata; not used directly now
    rate_mode: str = "mean"
    shock_expand_sec: float = 0.0

    # --- warm-start from rSLDS discrete states ---
    rslds_init_path: Optional[str] = None        # .npy path or template with {rat_id}
    rslds_init_use_head: Optional[int] = 200     # first N samples to estimate pi0 (None = all)
    prediction_horizons: Tuple[int, ...] = (10, 20, 30, 40)


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _resolve_rslds_path(rslds_path_or_template: Optional[str], rat_id: int) -> Optional[str]:
    if not rslds_path_or_template:
        return None
    s = str(rslds_path_or_template)
    if "{rat_id}" in s:
        s = s.format(rat_id=rat_id)
    return s


def _load_rslds_z(path: str, T_target: int, K: int) -> Optional[np.ndarray]:
    if path is None or (isinstance(path, str) and path.strip() == ""):
        return None
    try:
        z = np.load(path)
    except Exception as e:
        warnings.warn(f"[rSLDS init] Could not load {path}: {e}")
        return None
    z = np.asarray(z).ravel()
    if z.size == 0:
        return None
    # make sure labels are in [0, K-1]
    z = np.clip(z.astype(int), 0, K - 1)
    # Align length with training series length (Zz length == T_target after DR)
    if z.size >= T_target:
        z = z[:T_target]
    else:
        z = np.pad(z, (0, T_target - z.size), mode="edge")
    return z


def _estimate_pi0_and_A(z: np.ndarray, K: int, use_head: Optional[int] = 200):
    """Return initial log prior over states and log transition matrix from labels."""
    if use_head is not None:
        z_head = z[:max(1, int(use_head))]
    else:
        z_head = z
    # pi0
    counts = np.bincount(z_head, minlength=K).astype(float) + 1e-3
    pi = counts / counts.sum()
    log_pi = np.log(pi)

    # transitions
    A_counts = np.ones((K, K), float) * 1e-3  # Laplace smoothing
    for a, b in zip(z[:-1], z[1:]):
        A_counts[a, b] += 1.0
    A = A_counts / A_counts.sum(axis=1, keepdims=True)
    logA = np.log(A)
    return log_pi.astype(np.float32), logA.astype(np.float32)


# =========================
# Training / evaluation
# =========================

def fit_srnn_with_split(cfg: TrainConfig) -> Dict:
    device = get_device()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ----- IO -----
    save_suffix = f"{cfg.subset_name}_{cfg.dr_method}{cfg.dr_n}"
    save_dir = run_dir(cfg.rat_id, save_suffix, cfg.K_states, cfg.seed, cfg.kappa, cfg.outputs_root)
    save_dir.mkdir(parents=True, exist_ok=True)
    if any((save_dir / f).exists() for f in ("final_checkpoint.pth", "x_hat.npy")) and not cfg.overwrite:
        return {"status": "skip_exists", "path": str(save_dir)}

    h5p = h5_path_for_rat(cfg.rat_id, cfg.data_root)
    have_h5 = h5p.exists()

    if cfg.verbose:
        print(f"[inputs] using only HDF5 @ {h5p}")
        print(f"[inputs] h5_found={have_h5}")

    if (not have_h5) and (not cfg.h5_optional):
        return {"status": "missing_h5", "msg": str(h5p)}

    if not have_h5:
        return {"status": "missing_h5", "msg": f"HDF5 not found: {h5p}"}

    # ----- load from HDF5 (10 ms native) -----
    with h5py.File(h5p, "r") as h5:
        keys = list(h5.keys())
        if cfg.verbose:
            print(f"[H5] available keys: {keys}")

        if "firing_rates" not in h5:
            raise KeyError(
                "Could not find 'firing_rates' in HDF5. "
                f"Available top-level keys: {keys}"
            )
        FR_10 = h5["firing_rates"][...]  # may be (N, T) or (T, N)
        time_10 = h5["time"][...] if "time" in h5 else np.arange(FR_10.shape[-1]) * 0.01
        shock_times = h5["footshock_times"][...] if "footshock_times" in h5 else None
        pupil_10 = h5["pupil_diameter"][...] if "pupil_diameter" in h5 else None
        speed_10 = h5["speed"][...] if "speed" in h5 else None

    FR_10 = np.asarray(FR_10, float)
    time_10 = np.asarray(time_10, float)

    if cfg.verbose:
        print(f"[H5] FR_10 shape={FR_10.shape}")
        print(f"[H5] time_10 shape={time_10.shape}")
        if shock_times is not None:
            print(f"[H5] footshock_times count={len(shock_times)}")
        if pupil_10 is not None:
            print(f"[H5] pupil_10 shape={np.shape(pupil_10)}")
        if speed_10 is not None:
            print(f"[H5] speed_10 shape={np.shape(speed_10)}")

    # --- fix orientation: ensure FR_10 is (T, N) to match time_10 ---
    if FR_10.shape[0] == len(time_10):
        # already (T, N)
        pass
    elif FR_10.shape[1] == len(time_10):
        # currently (N, T) -> transpose to (T, N)
        FR_10 = FR_10.T
        if cfg.verbose:
            print(f"[H5] transposed FR_10 to (T,N) → {FR_10.shape}")
    else:
        raise ValueError(
            f"[H5] firing_rates shape {FR_10.shape} inconsistent with time length {len(time_10)}"
        )
        

    # ensure 1D for behavior arrays
    def _reshape_1d(x):
        if x is None:
            return None
        x = np.asarray(x, float)
        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1).mean(axis=1)
        return x

    pupil_10 = _reshape_1d(pupil_10)
    speed_10 = _reshape_1d(speed_10)

    # ----- downsample 10 ms -> 100 ms for FR, pupil, speed, time -----
    target_ms = 100.0
    FR_100, time_100 = _downsample_10ms_to_target(
        time_10, FR_10, target_ms=target_ms, mode=cfg.rate_mode
    )
    if pupil_10 is not None:
        pupil_100, _ = _downsample_10ms_to_target(
            time_10, pupil_10, target_ms=target_ms, mode="mean"
        )
    else:
        pupil_100 = None

    if speed_10 is not None:
        speed_100, _ = _downsample_10ms_to_target(
            time_10, speed_10, target_ms=target_ms, mode="mean"
        )
    else:
        speed_100 = None

    if cfg.verbose:
        print(f"[downsample] FR_100 shape={FR_100.shape}, time_100 shape={time_100.shape}")
        if pupil_100 is not None:
            print(f"[downsample] pupil_100 shape={pupil_100.shape}")
        if speed_100 is not None:
            print(f"[downsample] speed_100 shape={speed_100.shape}")

    # ----- build footshock regressor on 100 ms grid -----
    shock_reg_100 = build_footshock_regressor(
        time_100, shock_times, expand_sec=float(cfg.shock_expand_sec)
    )  # (T_100, 1)

    # This is what the model *actually* sees as input:
    if cfg.use_inputs:
        u_model = np.nan_to_num(shock_reg_100.astype(FR_100.dtype), nan=0.0, posinf=0.0, neginf=0.0)
        if cfg.verbose:
            nz = int(np.count_nonzero(u_model))
            print(f"[inputs] use_inputs=True → u_model nonzero_samples={nz}")
    else:
        # fully UNSUPERVISED: no shocks in transitions
        u_model = np.zeros((FR_100.shape[0], 1), dtype=FR_100.dtype)
        if cfg.verbose:
            print(f"[inputs] use_inputs=False → u_model zeroed, shape={u_model.shape}")

    FR_sec = np.nan_to_num(FR_100, nan=0.0, posinf=0.0, neginf=0.0)
    # --- NEW: behavior-only mode (use pupil + speed as observations instead of FR) ---
    if cfg.behavior_only:
        beh_list = []
        if pupil_100 is not None:
            # ensure 2D (T,1)
            if pupil_100.ndim == 1:
                pupil_100_use = pupil_100[:, None]
            else:
                pupil_100_use = pupil_100
            beh_list.append(pupil_100_use.astype(np.float32))

        if speed_100 is not None:
            if speed_100.ndim == 1:
                speed_100_use = speed_100[:, None]
            else:
                speed_100_use = speed_100
            beh_list.append(speed_100_use.astype(np.float32))

        if not beh_list:
            raise ValueError(
                "behavior_only=True but neither pupil_100 nor speed_100 is available in the HDF5 file."
            )

        # (T, n_beh) – observed space is now behavior only
        FR_sec = np.concatenate(beh_list, axis=1)
        if cfg.verbose:
            print(f"[behavior-only] Using behavior as observations: FR_sec shape={FR_sec.shape}")


    # Keep a z-scored copy of full-rate signals for later plots
    mu_full, sd_full = FR_sec.mean(0, keepdims=True), FR_sec.std(0, keepdims=True)
    sd_full[sd_full == 0.0] = 1.0
    FRz_full = np.nan_to_num((FR_sec - mu_full) / sd_full, 0.0, 0.0, 0.0)


    # ----- DR -----
    Z_raw, dr_meta = make_embedding(
        FR_sec, method=cfg.dr_method, n_components=cfg.dr_n, random_state=cfg.dr_random_state
    )
    Zz = StandardScaler().fit_transform(Z_raw).astype(np.float32)  # (T, d_in)
    d_in = Zz.shape[1]
    if cfg.verbose:
        print(f"[DR] {dr_meta}  → d_in={d_in}")

    # ----- windows + split -----
    T = Zz.shape[0]
    num_windows = 1 + (T - cfg.window_size) // max(1, cfg.stride)
    
    # --- FIX: disable split completely when test_split == 0 ---
    if cfg.test_split == 0.0:
        train_idx = list(range(num_windows))   # ALL windows
        test_idx = []                          # NO test windows
    else:
        train_idx, test_idx = make_time_split_indices(
            T, cfg.window_size, cfg.stride, cfg.test_split
        )
    
    ds = NeuralWindows(Zz, u_model, window=cfg.window_size, stride=cfg.stride)


    n_train = len(train_idx)
    n_test = len(test_idx)
    train_bs = max(1, min(cfg.batch_size, n_train))
    test_bs = max(1, min(cfg.batch_size, n_test))

    if cfg.verbose:
        num_windows = 1 + (T - cfg.window_size) // max(1, cfg.stride) if T >= cfg.window_size else 0
        print(
            f"[split] T={T}, windows≈{num_windows}, train={n_train}, test={n_test}, "
            f"train_bs={train_bs}, test_bs={test_bs}"
        )

    if n_train == 0:
        warnings.warn(
            f"No training windows. T={T}, window={cfg.window_size}, stride={cfg.stride}. "
            f"Reduce window_size/stride or ensure longer data."
        )
        return {
            "status": "no_train_windows",
            "T": int(T),
            "window": int(cfg.window_size),
            "stride": int(cfg.stride),
        }

    train_loader = DataLoader(
        ds,
        batch_size=train_bs,
        shuffle=False,
        drop_last=False,
        sampler=subset_sampler(train_idx),
        pin_memory=(device == "cuda"),
        num_workers=max(0, (os.cpu_count() or 2) // 2),
    )
    test_loader = DataLoader(
        ds,
        batch_size=test_bs,
        shuffle=False,
        drop_last=False,
        sampler=subset_sampler(test_idx),
        pin_memory=(device == "cuda"),
        num_workers=max(0, (os.cpu_count() or 2) // 2),
    )

    # ----- SRNN -----
    _Inf, _Gen = apply_srnn_patches()
    infer = _Inf(input_dim=d_in, hidden_dim=cfg.latent_dim).to(device)
    gen = _Gen(D=d_in, K=cfg.K_states, H=cfg.latent_dim).to(device)
    gen.kappa = float(cfg.kappa)
    gen.use_inputs = bool(cfg.use_inputs)  # set flag for generator
    infer.apply(_init_weights)
    gen.apply(_init_weights)

    # ----- OPTIONAL: warm-start from rSLDS z_hat -----
    rslds_path = _resolve_rslds_path(cfg.rslds_init_path, cfg.rat_id)
    if rslds_path:
        z_init = _load_rslds_z(rslds_path, T_target=Zz.shape[0], K=cfg.K_states)
        if z_init is not None:
            log_pi, logA = _estimate_pi0_and_A(z_init, cfg.K_states, cfg.rslds_init_use_head)
            if not hasattr(gen, "pi0"):
                gen.pi0 = nn.Parameter(torch.zeros(cfg.K_states, device=device))
            if not hasattr(gen, "trans_h"):
                gen.trans_h = nn.Linear(cfg.latent_dim, cfg.K_states * cfg.K_states, bias=True).to(device)
            if getattr(gen, "use_inputs", False) and not hasattr(gen, "trans_u"):
                gen.trans_u = nn.Linear(1, cfg.K_states * cfg.K_states, bias=False).to(device)
            with torch.no_grad():
                gen.pi0.data = torch.from_numpy(log_pi).to(device)
                nn.init.constant_(gen.trans_h.weight, 0.0)
                nn.init.constant_(gen.trans_h.bias, 0.0)
                if getattr(gen, "use_inputs", False) and gen.trans_u.weight is not None:
                    nn.init.constant_(gen.trans_u.weight, 0.0)
                gen.trans_h.bias.copy_(torch.from_numpy(logA.reshape(-1)).to(device))
            if cfg.verbose:
                print(f"[rSLDS init] Warm-started from {rslds_path}")
        else:
            if cfg.verbose:
                print(f"[rSLDS init] Skipping warm-start (failed to load or empty): {rslds_path}")

    params = list(infer.parameters()) + list(gen.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_iters, eta_min=cfg.lr * 0.1)

    # ----- helpers (TBPTT-aware) -----
    def _batch_step_chunk(x_chunk, u_chunk, warmup=True, h0: Optional[torch.Tensor] = None):
        """One truncated chunk: returns (loss, elbo, h_last_detached)."""
        h_samp, q_dist, h_last = infer(x_chunk, h0=h0)
        p0, p_s, p_h, *_ = gen(x_chunk, u_chunk, h_samp)

        log_q = q_dist.log_prob(h_samp).sum(dim=1)                      # (B,)
        elbo = (p0.sum(1) + p_s.sum((1, 2, 3)) + p_h.sum((1, 2)) - log_q).mean()

        trans_probs = torch.exp(p_s).clamp_min(1e-12)
        entropy_z = -(trans_probs * p_s).sum(dim=(1, 2, 3)).mean()

        loss = -elbo - cfg.lambda_entropy * entropy_z

        if warmup:
            with torch.no_grad():
                post = torch.softmax(p_s, dim=-1)                       # (B,T,K,K)
                marg = post.mean(dim=(0, 1))                            # (K,K)
                usage = marg.sum(dim=1) / marg.sum()                    # (K,)
            target = torch.full_like(usage, 1.0 / usage.numel())
            usage_loss = F.kl_div(torch.log(usage + 1e-12), target, reduction="sum")
            loss = loss + cfg.lambda_usage * usage_loss

        return loss, elbo.detach(), h_last.detach()
        # ----- train (TBPTT if cfg.tbptt_steps is set) -----
    losses, elbos = [], []
    tbptt = cfg.tbptt_steps if (cfg.tbptt_steps and cfg.tbptt_steps > 0) else None

    for epoch in range(1, cfg.num_iters + 1):
        infer.train()
        gen.train()

        # warm-up LR
        base_lr = cfg.lr
        for g in opt.param_groups:
            g["lr"] = (base_lr * epoch / 10.0) if epoch <= 10 else g.get("lr", base_lr)

        total = 0.0
        total_elbo = 0.0
        nb = 0

        # === tqdm progress bar ===
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_iters}", leave=False)

        for xb, ub in pbar:
            x = xb.to(device, non_blocking=True)
            u = ub.to(device, non_blocking=True)
            warm = (epoch <= cfg.warmup_epochs)

            if (tbptt is None) or (tbptt >= x.size(1)):
                opt.zero_grad()
                loss, elbo, _ = _batch_step_chunk(x, u, warmup=warm, h0=None)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()

                total += float(loss.item())
                total_elbo += float(elbo.item())
                nb += 1

            else:
                # --- TBPTT ---
                B, Wtot, _ = x.shape
                h_carry = None

                for t0 in range(0, Wtot, tbptt):
                    t1 = min(Wtot, t0 + tbptt)
                    x_chunk = x[:, t0:t1, :]
                    u_chunk = u[:, t0:t1, :]

                    opt.zero_grad()
                    loss, elbo, h_carry = _batch_step_chunk(x_chunk, u_chunk, warmup=warm, h0=h_carry)

                    if torch.isnan(loss):
                        h_carry = None
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    opt.step()

                    h_carry = h_carry.detach()

                    total += float(loss.item())
                    total_elbo += float(elbo.item())
                    nb += 1

            # update progress bar
            if nb > 0:
                pbar.set_postfix({"loss": total / nb, "elbo": total_elbo / nb})

        if nb == 0:
            warnings.warn("No batches were processed. Check window/stride and dataset length.")
            break

        losses.append(total / nb)
        elbos.append(total_elbo / nb)
        scheduler.step()

        if cfg.verbose and (epoch % max(1, cfg.num_iters // 20) == 0 or epoch == 1):
            print(f"[{save_dir.name}] epoch {epoch}/{cfg.num_iters}  loss={losses[-1]:.4f}  elbo={elbos[-1]:.4f}")

        if epoch % max(1, cfg.num_iters // 6) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "infer_state_dict": infer.state_dict(),
                    "gen_state_dict": gen.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss_history": losses,
                    "elbo_history": elbos,
                    "cfg": _cfg_to_jsonable(cfg),
                },
                save_dir / f"epoch_{epoch}.pth",
            )


        # ----- train (TBPTT if cfg.tbptt_steps is set) -----
    

    # ----- inference snapshot on train loader (for usage/dwell) -----
    infer.eval()
    gen.eval()
    with torch.no_grad():
        all_h, all_z = [], []
        for xb, ub in train_loader:
            x = xb.to(device, non_blocking=True)
            u = ub.to(device, non_blocking=True)
            h_samp, _, _ = infer(x, h0=None)
            p0, p_s, p_h, *_ = gen(x, u, h_samp)
            z_hat = torch.argmax(torch.logsumexp(p_s, dim=-1), dim=-1)  # (B,T)
            all_h.append(h_samp.cpu().numpy())
            all_z.append(z_hat.cpu().numpy())
        if all_h:
            XH = np.concatenate(all_h, axis=0)
            ZH = np.concatenate(all_z, axis=0)
        else:
            XH = np.zeros((0, cfg.window_size, cfg.latent_dim), dtype=np.float32)
            ZH = np.zeros((0, cfg.window_size), dtype=np.int64)

    # ----- state usage & dwell -----
    usage = {}
    dwell_mean = dwell_median = np.nan
    if ZH.size:
        z_flat = ZH.reshape(-1)
        K_det = int(z_flat.max()) + 1 if z_flat.size else cfg.K_states
        for k in range(K_det):
            usage[f"p_state{k}"] = float(np.mean(z_flat == k))
        runs, cur, run = [], None, 0
        for v in z_flat:
            v = int(v)
            if cur is None or v == cur:
                run += 1
                cur = v
            else:
                runs.append(run)
                cur = v
                run = 1
        if run > 0:
            runs.append(run)
        if runs:
            dwell_mean = float(np.mean(runs))
            dwell_median = float(np.median(runs))

    # ----- post-hoc linear decoder h->Z (train on train windows) -----
    H_list, Z_list = [], []
    with torch.no_grad():
        for xb, ub in train_loader:
            x = xb.to(device, non_blocking=True)
            u = ub.to(device, non_blocking=True)
            h_samp, _, _ = infer(x, h0=None)       # (B,W,H)
            H_list.append(h_samp.cpu().numpy().reshape(-1, cfg.latent_dim))
            Z_list.append(x.cpu().numpy().reshape(-1, d_in))
    if H_list:
        H_mat = np.concatenate(H_list, axis=0)           # (M,H)
        Z_mat = np.concatenate(Z_list, axis=0)           # (M,d_in)
        lam = 1e-3
        H_aug = np.concatenate([H_mat, np.ones((H_mat.shape[0], 1))], axis=1)  # bias
        W_full = np.linalg.lstsq(
            H_aug.T @ H_aug + lam * np.eye(cfg.latent_dim + 1),
            H_aug.T @ Z_mat,
            rcond=None,
        )[0]               # ((H+1), d_in)
        W = W_full[:-1, :]
        b = W_full[-1:, :]
    else:
        W = np.zeros((cfg.latent_dim, d_in), np.float32)
        b = np.zeros((1, d_in), np.float32)

    # ----- K-step prediction MSE on test windows -----
    def k_step_pred_mse(loader, horizons: Sequence[int]):
        mses = {int(k): [] for k in horizons}
        with torch.no_grad():
            for xb, ub in loader:
                x = xb.to(device, non_blocking=True)  # (B,W,d_in)
                u = ub.to(device, non_blocking=True)
                B, Ww, Din = x.shape
                h_samp, _, _ = infer(x, h0=None)      # (B,W,H)
                for Kpred in horizons:
                    if Kpred >= Ww:
                        continue
                    t0 = Ww - Kpred - 1
                    h_t = h_samp[:, t0, :]
                    x_t = x[:, t0, :].contiguous()
                    preds = []
                    for t in range(t0 + 1, t0 + 1 + Kpred):
                        Ah = gen.trans_h(h_t)
                        if getattr(gen, "use_inputs", False):
                            Au = gen.trans_u(u[:, t, :])
                            A = (Ah + Au).view(B, gen.K, gen.K)
                        else:
                            A = Ah.view(B, gen.K, gen.K)
                        A = A - torch.logsumexp(A, dim=2, keepdim=True)
                        z_next = torch.argmax(A.mean(dim=1), dim=1)
                        out_list = []
                        for k in range(gen.K):
                            mask = (z_next == k)
                            if mask.any():
                                x_step = x_t[mask].unsqueeze(1)
                                h0 = h_t[mask].unsqueeze(0)
                                out_k, _ = gen.rnns[k](x_step, h0)
                                out_list.append((mask, out_k[:, 0, :]))
                        h_next = torch.zeros_like(h_t)
                        for mask, val in out_list:
                            h_next[mask] = val
                        z_pred = h_next @ torch.from_numpy(W).to(h_next) + torch.from_numpy(b).to(h_next)
                        preds.append(z_pred)
                        h_t = h_next
                        x_t = z_pred
                    if preds:
                        Zpred = torch.stack(preds, dim=1)
                        Ztrue = x[:, t0 + 1 : t0 + 1 + Kpred, :]
                        mse = torch.mean((Zpred - Ztrue) ** 2, dim=(0, 1, 2)).item()
                        mses[int(Kpred)].append(mse)
        return {k: (float(np.mean(v)) if len(v) else np.nan) for k, v in mses.items()}

    pred_mse_test = k_step_pred_mse(test_loader, cfg.prediction_horizons)

    # ----- save artifacts -----
    np.save(save_dir / "elbos.npy", np.array(elbos, dtype=np.float32))
    np.save(save_dir / "losses.npy", np.array(losses, dtype=np.float32))
    np.save(save_dir / "x_hat.npy", XH)            # inferred h on train windows
    np.save(save_dir / "z_hat.npy", ZH)            # hard states on train windows
    np.save(save_dir / "FR_z.npy", FRz_full)       # z-scored full signal
    np.save(save_dir / "X_dr.npy", Zz)             # reduced signal (full T)

    # very important: save TRUE shock regressor (not u_model) + behavior & time
    np.save(save_dir / "footshock.npy", shock_reg_100)
    np.save(save_dir / "time_100ms.npy", time_100)
    if pupil_100 is not None:
        np.save(save_dir / "pupil_100ms.npy", pupil_100)
    if speed_100 is not None:
        np.save(save_dir / "speed_100ms.npy", speed_100)

    with open(save_dir / "dr_meta.json", "w") as f:
        json.dump(dr_meta, f, indent=2)

    with open(save_dir / "train_config.json", "w") as f:
        json.dump(_cfg_to_jsonable(cfg), f, indent=2)

    with open(save_dir / "prediction_mse.json", "w") as f:
        json.dump({str(k): v for k, v in pred_mse_test.items()}, f, indent=2)

    # quick ELBO plot
    plt.figure()
    plt.plot(-np.array(losses), label="ELBO (approx)")
    plt.xlabel("epoch")
    plt.ylabel("ELBO ↑")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "elbo.png")
    plt.close()

    # usage bar
    if usage:
        plt.figure()
        keys = sorted(usage.keys(), key=lambda s: int(s.replace("p_state", "")))
        vals = [usage[k] for k in keys]
        plt.bar(keys, vals)
        plt.ylabel("state usage")
        plt.tight_layout()
        plt.savefig(save_dir / "state_usage.png")
        plt.close()

    # finalize checkpoint
    torch.save(
    {
        "epoch": cfg.num_iters,
        "infer_state_dict": infer.state_dict(),
        "gen_state_dict": gen.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "loss_history": losses,
        "elbo_history": elbos,
        "cfg": _cfg_to_jsonable(cfg),
    },
    save_dir / "final_checkpoint.pth",
)


    out = {
        "status": "ok",
        "path": str(save_dir),
        "usage": usage,
        "dwell_mean": dwell_mean,
        "dwell_median": dwell_median,
        "pred_mse_test": pred_mse_test,
    }
    if cfg.verbose:
        print(f"✓ saved → {save_dir}")
        print(f"   usage={usage}  dwell_mean={dwell_mean:.1f}  pred_mse={pred_mse_test}")
    return out


# =========================
# Batch helpers (optional)
# =========================

def run_kappa_sweep(*,
                    rat: int,
                    data_root: str | Path,
                    outputs_root: str | Path,
                    K: int = 3,
                    seed: int = 0,
                    kappa_grid=(0.0, 0.5, 1.0, 2.0),
                    dr_method="dca1",
                    dr_n=8,
                    latent_dim=8,
                    num_iters=1000,
                    warmup_epochs=200,
                    window_size=100,
                    batch_size=128,
                    lr=1e-4,
                    overwrite=True,
                    verbose=True,
                    subset="responsive",
                    rslds_init_path: Optional[str] = None,
                    rslds_init_use_head: Optional[int] = 200,
                    ms_per_sample: Optional[int] = None,
                    rate_mode: str = "mean",
                    shock_expand_sec: float = 0.0,
                    prediction_horizons=(10, 20, 30, 40),
                    tbptt_steps: Optional[int] = None,
                    use_inputs: bool = True):
    results = []
    for kappa in kappa_grid:
        cfg = TrainConfig(
            rat_id=rat,
            data_root=Path(data_root),
            outputs_root=Path(outputs_root),
            subset_name=subset,
            dr_method=dr_method,
            dr_n=dr_n,
            dr_random_state=seed,
            K_states=K,
            latent_dim=latent_dim,
            kappa=kappa,
            num_iters=num_iters,
            warmup_epochs=warmup_epochs,
            window_size=window_size,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            overwrite=overwrite,
            verbose=verbose,
            rslds_init_path=rslds_init_path,
            rslds_init_use_head=rslds_init_use_head,
            ms_per_sample=ms_per_sample,
            rate_mode=rate_mode,
            shock_expand_sec=shock_expand_sec,
            prediction_horizons=tuple(prediction_horizons),
            tbptt_steps=tbptt_steps,
            use_inputs=use_inputs,
        )
        res = fit_srnn_with_split(cfg)
        results.append({"kappa": kappa, **res})
    return results


# =========================
# CLI (YAML)
# =========================

if __name__ == "__main__":
    import argparse, yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    args = p.parse_args()

    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)

    # map YAML to TrainConfig fields (with safe defaults)
    exp = conf.get("experiment", {})
    data = conf.get("data", {})
    dr = conf.get("dr", {})
    model = conf.get("model", {})
    train = conf.get("training", {})
    reg = conf.get("initialization", {})
    eval_blk = conf.get("evaluation", {})
    rslds_blk = conf.get("rslds_init", {})  # optional block

    rats = data.get("rat_id", 0)
    rat_ids: List[int] = [int(r) for r in (rats if isinstance(rats, (list, tuple)) else [rats])]

    horizons = eval_blk.get("prediction_horizons", [10, 20, 30, 40])

    rslds_path = rslds_blk.get("path", conf.get("rslds_init_path", None))
    rslds_use_head = rslds_blk.get("use_head", reg.get("rslds_init_use_head", 200))

    for rid in rat_ids:
            cfg = TrainConfig(
            rat_id=rid,
            data_root=Path(data.get("data_root", str(DEFAULT_DATA_ROOT))),
            outputs_root=Path(data.get("outputs_root", str(DEFAULT_OUTPUTS_ROOT))),
            subset_name=str(data.get("subset", "responsive")),
            h5_optional=bool(data.get("h5_optional", True)),
            dr_method=str(dr.get("method", "dca1")),
            dr_n=int(dr.get("n_components", 8)),
            dr_random_state=dr.get("random_state", exp.get("seed", 0)),
            K_states=int(model.get("K_states", 5)),
            latent_dim=int(model.get("latent_dim", 8)),
            kappa=float(model.get("kappa_values", [model.get("kappa", 0.0)])[0]) if "kappa_values" in model else float(model.get("kappa", 0.0)),
            num_iters=int(train.get("num_iters", 2000)),
            warmup_epochs=int(reg.get("warmup_epochs", 200)),
            window_size=int(train.get("window_size", 100)),
            stride=int(train.get("stride", 1)),
            batch_size=int(train.get("batch_size", 128)),
            lr=float(train.get("lr", 1e-4)),
            seed=int(exp.get("seed", 0)),
            test_split=float(train.get("test_split", 0.2)),
            overwrite=bool(train.get("overwrite", False)),
            verbose=bool(train.get("verbose", True)),
            lambda_entropy=float(reg.get("lambda_entropy", 1e-3)),
            lambda_usage=float(reg.get("lambda_usage", 1e-2)),
            ms_per_sample=train.get("ms_per_sample", None),
            rate_mode=str(train.get("rate_mode", "mean")),
            shock_expand_sec=float(train.get("shock_expand_sec", 0.0)),
            rslds_init_path=_resolve_rslds_path(rslds_path, rid),
            rslds_init_use_head=rslds_use_head,
            prediction_horizons=tuple(int(h) for h in horizons),
            tbptt_steps=int(train.get("tbptt_steps")) if ("tbptt_steps" in train and train.get("tbptt_steps") is not None) else None,
            use_inputs=bool(model.get("use_inputs", True)),
            # NEW:
            behavior_only=bool(model.get("behavior_only", False)),
        )

        out = fit_srnn_with_split(cfg)
        print(json.dumps({"rat_id": rid, **out}, indent=2))
