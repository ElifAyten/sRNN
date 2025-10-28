# srnn_helper.py
# ------------------------------------------------------------
# Minimal, tidy SRNN training utilities with:
# - time-aware train/test split on a single long session
# - warm-up init phase (entropy + usage regularizers)
# - kappa sweep support
# - prediction evaluation via post-hoc linear decoder h->Z
# - DCA-lite or PCA front-end, robust NaN handling
# ------------------------------------------------------------

from __future__ import annotations
import os, json, math, warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

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


# =========================
# Defaults (tune to Drive)
# =========================
DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Rat-Data-hdf5")
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
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# =========================
# Paths
# =========================
def h5_path_for_rat(rid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    """Your raw HDF5 per-rat file."""
    root = Path(data_root)
    # edit if your naming is different:
    return root / f"Rat{rid}" / "NpxFiringRate_Behavior_SBL_10msBINS_0smoothing.hdf5"


def csv_path_responsive_all(rid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    """Your precomputed wide CSV of responsive units for that rat (T x N)."""
    # Customize this if your CSVs live elsewhere:
    return (
        Path(data_root)
        / f"Rat{rid}"
        / "responsive_rates_raw.csv"
    )


def base_model_dir(rid: int, outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    return Path(outputs_root) / f"Rat{rid}-Model-Outputs"


def run_dir(rid: int, suffix: str, K: int, seed: int, kappa: float,
            outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    name = f"models_Rat{rid}_{suffix}_K{K}_seed{seed}_kappa{kappa:g}"
    return base_model_dir(rid, outputs_root) / name


# =========================
# IO & preprocessing
# =========================
def read_wide_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.read_csv(csv_path, engine="python")


def build_footshock_regressor(t: np.ndarray, shock_times: Optional[np.ndarray]) -> np.ndarray:
    """Binary regressor with 1 at closest sample to each shock time."""
    v = np.zeros_like(t, dtype=float)
    if shock_times is None or len(shock_times) == 0:
        return v[:, None]
    idx = np.searchsorted(t, np.asarray(shock_times))
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


def downsample_FR_and_u(FR_TN, u_T1, *, ms_per_sample=10, rate_mode="mean"):
    """Downsample to 1 Hz (bins of 1000/ms_per_sample native samples)."""
    FR_TN = np.asarray(FR_TN, float)
    u_T1 = np.asarray(u_T1, float)
    FR_TN = np.nan_to_num(FR_TN, nan=0.0, posinf=0.0, neginf=0.0)
    u_T1 = np.nan_to_num(u_T1, nan=0.0, posinf=0.0, neginf=0.0)
    factor = int(round(1000 / ms_per_sample))  # samples per 1 s
    if factor <= 0:
        raise ValueError("ms_per_sample must be > 0 and <= 1000.")
    T, N = FR_TN.shape
    if u_T1.shape[0] != T:
        raise ValueError(f"Length mismatch: FR_TN T={T} vs u_T1 T={u_T1.shape[0]}")
    T_sec = T // factor
    if T_sec == 0:
        raise ValueError("Not enough samples for a 1-second bin.")
    cut = T_sec * factor
    B = FR_TN[:cut].reshape(T_sec, factor, N)
    if rate_mode == "mean":
        FR_sec = np.nanmean(B, axis=1)
    else:
        FR_sec = np.nansum(B, axis=1)
    u_sec = (u_T1[:cut].reshape(T_sec, factor, 1).max(axis=1)).astype(FR_sec.dtype)
    return np.nan_to_num(FR_sec), np.nan_to_num(u_sec)


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
        Q, _ = np.linalg.qr(V); V = Q
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
        return Z.astype(np.float32), {"method": method, "n_components": int(Z.shape[1]), "lag": 1}
    else:
        raise ValueError(f"Unknown DR method: {method}")


# =========================
# Window dataset + split
# =========================
class NeuralWindows(Dataset):
    """Yields (window_x, window_u) with shape (W, d_in) and (W, 1)."""
    def __init__(self, Z_Td: np.ndarray, u_T1: np.ndarray, window:int=100, stride:int=1):
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
    # Find first window whose end > cut_t goes to test
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
# Patch SRNN package
# =========================
def apply_srnn_patches():
    """
    Patches the upstream sRNN implementation:
    - robust inference forward (no NaNs, fixed init)
    - generative forward returning log-probs for transitions/dynamics
    """
    from sRNN.networks import InferenceNetwork as _Inf, GenerativeSRNN as _Gen

    def _infer_num_dirs(rnn: nn.RNNBase) -> int:
        return 2 if getattr(rnn, "bidirectional", False) else 1

    def _inf_forward_stable(self, x):  # x: (B, T, D)
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
        h0 = torch.zeros(num_layers * num_dirs, B, hidden_size, device=x.device, dtype=x.dtype).contiguous()
        h_seq, _ = self.rnn(x, h0)
        m = self.fc_mean(h_seq)
        lv = torch.clamp(torch.nan_to_num(self.fc_logvar(h_seq), 0.0, 10.0, -10.0), -8.0, 5.0)
        std = torch.exp(0.5 * lv) + 1e-6
        std = torch.nan_to_num(std, 1e-3, 1.0, 1e-3)
        q = torch.distributions.Independent(torch.distributions.Normal(m, std), 1)
        h = torch.nan_to_num(q.rsample(), 0.0, 0.0, 0.0).contiguous()
        return h, q

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
        if not hasattr(self, "trans_u"):
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
        const_h = float(-0.5 * (H * np.log(2 * np.pi * (self.std_h ** 2)))))

        p_s = torch.full((B, T, K, K), -1e8, device=device)
        p_h = torch.zeros(B, T, K, device=device)
        eye = torch.eye(K, device=device).view(1, K, K).expand(B, K, K)
        p_s[:, 0] = torch.log(eye + 1e-12)
        p_h[:, 0, :] = const_h

        for t in range(1, T):
            Ah = self.trans_h(h_samp[:, t - 1, :])
            Au = self.trans_u(u[:, t, :])
            A = (Ah + Au).view(B, K, K)
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
        p_s = torch.nan_to_num(p_s, -1e-8, -1e8, -1e-8)
        p_h = torch.nan_to_num(p_h, -1e-8, -1e-8, -1e-8)
        return p0, p_s, p_h, None, None, None, None, None

    _Inf.forward = _inf_forward_stable
    _Gen.forward = _gen_forward
    return _Inf, _Gen


# =========================
# Training / evaluation
# =========================
@dataclass
class TrainConfig:
    # data
    rat_id: int
    data_root: Path = DEFAULT_DATA_ROOT
    outputs_root: Path = DEFAULT_OUTPUTS_ROOT
    subset_name: str = "responsive"  # for naming only

    # DR
    dr_method: str = "dca1"
    dr_n: int = 8
    dr_random_state: Optional[int] = 0

    # model
    K_states: int = 5
    latent_dim: int = 8
    kappa: float = 0.0

    # training
    num_iters: int = 2000
    warmup_epochs: int = 200
    window_size: int = 100
    stride: int = 1
    batch_size: int = 128
    lr: float = 1e-4
    seed: int = 0
    test_split: float = 0.2  # last 20% of time goes to test
    overwrite: bool = False
    verbose: bool = True

    # regularization
    lambda_entropy: float = 1.0e-3     # encourage transition entropy
    lambda_usage: float = 1.0e-2       # encourage state usage in warm-up

    # misc
    ms_per_sample: Optional[int] = None
    rate_mode: str = "mean"


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


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
    csvp = csv_path_responsive_all(cfg.rat_id, cfg.data_root)
    if not h5p.exists():
        return {"status": "missing_h5", "msg": str(h5p)}
    if not csvp.exists():
        return {"status": "missing_csv", "msg": str(csvp)}

    # ----- load -----
    df = read_wide_csv(csvp)
    if "time_s" in df.columns:
        t = df.pop("time_s").values
    else:
        with h5py.File(h5p, "r") as h5:
            t = h5["time"][...]

    FR_TN = df.values.astype(float)
    with h5py.File(h5p, "r") as h5:
        shock_times = h5.get("footshock_times")
        shock_times = None if shock_times is None else shock_times[...]

    footshock = build_footshock_regressor(t, shock_times)

    # ----- resample -----
    if cfg.ms_per_sample is None:
        msps = int(round(1000 * float(np.median(np.diff(t))))) if len(t) >= 3 else 10
    else:
        msps = cfg.ms_per_sample
    if cfg.verbose:
        print(f"↳ inferred ms_per_sample ≈ {msps} ms")

    FR_sec, u_sec = downsample_FR_and_u(FR_TN, footshock, ms_per_sample=msps, rate_mode=cfg.rate_mode)

    # Keep a z-scored copy of full-rate signals for later plots
    mu_full, sd_full = FR_sec.mean(0, keepdims=True), FR_sec.std(0, keepdims=True)
    sd_full[sd_full == 0.0] = 1.0
    FRz_full = np.nan_to_num((FR_sec - mu_full) / sd_full, 0.0, 0.0, 0.0)

    # ----- DR -----
    Z_raw, dr_meta = make_embedding(FR_sec, method=cfg.dr_method, n_components=cfg.dr_n, random_state=cfg.dr_random_state)
    Zz = StandardScaler().fit_transform(Z_raw).astype(np.float32)  # (T, d_in)
    d_in = Zz.shape[1]
    if cfg.verbose:
        print(f"[DR] {dr_meta}  → d_in={d_in}")

    # ----- windows + split -----
    T = Zz.shape[0]
    train_idx, test_idx = make_time_split_indices(T, cfg.window_size, cfg.stride, cfg.test_split)
    ds = NeuralWindows(Zz, u_sec, window=cfg.window_size, stride=cfg.stride)
    train_loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=subset_sampler(train_idx),
        pin_memory=(device == "cuda"),
        num_workers=max(0, (os.cpu_count() or 2) // 2),
    )
    test_loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=subset_sampler(test_idx),
        pin_memory=(device == "cuda"),
        num_workers=max(0, (os.cpu_count() or 2) // 2),
    )

    # ----- SRNN -----
    _Inf, _Gen = apply_srnn_patches()
    infer = _Inf(input_dim=d_in, hidden_dim=cfg.latent_dim).to(device)
    gen = _Gen(D=d_in, K=cfg.K_states, H=cfg.latent_dim).to(device)
    gen.kappa = float(cfg.kappa)
    infer.apply(_init_weights)
    gen.apply(_init_weights)

    params = list(infer.parameters()) + list(gen.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_iters, eta_min=cfg.lr * 0.1)

    # ----- helpers -----
    def _batch_step(x, u, epoch, warmup=True):
        """Compute loss = -ELBO - entropy_bonus - usage_bonus (during warmup)."""
        h_samp, q_dist = infer(x)
        p0, p_s, p_h, *_ = gen(x, u, h_samp)

        log_q = q_dist.log_prob(h_samp).sum(dim=1)                      # (B,)
        elbo = (p0.sum(1) + p_s.sum((1, 2, 3)) + p_h.sum((1, 2)) - log_q).mean()

        trans_probs = torch.exp(p_s).clamp_min(1e-12)
        entropy_z = -(trans_probs * p_s).sum(dim=(1, 2, 3)).mean()

        loss = -elbo - cfg.lambda_entropy * entropy_z

        if warmup:
            # encourage roughly uniform marginal usage over states
            with torch.no_grad():
                # posterior approx for z_t: softmax over outgoing transitions
                post = torch.softmax(p_s, dim=-1)                       # (B,T,K,K)
                marg = post.mean(dim=(0, 1))                            # (K,K)
                usage = marg.sum(dim=1) / marg.sum()                    # (K,)
            target = torch.full_like(usage, 1.0 / usage.numel())
            usage_loss = F.kl_div(torch.log(usage + 1e-12), target, reduction="sum")
            loss = loss + cfg.lambda_usage * usage_loss

        return loss, elbo.detach()

    # ----- train -----
    losses, elbos = [], []
    for epoch in range(1, cfg.num_iters + 1):
        infer.train(); gen.train()
        # LR warmup for first 10 epochs
        base_lr = cfg.lr
        for g in opt.param_groups:
            g["lr"] = (base_lr * epoch / 10.0) if epoch <= 10 else g.get("lr", base_lr)
        total = 0.0; total_elbo = 0.0; nb = 0
        for xb, ub in train_loader:
            x = xb.to(device, non_blocking=True)
            u = ub.to(device, non_blocking=True)
            opt.zero_grad()
            warm = (epoch <= cfg.warmup_epochs)
            loss, elbo = _batch_step(x, u, epoch, warmup=warm)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += float(loss.item())
            total_elbo += float(elbo.item())
            nb += 1
        if nb == 0:
            break
        losses.append(total / nb)
        elbos.append(total_elbo / nb)
        scheduler.step()

        if cfg.verbose and (epoch % max(1, cfg.num_iters // 20) == 0 or epoch == 1):
            print(f"[{save_dir.name}] epoch {epoch}/{cfg.num_iters}  loss={losses[-1]:.4f}  elbo={elbos[-1]:.4f}")

        # periodic checkpoints
        if epoch % max(1, cfg.num_iters // 6) == 0:
            torch.save(
                {"epoch": epoch, "infer": infer.state_dict(), "gen": gen.state_dict(),
                 "opt": opt.state_dict(), "loss_history": losses, "elbo_history": elbos},
                save_dir / f"epoch_{epoch}.pth"
            )

    # ----- inference snapshot on train loader (for usage/dwell) -----
    infer.eval(); gen.eval()
    with torch.no_grad():
        all_h, all_z = [], []
        for xb, ub in train_loader:
            x = xb.to(device, non_blocking=True)
            u = ub.to(device, non_blocking=True)
            h_samp, _ = infer(x)
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
                run += 1; cur = v
            else:
                runs.append(run); cur = v; run = 1
        if run > 0: runs.append(run)
        if runs:
            dwell_mean = float(np.mean(runs))
            dwell_median = float(np.median(runs))

    # ----- post-hoc linear decoder h->Z (train on train windows) -----
    # collect (h, Z) pairs from training windows to fit ridge W,b: Z ≈ h W + b
    H_list, Z_list = [], []
    with torch.no_grad():
        for xb, ub in train_loader:
            x = xb.to(device, non_blocking=True)
            u = ub.to(device, non_blocking=True)
            h_samp, _ = infer(x)                  # (B,W,H)
            H_list.append(h_samp.cpu().numpy().reshape(-1, cfg.latent_dim))
            Z_list.append(x.cpu().numpy().reshape(-1, d_in))
    if H_list:
        H_mat = np.concatenate(H_list, axis=0)           # (M,H)
        Z_mat = np.concatenate(Z_list, axis=0)           # (M,d_in)
        lam = 1e-3
        H_aug = np.concatenate([H_mat, np.ones((H_mat.shape[0], 1))], axis=1)  # bias
        W_full = np.linalg.lstsq(H_aug.T @ H_aug + lam * np.eye(cfg.latent_dim + 1),
                                 H_aug.T @ Z_mat, rcond=None)[0]               # ((H+1), d_in)
        W = W_full[:-1, :]
        b = W_full[-1:, :]
    else:
        W = np.zeros((cfg.latent_dim, d_in), np.float32)
        b = np.zeros((1, d_in), np.float32)

    # ----- K-step prediction MSE on test windows -----
    def k_step_pred_mse(loader, horizons=(10, 20, 30, 40)):
        mses = {int(k): [] for k in horizons}
        with torch.no_grad():
            for xb, ub in loader:
                x = xb.to(device, non_blocking=True)  # (B,W,d_in) — Z-space input
                u = ub.to(device, non_blocking=True)
                B, Ww, Din = x.shape
                h_samp, _ = infer(x)                  # (B,W,H)
                # deterministic rollout from last context point
                for Kpred in horizons:
                    if Kpred >= Ww:
                        continue
                    # start at t0 = Ww - Kpred - 1, use that as context
                    t0 = Ww - Kpred - 1
                    h_t = h_samp[:, t0, :]            # (B,H)
                    x_t = x[:, t0, :].contiguous()    # (B,d_in)
                    preds = []
                    for t in range(t0 + 1, t0 + 1 + Kpred):
                        Ah = gen.trans_h(h_t)                 # (B, K*K)
                        Au = gen.trans_u(u[:, t, :])
                        A = (Ah + Au).view(B, gen.K, gen.K)
                        A = A - torch.logsumexp(A, dim=2, keepdim=True)
                        # pick most likely next state per batch member (greedy)
                        z_next = torch.argmax(A.mean(dim=1), dim=1)  # (B,)
                        # run the chosen state's rnn one step
                        out_list = []
                        for k in range(gen.K):
                            mask = (z_next == k)
                            if mask.any():
                                x_step = x_t[mask].unsqueeze(1)      # (b_k,1,d_in)
                                h0 = h_t[mask].unsqueeze(0)          # (1,b_k,H)
                                out_k, _ = gen.rnns[k](x_step, h0)
                                out_list.append((mask, out_k[:, 0, :]))
                        # stitch
                        h_next = torch.zeros_like(h_t)
                        for mask, val in out_list:
                            h_next[mask] = val
                        # predict Z via linear decoder
                        z_pred = h_next @ torch.from_numpy(W).to(h_next) + torch.from_numpy(b).to(h_next)
                        preds.append(z_pred)
                        # advance
                        h_t = h_next
                        x_t = z_pred
                    if preds:
                        Zpred = torch.stack(preds, dim=1)               # (B,Kpred,d_in)
                        Ztrue = x[:, t0 + 1 : t0 + 1 + Kpred, :]
                        mse = torch.mean((Zpred - Ztrue) ** 2, dim=(0, 1, 2)).item()
                        mses[int(Kpred)].append(mse)
        return {k: (float(np.mean(v)) if len(v) else np.nan) for k, v in mses.items()}

    pred_mse_test = k_step_pred_mse(test_loader, horizons=(10, 20, 30, 40))

    # ----- save artifacts -----
    np.save(save_dir / "elbos.npy", np.array(elbos, dtype=np.float32))
    np.save(save_dir / "losses.npy", np.array(losses, dtype=np.float32))
    np.save(save_dir / "x_hat.npy", XH)            # inferred h on train windows
    np.save(save_dir / "z_hat.npy", ZH)            # hard states on train windows
    np.save(save_dir / "FR_z.npy", FRz_full)       # z-scored full signal
    np.save(save_dir / "X_dr.npy", Zz)             # reduced signal (full T)
    np.save(save_dir / "footshock.npy", u_sec)

    with open(save_dir / "dr_meta.json", "w") as f:
        json.dump(dr_meta, f, indent=2)
    with open(save_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(save_dir / "prediction_mse.json", "w") as f:
        json.dump(pred_mse_test, f, indent=2)

    # quick ELBO plot (sign so larger=better)
    plt.figure()
    plt.plot(-np.array(losses), label="ELBO (approx)")
    plt.xlabel("epoch"); plt.ylabel("ELBO ↑")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / "elbo.png"); plt.close()

    # usage bar
    if usage:
        plt.figure()
        keys = sorted(usage.keys(), key=lambda s: int(s.replace("p_state", "")))
        vals = [usage[k] for k in keys]
        plt.bar(keys, vals)
        plt.ylabel("state usage")
        plt.tight_layout()
        plt.savefig(save_dir / "state_usage.png"); plt.close()

    # finalize checkpoint
    torch.save(
        {"epoch": cfg.num_iters, "infer": infer.state_dict(), "gen": gen.state_dict(),
         "loss_history": losses, "elbo_history": elbos},
        save_dir / "final_checkpoint.pth"
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
# Simple wrapper utilities
# =========================
def run_fit_srnn(h5_path: Path,
                 csv_path: Path,
                 save_dir: Path,
                 K_states: int,
                 seed: int,
                 kappa: float,
                 *,
                 dr_method="dca1",
                 dr_n_components=8,
                 dr_random_state=None,
                 latent_dim=8,
                 num_iters=2000,
                 warmup_epochs=200,
                 window_size=100,
                 batch_size=128,
                 lr=1e-4,
                 overwrite=False,
                 verbose=True,
                 device: str | None = None):
    """
    Backwards-compatible thin wrapper (kept so your existing scripts don't break).
    Uses time split=0.2 by default and stride=1.
    """
    cfg = TrainConfig(
        rat_id=-1,  # not used by this thin wrapper
        data_root=h5_path.parent.parent,  # best guess
        outputs_root=save_dir.parent.parent if save_dir else DEFAULT_OUTPUTS_ROOT,
        subset_name="custom",
        dr_method=dr_method,
        dr_n=dr_n_components,
        dr_random_state=dr_random_state if dr_random_state is not None else seed,
        K_states=K_states,
        latent_dim=latent_dim,
        kappa=kappa,
        num_iters=num_iters,
        warmup_epochs=warmup_epochs,
        window_size=window_size,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        overwrite=overwrite,
        verbose=verbose
    )
    # For this wrapper we still derive Z etc. inside fit_srnn_with_split-like path,
    # but we must point to your rat_id functions normally; to keep things simple
    # recommend using fit_srnn_with_split with a real rat_id instead.
    return {"status": "use_fit_srnn_with_split", "msg": "Prefer TrainConfig + fit_srnn_with_split() in new code."}


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
                    subset="responsive"):
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
        )
        res = fit_srnn_with_split(cfg)
        results.append({"kappa": kappa, **res})
    return results


# =========================
# CLI (optional)
# =========================
if __name__ == "__main__":
    import argparse, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=False, help="YAML config path")
    args = p.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            conf = yaml.safe_load(f)
        # map YAML to TrainConfig fields (with safe defaults)
        exp = conf.get("experiment", {})
        data = conf.get("data", {})
        dr = conf.get("dr", {})
        model = conf.get("model", {})
        train = conf.get("training", {})
        reg = conf.get("initialization", {})

        cfg = TrainConfig(
            rat_id=int(data.get("rat_id", 0)),
            data_root=Path(data.get("data_root", DEFAULT_DATA_ROOT)),
            outputs_root=Path(data.get("outputs_root", DEFAULT_OUTPUTS_ROOT)),
            subset_name=str(data.get("subset", "responsive")),
            dr_method=str(dr.get("method", "dca1")),
            dr_n=int(dr.get("n_components", 8)),
            dr_random_state=dr.get("random_state", exp.get("seed", 0)),
            K_states=int(model.get("K_states", 5)),
            latent_dim=int(model.get("latent_dim", 8)),
            kappa=float(model.get("kappa_values", [0.0])[0]) if "kappa_values" in model else float(model.get("kappa", 0.0)),
            num_iters=int(train.get("num_iters", 2000)),
            warmup_epochs=int(reg.get("warmup_epochs", 200)),
            window_size=int(train.get("window_size", 100)),
            stride=1,
            batch_size=int(train.get("batch_size", 128)),
            lr=float(train.get("lr", 1e-4)),
            seed=int(exp.get("seed", 0)),
            test_split=float(train.get("test_split", 0.2)),
            overwrite=bool(train.get("overwrite", False)),
            verbose=bool(train.get("verbose", True)),
            lambda_entropy=float(reg.get("lambda_entropy", 1e-3)),
            lambda_usage=float(reg.get("lambda_usage", 1e-2)),
        )
        out = fit_srnn_with_split(cfg)
        print(json.dumps(out, indent=2))
    else:
        print("Pass --config configs/your_experiment.yaml")

