# `srnn_helper.py` 
from __future__ import annotations
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, NMF
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
import h5py
import matplotlib.pyplot as plt

# Module-level defaults so short helper calls (no explicit roots) work too
DATA_ROOT = Path(os.environ.get("SRNN_DATA_ROOT", ".")).resolve()
OUTPUTS_ROOT = Path(os.environ.get("SRNN_OUTPUTS_ROOT", "./sRNN-Model-Outputs")).resolve()

# Device
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Paths (parametrized + convenient wrappers)

def h5_path_for_rat(rid: int, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root / "Rat-Data-hdf5" / f"Rat{rid}" / "NpxFiringRate_Behavior_SBL_10msBINS_0smoothing.hdf5"

def csv_path_area(rid: int, area: str, subset: str, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root / "Sub-Data" / "Seperate-by-Area" / f"Rat{rid}" / f"area_splits_rat{rid}_{subset}" / f"{area}_wide.csv"

def csv_path_responsive_all(rid: int, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root / "Sub-Data" / "Only-Responsive" / f"Rat{rid}" / f"area_splits_rat{rid}_responsive" / "responsive_rates_raw.csv"

def base_model_dir(rid: int, outputs_root: Path | None = None) -> Path:
    out = Path(outputs_root) if outputs_root is not None else OUTPUTS_ROOT
    return out / f"Rat{rid}-Model-Outputs"

def run_dir(rid: int, suffix: str, K: int, seed: int, kappa: float, outputs_root: Path | None = None) -> Path:
    name = f"models_Rat{rid}_{suffix}_K{K}_seed{seed}_kappa{kappa:g}"
    return base_model_dir(rid, outputs_root) / name

def suffix_from(area: str | None, subset: str | None) -> str:
    if area is None and subset == "responsive": return "responsive_all"
    if subset == "responsive": return f"{area}_responsive"
    if subset == "allActive":  return f"{area}"
    raise ValueError(f"Unknown (area={area}, subset={subset})")
# Data utils
def read_wide_csv(csv_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.read_csv(csv_path, engine="python")

def build_footshock_regressor(t: np.ndarray, shock_times: np.ndarray | None) -> np.ndarray:
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
    sd = X.std(axis=0, ddof=0, keepdims=True); sd[sd == 0.0] = 1.0
    return (X - mu) / sd

def downsample_FR_and_u(FR_TN, u_T1, *, ms_per_sample=10, rate_mode="mean"):
    FR_TN = np.asarray(FR_TN, float)
    u_T1  = np.asarray(u_T1,  float)
    FR_TN = np.nan_to_num(FR_TN, nan=0.0, posinf=0.0, neginf=0.0)
    u_T1  = np.nan_to_num(u_T1,  nan=0.0, posinf=0.0, neginf=0.0)
    factor = int(round(1000 / ms_per_sample))  # samples per 1 s
    if factor <= 0:
        raise ValueError("ms_per_sample must be > 0 and <= 1000.")
    T, N = FR_TN.shape
    if u_T1.shape[0] != T:
        raise ValueError(f"Length mismatch: FR_TN T={T} vs u_T1 T={u_T1.shape[0]}")
    T_sec = T // factor
    if T_sec == 0:
        raise ValueError("Not enough samples for a 1-second bin.")
    cut    = T_sec * factor
    B      = FR_TN[:cut].reshape(T_sec, factor, N)
    FR_sec = np.nanmean(B, axis=1) if rate_mode == "mean" else np.nansum(B, axis=1)
    u_sec  = (u_T1[:cut].reshape(T_sec, factor, 1).max(axis=1)).astype(FR_sec.dtype)
    return np.nan_to_num(FR_sec), np.nan_to_num(u_sec)

def make_embedding(FR_sec, method="pca", n_components=None, random_state=None, allow_2d_input=False):
    X = _std_nan_robust(FR_sec)
    if method == "none":
        Z = X.astype(np.float32);  return Z, {"method":"none","n_components":int(Z.shape[1])}
    if method in ("pca","fa","ica","nmf"):
        d = n_components if n_components is not None else min(10, X.shape[1])
        if method == "pca":
            model = PCA(n_components=d, random_state=random_state).fit(X)
            Z = model.transform(X)
        elif method == "fa":
            model = FactorAnalysis(n_components=d, random_state=random_state).fit(X); Z = model.transform(X)
        elif method == "ica":
            model = FastICA(n_components=d, random_state=random_state, max_iter=2000).fit(X); Z = model.transform(X)
        else:
            Xpos = X - X.min() + 1e-6
            model = NMF(n_components=d, init="nndsvda", random_state=random_state, max_iter=2000).fit(Xpos)
            Z = model.transform(Xpos)
        return Z.astype(np.float32), {"method":method,"n_components":int(Z.shape[1])}
    if method in ("tsne2","isomap2","lle2"):
        pca50_dim = min(50, X.shape[1])
        X_pca50 = PCA(n_components=pca50_dim, random_state=random_state).fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "tsne2":
                Z = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca",
                         random_state=random_state).fit_transform(X_pca50)
            elif method == "isomap2":
                Z = Isomap(n_neighbors=15, n_components=2).fit_transform(X_pca50)
            else:
                Z = LocallyLinearEmbedding(n_neighbors=15, n_components=2,
                                           method="standard", random_state=random_state).fit_transform(X_pca50)
        if not allow_2d_input:
            print(f"[warn] {method} is 2-D; great for viz, not ideal as SRNN inputs.")
        return Z.astype(np.float32), {"method":method,"n_components":2}
    raise ValueError(f"Unknown DR method: {method}")

def choose_latent_dim(Z_raw, FR_sec, strategy, fixed=None, cap=20, variance_goal=0.90, rule_mult=1.0):
    d_in = Z_raw.shape[1]
    if strategy == "fixed":
        if fixed is None: raise ValueError("fixed latent_dim requested but `fixed` is None")
        return int(max(1, fixed))
    if strategy == "input_dim_cap": return int(np.clip(d_in, 2, cap))
    if strategy == "pca_variance":
        pcs = PCA().fit(_std_nan_robust(FR_sec))
        cum = np.cumsum(pcs.explained_variance_ratio_)
        d = np.argmax(cum >= variance_goal) + 1
        return int(np.clip(d, 2, cap))
    if strategy == "rule_of_thumb":
        d = int(round(rule_mult * d_in)); return int(np.clip(d, 2, cap))
    raise ValueError(f"Unknown latent dim strategy: {strategy}")


# Dataset
class NeuralDataset(Dataset):
    def __init__(self, rates_z: np.ndarray, footshock: np.ndarray, window_size=100, stride=1, dtype=np.float32):
        X = np.asarray(rates_z, dtype=dtype).T
        u = np.asarray(footshock, dtype=dtype)
        if u.ndim == 1: u = u[:, None]
        if X.shape[0] != u.shape[0]:
            raise ValueError(f"time length mismatch: X T={X.shape[0]} vs footshock T={u.shape[0]}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X)
        self.u = torch.from_numpy(u)
        self.W = int(window_size)
        self.stride = max(1, int(stride))
        if len(self.X) < self.W: raise ValueError(f"T={len(self.X)} shorter than window={self.W}")
    def __len__(self):
        return 1 + (len(self.X) - self.W) // self.stride
    def __getitem__(self, idx):
        s = idx * self.stride; e = s + self.W
        x = self.X[s:e, :]; u = self.u[s:e, :]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return x, u

# Import SRNN & Patch (fixes contiguity + cuDNN friendliness)

def apply_srnn_patches():
    from sRNN.networks import InferenceNetwork as _Inf, GenerativeSRNN as _Gen

    def _inf_forward_stable(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        # Ensure RNN params are packed for cuDNN
        try:
            self.rnn.flatten_parameters()
        except Exception:
            pass
        # Provide an explicit hx that is contiguous
        B, T, D = x.shape
        h0 = torch.zeros(1, B, self.hidden_dim, device=x.device, dtype=x.dtype).contiguous()
        h_seq, _ = self.rnn(x, h0)
        m  = self.fc_mean(h_seq)
        lv = torch.clamp(torch.nan_to_num(self.fc_logvar(h_seq), nan=0.0, posinf=10.0, neginf=-10.0), min=-8.0, max=5.0)
        std = torch.exp(0.5 * lv) + 1e-6
        std = torch.nan_to_num(std, nan=1e-3, posinf=1.0, neginf=1e-3)
        q = torch.distributions.Independent(torch.distributions.Normal(m, std), 1)
        h = torch.nan_to_num(q.rsample(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        return h, q

    def _gen_forward_footshock(self, x, u, h_samp):
        B, T, N = x.shape; K = self.K; H = self.H; device = x.device
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        # ensure contiguous latent samples for RNN initial states
        h_samp = torch.nan_to_num(h_samp, nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        if not hasattr(self, "pi0"): self.pi0 = nn.Parameter(torch.zeros(K, device=device))
        p0 = F.log_softmax(self.pi0, dim=0).view(1, K).expand(B, K).contiguous()
        p_s = torch.full((B, T, K, K), -1e8, device=device)
        p_h = torch.zeros(B, T, K, device=device)
        if hasattr(self, "trans"): del self.trans
        if not hasattr(self, "trans_h"): self.trans_h = nn.Linear(H, K*K, bias=True).to(device)
        if not hasattr(self, "trans_u"): self.trans_u = nn.Linear(1, K*K, bias=False).to(device)
        if not hasattr(self, "rnns") or len(self.rnns) != K:
            self.rnns = nn.ModuleList([nn.RNN(N, H, batch_first=True).to(device) for _ in range(K)])
        # Pack parameters for cuDNN (safe no-op on non-cuDNN backends)
        for r in self.rnns:
            try:
                r.flatten_parameters()
            except Exception:
                pass
        if not hasattr(self, "std_h"):   self.std_h = 1e-2
        if not hasattr(self, "const_h"): self.const_h = float(-0.5*(H*np.log(2*np.pi*(self.std_h**2))))
        eye = torch.eye(K, device=device).view(1, K, K).expand(B, K, K)
        p_s[:, 0] = torch.log(eye + 1e-12); p_h[:, 0, :] = self.const_h
        for t in range(1, T):
            Ah = self.trans_h(h_samp[:, t-1, :]); Au = self.trans_u(u[:, t, :])
            A  = (Ah + Au).view(B, K, K)
            kappa_val = float(getattr(self, "kappa", 0.0))
            if kappa_val != 0.0:
                A = A + torch.eye(K, device=device).view(1, K, K) * kappa_val
            A = A - torch.logsumexp(A, dim=2, keepdim=True)
            p_s[:, t] = A
            # ensure RNN inputs & hx are contiguous and fresh storage
            x_step = x[:, t-1:t, :].contiguous().clone()
            h0     = h_samp[:, t-1, :].contiguous().unsqueeze(0).clone()  # (1, B, H)
            for k in range(K):
                out_k, _ = self.rnns[k](x_step, h0)
                mean_h = out_k[:, 0, :]
                diff   = h_samp[:, t, :] - mean_h
                p_h[:, t, k] = self.const_h - (diff.pow(2).sum(dim=1) / (2 * (self.std_h ** 2)))
        p0 = torch.nan_to_num(p0, nan=-1e-8, posinf=-1e8, neginf=-1e8)
        p_s = torch.nan_to_num(p_s, nan=-1e-8, posinf=-1e8, neginf=-1e8)
        p_h = torch.nan_to_num(p_h, nan=-1e-8, posinf=-1e8, neginf=-1e-8)
        return p0, p_s, p_h, None, None, None, None, None

    _Inf.forward = _inf_forward_stable
    _Gen.forward = _gen_forward_footshock
    return _Inf, _Gen

# Trainer
def fit_single_srnn(h5_path: Path, csv_path: Path, save_dir: Path,
                    *, K_states=3, latent_dim=None, latent_dim_strategy="input_dim_cap",
                    variance_goal=0.90, latent_cap=20, latent_fixed=None,
                    kappa=0.0, num_iters=300, window_size=100, batch_size=128,
                    lr=1e-3, seed=0, overwrite=False, verbose=True,
                    ms_per_sample=None, rate_mode="mean",
                    dr_method="pca", dr_n_components=None, dr_random_state=None,
                    device: str | None = None):
    if device is None: device = get_device()
    torch.manual_seed(seed); np.random.seed(seed)
    # Optional cuDNN autotuner for fixed shapes
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    save_dir = Path(save_dir)
    if save_dir.exists() and not overwrite:
        raise FileExistsError(f"{save_dir} exists (use overwrite=True)")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = read_wide_csv(csv_path)
    if "time_s" in df.columns: t = df.pop("time_s").values
    else:
        with h5py.File(h5_path, "r") as h5: t = h5["time"][...]
    FR_TN = df.values.astype(float)
    with h5py.File(h5_path, "r") as h5:
        shock_times = h5.get("footshock_times")
        shock_times = None if shock_times is None else shock_times[...]
    footshock = build_footshock_regressor(t, shock_times)

    if ms_per_sample is None:
        ms_per_sample = max(1, int(round(1000 * float(np.median(np.diff(t))) ))) if len(t) >= 3 else 10
        if verbose: print(f"↳ inferred ms_per_sample ≈ {ms_per_sample} ms")
    FR_sec, u_sec = downsample_FR_and_u(FR_TN, footshock, ms_per_sample=ms_per_sample, rate_mode=rate_mode)

    # keep raw FR_sec for saving; model uses DR+z
    mu_full, sd_full = FR_sec.mean(0, keepdims=True), FR_sec.std(0, keepdims=True)
    sd_full[sd_full == 0.0] = 1.0
    FRz_full = np.nan_to_num((FR_sec - mu_full) / sd_full, nan=0.0, posinf=0.0, neginf=0.0)

    rs = dr_random_state if dr_random_state is not None else seed
    Z_raw, dr_meta = make_embedding(FR_sec, method=dr_method, n_components=dr_n_components, random_state=rs)
    d_in = Z_raw.shape[1]
    Zz   = StandardScaler().fit_transform(Z_raw).astype(np.float32)

    if latent_dim is None:
        latent_dim = choose_latent_dim(Z_raw, FR_sec, strategy=latent_dim_strategy,
                                       fixed=latent_fixed, cap=latent_cap, variance_goal=variance_goal)
        if verbose: print(f"latent_dim → {latent_dim} (input d={d_in}, strategy={latent_dim_strategy}, DR={dr_meta['method']})")

    ds = NeuralDataset(Zz.T, u_sec, window_size=window_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=max(0, (os.cpu_count() or 2) // 2),
    )

    _Inf, _Gen = apply_srnn_patches()
    infer = _Inf(input_dim=d_in, hidden_dim=latent_dim).to(device)
    gen   = _Gen(D=d_in, K=K_states, H=latent_dim).to(device)
    gen.kappa = float(kappa)

    for m in list(infer.modules()) + list(gen.modules()):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    opt = torch.optim.Adam(list(infer.parameters()) + list(gen.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_iters, eta_min=lr*0.1)

    def _lr_with_warmup(epoch, base_lr, warmup_epochs=10):
        return base_lr * float(epoch + 1) / float(warmup_epochs) if epoch < warmup_epochs else base_lr

    losses, ckpt_prefix = [], save_dir / "epoch"
    for epoch in range(1, num_iters + 1):
        for g in opt.param_groups:
            base_lr = g.get('initial_lr', lr); g['lr'] = _lr_with_warmup(epoch-1, base_lr, warmup_epochs=10)
        infer.train(); gen.train()
        total = 0.0
        for xb, ub in loader:
            # make batch tensors contiguous on device
            x = xb.to(device, non_blocking=True).contiguous()
            u = ub.to(device, non_blocking=True).contiguous()
            opt.zero_grad()
            h_samp, q_dist = infer(x)
            p0, p_s, p_h, _, *_ = gen(x, u, h_samp)
            log_q = q_dist.log_prob(h_samp).sum(dim=1)
            anneal = min(1.0, epoch / 50.0)
            elbo = (p0.sum(dim=1) + p_s.sum(dim=(1,2,3)) + p_h.sum(dim=(1,2)) - anneal * log_q).mean()
            trans_probs = torch.exp(p_s).clamp_min(1e-12)
            entropy_z = -(trans_probs * p_s).sum(dim=(1,2,3)).mean()
            loss = -elbo - 1e-3 * entropy_z
            if torch.isnan(loss):
                if verbose: print("NaN loss, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(infer.parameters()) + list(gen.parameters()), 5.0)
            opt.step()
            total += loss.item()
        losses.append(total / max(1, len(loader)))
        scheduler.step()
        if verbose: print(f"[{save_dir.name}] epoch {epoch}/{num_iters} loss={losses[-1]:.4f}")
        if epoch % max(1, num_iters // 6) == 0:
            torch.save({"epoch": epoch, "infer": infer.state_dict(),
                        "gen": gen.state_dict(), "opt": opt.state_dict(),
                        "loss_history": losses}, f"{ckpt_prefix}_{epoch}.pth")

    # Inference snapshot (on training windows)
    infer.eval(); gen.eval()
    with torch.no_grad():
        all_h, all_z = [], []
        for xb, ub in loader:
            x = xb.to(device, non_blocking=True).contiguous()
            u = ub.to(device, non_blocking=True).contiguous()
            h_samp, _ = infer(x); all_h.append(h_samp.cpu().numpy())
            p0, p_s, p_h, _, *_ = gen(x, u, h_samp)
            z_hat = torch.argmax(torch.logsumexp(p_s, dim=-1), dim=-1)
            all_z.append(z_hat.cpu().numpy())
        XH = np.concatenate(all_h, axis=0); ZH = np.concatenate(all_z, axis=0)

    # Save
    np.save(save_dir / "elbos.npy", np.array(losses, dtype=np.float32))
    # sign hint: losses trend down => -ELBO
    if len(losses) > 1:
        trend = float(np.nanmean(np.diff(losses)))
        sign_hint = -1.0 if trend < 0 else 1.0
        np.save(save_dir / "elbo_sign.npy", np.array([sign_hint], dtype=float))
    np.save(save_dir / "x_hat.npy",  XH)
    np.save(save_dir / "z_hat.npy",  ZH)
    np.save(save_dir / "FR_z.npy",   FRz_full)
    np.save(save_dir / "footshock.npy", u_sec)
    np.save(save_dir / "X_dr.npy",   Zz)
    with open(save_dir / "dr_meta.json", "w") as f:
        json.dump({"method": dr_meta["method"], "n_components": int(dr_meta["n_components"])}, f, indent=2)

    # Plot ELBO (sign-corrected)
    plt.figure(); plt.plot(-np.array(losses))
    plt.xlabel("epoch"); plt.ylabel("ELBO")
    plt.title(f"sRNN (K={K_states}, Latent={latent_dim}, d_in={d_in}, {dr_meta['method']})")
    plt.tight_layout(); plt.savefig(save_dir / "elbo.png"); plt.close()

    torch.save({"epoch": num_iters, "infer": infer.state_dict(),
                "gen": gen.state_dict(), "opt": opt.state_dict(),
                "loss_history": losses}, save_dir / "final_checkpoint.pth")
    if verbose: print(f"✓ sRNN trained and saved → {save_dir}")
    return dict(status="ok", path=str(save_dir), msg="trained")

# -----------------------------------------------------------------------------
# Runner wrappers
# -----------------------------------------------------------------------------

def run_fit_srnn(h5_path: Path, csv_path: Path, save_dir: Path,
                 K_states: int, seed: int, kappa: float,
                 *, dr_method="pca", dr_n_components=None, dr_random_state=None,
                 latent_dim_strategy="input_dim_cap", variance_goal=0.90, latent_cap=20, latent_fixed=None,
                 num_iters=300, window_size=100, batch_size=128, lr=1e-3,
                 overwrite=False, verbose=True, device: str | None = None):
    try:
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        if not Path(h5_path).exists(): return dict(status="skip_h5_missing", path=str(save_dir), msg=str(h5_path))
        if not Path(csv_path).exists(): return dict(status="skip_csv_missing", path=str(save_dir), msg=str(csv_path))
        out = fit_single_srnn(
            h5_path=h5_path, csv_path=csv_path, save_dir=save_dir,
            kappa=kappa, K_states=K_states, num_iters=num_iters,
            window_size=window_size, batch_size=batch_size, lr=lr,
            seed=seed, overwrite=overwrite, verbose=verbose,
            dr_method=dr_method, dr_n_components=dr_n_components, dr_random_state=dr_random_state,
            latent_dim=None, latent_dim_strategy=latent_dim_strategy,
            variance_goal=variance_goal, latent_cap=latent_cap, latent_fixed=latent_fixed,
            device=device
        )
        return out
    except FileExistsError as e:
        return dict(status="skip_exists", path=str(save_dir), msg=str(e))
    except Exception as e:
        import traceback; traceback.print_exc(limit=1)
        return dict(status="error", path=str(save_dir), msg=f"{type(e).__name__}: {e}")


def run_kappa_sweep(*, rat: int, data_root: str | Path, outputs_root: str | Path,
                    K: int = 3, seed: int = 0, kappa_grid=(0.0,0.5,1.0,2.0),
                    dr_method="pca", dr_n=8, latent_dim=8, latent_strategy="fixed",
                    variance_goal=0.85, latent_cap=10, num_iters=10,
                    window_size=100, batch_size=128, lr=1e-4,
                    overwrite=True, verbose=True, suffix="responsive"):
    data_root   = Path(data_root)
    outputs_root= Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    h5p  = h5_path_for_rat(rat, data_root)
    csvp = csv_path_responsive_all(rat, data_root) if suffix=="responsive" else csv_path_area(rat, suffix, "allActive", data_root)

    suff = "responsive_all" if suffix=="responsive" else suffix
    base_model_dir(rat, outputs_root).mkdir(parents=True, exist_ok=True)

    print(f"[Rat {rat}] {suff}  K={K} seed={seed}  DR={dr_method}({dr_n})  H={latent_dim}")
    results = []
    for kappa in kappa_grid:
        savedir = run_dir(rat, f"{suff}_{dr_method}{dr_n}", K, seed, kappa, outputs_root)
        print(f"\n[RUN] κ={kappa}  →  {savedir}")
        info = run_fit_srnn(
            h5p, csvp, savedir, K, seed, kappa,
            dr_method=dr_method, dr_n_components=dr_n, dr_random_state=seed,
            latent_dim_strategy=latent_strategy, variance_goal=variance_goal,
            latent_cap=latent_cap, latent_fixed=latent_dim,
            num_iters=num_iters, window_size=window_size, batch_size=batch_size, lr=lr,
            overwrite=overwrite, verbose=verbose
        )
        results.append({"kappa": kappa, "seed": seed, "status": info["status"], "path": info["path"]})
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rat", type=int, default=16)
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--outputs", type=str, required=True)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kappa-grid", type=float, nargs="+", default=[0.0,0.5,1.0,2.0])
    p.add_argument("--dr-method", type=str, default="pca")
    p.add_argument("--dr-n", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--latent-strategy", type=str, default="fixed")
    p.add_argument("--variance-goal", type=float, default=0.85)
    p.add_argument("--latent-cap", type=int, default=10)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    run_kappa_sweep(
        rat=args.rat, data_root=args.data_root, outputs_root=args.outputs,
        K=args.K, seed=args.seed, kappa_grid=tuple(args.kappa_grid),
        dr_method=args.dr_method, dr_n=args.dr_n,
        latent_dim=args.latent_dim, latent_strategy=args.latent_strategy,
        variance_goal=args.variance_goal, latent_cap=args.latent_cap,
        num_iters=args.iters, window_size=args.window, batch_size=args.batch, lr=args.lr,
        overwrite=args.overwrite, suffix="responsive"
    )

