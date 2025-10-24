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
from numpy.linalg import eigh

# === Defaults ===
DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/sRNN")
DEFAULT_OUTPUTS_ROOT = Path("/content/drive/MyDrive/sRNN/sRNN-Model-Outputs")

# --- paths / device ---
def set_roots(data_root: str | Path = None, outputs_root: str | Path = None):
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

def h5_path_for_rat(rid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "Rat-Data-hdf5" / f"Rat{rid}" / "NpxFiringRate_Behavior_SBL_10msBINS_0smoothing.hdf5"

def csv_path_responsive_all(rid: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root) / "Sub-Data" / "Only-Responsive" / f"Rat{rid}" / f"area_splits_rat{rid}_responsive" / "responsive_rates_raw.csv"

def base_model_dir(rid: int, outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    return Path(outputs_root) / f"Rat{rid}-Model-Outputs"

def run_dir(rid: int, suffix: str, K: int, seed: int, kappa: float, outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    name = f"models_Rat{rid}_{suffix}_K{K}_seed{seed}_kappa{kappa:g}"
    return base_model_dir(rid, outputs_root) / name


# --- data utils ---
def read_wide_csv(csv_path: Path) -> pd.DataFrame:
    try: return pd.read_csv(csv_path)
    except Exception: return pd.read_csv(csv_path, engine="python")

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
    sd = X.std(axis=0, ddof=0, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd

def downsample_FR_and_u(FR_TN, u_T1, *, ms_per_sample=10, rate_mode="mean"):
    FR_TN = np.asarray(FR_TN, float)
    u_T1  = np.asarray(u_T1,  float)
    FR_TN = np.nan_to_num(FR_TN)
    u_T1  = np.nan_to_num(u_T1)
    factor = int(round(1000 / ms_per_sample))
    T, N = FR_TN.shape
    T_sec = T // factor
    if T_sec == 0: raise ValueError("Not enough samples for a 1-second bin.")
    cut = T_sec * factor
    FR_sec = np.nanmean(FR_TN[:cut].reshape(T_sec, factor, N), axis=1)
    u_sec  = np.nanmax(u_T1[:cut].reshape(T_sec, factor, 1), axis=1)
    return FR_sec, u_sec


# --- DCA auto-dim ---
def _dca_auto_select(X_TN, *, lag=1, ridge=1e-6, symmetric=False,
                     variance_goal=0.90, cap=20, min_d=2):
    X = np.asarray(X_TN, float)
    X -= X.mean(0, keepdims=True)
    T, N = X.shape
    if T <= lag: raise ValueError(f"lag={lag} requires T>{lag}, but T={T}")

    C0 = (X.T @ X) / T + ridge * np.eye(N)
    Xp, Xf = X[:-lag], X[lag:]
    C1 = (Xp.T @ Xf) / (T - lag)

    w, Vc0 = eigh(C0)
    w = np.maximum(w, ridge)
    Wm12 = Vc0 @ np.diag(1.0 / np.sqrt(w)) @ Vc0.T

    if symmetric:
        M = Wm12 @ (0.5 * (C1 + C1.T)) @ Wm12
        M = 0.5 * (M + M.T)
    else:
        Mcca = Wm12 @ C1 @ Wm12
        M = 0.5 * (Mcca + Mcca.T)

    vals, vecs = eigh(M)
    order = np.argsort(np.abs(vals))[::-1]
    vals_sorted, vecs_sorted = vals[order], vecs[:, order]
    abs_vals = np.abs(vals_sorted)
    cum = np.cumsum(abs_vals) / abs_vals.sum() if abs_vals.sum() > 0 else np.array([])
    d = int(np.searchsorted(cum, variance_goal) + 1) if len(cum) else min_d
    d = int(np.clip(d, min_d, cap))
    V = Wm12 @ vecs_sorted[:, :d]
    Z = X @ V
    return Z.astype(np.float32), V.astype(np.float32), d, vals_sorted


# --- embedding ---
def make_embedding(FR_sec, method="pca", n_components=None, random_state=None, allow_2d_input=False):
    X = _std_nan_robust(FR_sec)

    if method == "none":
        Z = X.astype(np.float32)
        return Z, {"method":"none","n_components":int(Z.shape[1])}

    if method in ("pca","fa","ica","nmf"):
        d = n_components if n_components is not None else min(10, X.shape[1])
        if method == "pca":
            model = PCA(n_components=d, random_state=random_state).fit(X); Z = model.transform(X)
        elif method == "fa":
            model = FactorAnalysis(n_components=d, random_state=random_state).fit(X); Z = model.transform(X)
        elif method == "ica":
            model = FastICA(n_components=d, random_state=random_state, max_iter=2000).fit(X); Z = model.transform(X)
        else:
            Xpos = X - X.min() + 1e-6
            model = NMF(n_components=d, init="nndsvda", random_state=random_state, max_iter=2000).fit(Xpos)
            Z = model.transform(Xpos)
        return Z.astype(np.float32), {"method":method,"n_components":int(Z.shape[1])}

    if method in ("dca1","dca1_sym"):
        symmetric = (method == "dca1_sym")
        if n_components is None:
            Z, V, d_auto, eigvals = _dca_auto_select(
                X, lag=1, ridge=1e-6, symmetric=symmetric, variance_goal=0.9, cap=20, min_d=2
            )
            return Z, {"method":method,"n_components":int(d_auto),"auto":True,"variance_goal":0.9,"cap":20}
        else:
            d = int(n_components)
            Z, V, _, _ = _dca_auto_select(X, lag=1, ridge=1e-6, symmetric=symmetric, variance_goal=1.0, cap=d, min_d=d)
            return Z, {"method":method,"n_components":d,"auto":False}

    if method in ("tsne2","isomap2","lle2"):
        pca50_dim = min(50, X.shape[1])
        X_pca50 = PCA(n_components=pca50_dim, random_state=random_state).fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "tsne2":
                Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=random_state).fit_transform(X_pca50)
            elif method == "isomap2":
                Z = Isomap(n_neighbors=15, n_components=2).fit_transform(X_pca50)
            else:
                Z = LocallyLinearEmbedding(n_neighbors=15, n_components=2, random_state=random_state).fit_transform(X_pca50)
        if not allow_2d_input:
            print(f"[warn] {method} is 2-D; great for viz, not ideal as SRNN inputs.")
        return Z.astype(np.float32), {"method":method,"n_components":2}

    raise ValueError(f"Unknown DR method: {method}")


# --- dataset ---
class NeuralDataset(Dataset):
    def __init__(self, rates_z, footshock, window_size=100, stride=1):
        X = np.asarray(rates_z, np.float32).T
        u = np.asarray(footshock, np.float32)
        if u.ndim == 1: u = u[:,None]
        if X.shape[0] != u.shape[0]: raise ValueError("length mismatch")
        self.X, self.u = torch.from_numpy(X), torch.from_numpy(u)
        self.W, self.stride = window_size, max(1, int(stride))
    def __len__(self): return 1 + (len(self.X)-self.W)//self.stride
    def __getitem__(self, i):
        s=i*self.stride; e=s+self.W
        x=self.X[s:e,:]; u=self.u[s:e,:]
        return torch.nan_to_num(x), torch.nan_to_num(u)


# --- SRNN patches (forward_footshock etc.) ---
def apply_srnn_patches():
    from sRNN.networks import InferenceNetwork as _Inf, GenerativeSRNN as _Gen

    def _inf_forward_stable(self, x):
        x=torch.nan_to_num(x).contiguous()
        try:self.rnn.flatten_parameters()
        except:pass
        B,T,D=x.shape; num_dirs=2 if getattr(self.rnn,"bidirectional",False) else 1
        h0=torch.zeros(num_dirs, B, self.rnn.hidden_size, device=x.device)
        h_seq,_=self.rnn(x,h0)
        m=self.fc_mean(h_seq)
        lv=torch.clamp(torch.nan_to_num(self.fc_logvar(h_seq)),min=-8,max=5)
        std=torch.exp(0.5*lv)+1e-6
        q=torch.distributions.Independent(torch.distributions.Normal(m,std),1)
        return torch.nan_to_num(q.rsample()), q

    def _gen_forward_footshock(self,x,u,h_samp):
        B,T,N=x.shape; K=self.K; H=self.H; device=x.device
        x,u,h_samp=torch.nan_to_num(x),torch.nan_to_num(u),torch.nan_to_num(h_samp)
        if not hasattr(self,"pi0"): self.pi0=nn.Parameter(torch.zeros(K,device=device))
        p0=F.log_softmax(self.pi0,dim=0).view(1,K).expand(B,K)
        p_s=torch.full((B,T,K,K),-1e8,device=device); p_h=torch.zeros(B,T,K,device=device)
        if not hasattr(self,"trans_h"): self.trans_h=nn.Linear(H,K*K).to(device)
        if not hasattr(self,"trans_u"): self.trans_u=nn.Linear(1,K*K,bias=False).to(device)
        if not hasattr(self,"rnns") or len(self.rnns)!=K:
            self.rnns=nn.ModuleList([nn.RNN(N,H,batch_first=True).to(device) for _ in range(K)])
        for t in range(1,T):
            A=(self.trans_h(h_samp[:,t-1,:])+self.trans_u(u[:,t,:])).view(B,K,K)
            kappa=getattr(self,"kappa",0.0)
            if kappa!=0.0: A=A+torch.eye(K,device=device)*kappa
            A=A-torch.logsumexp(A,dim=2,keepdim=True)
            p_s[:,t]=A
            for k in range(K):
                out,_=self.rnns[k](x[:,t-1:t,:],h_samp[:,t-1,:].unsqueeze(0))
                diff=h_samp[:,t,:]-out[:,0,:]
                p_h[:,t,k]=-0.5*(diff.pow(2).sum(dim=1)/(1e-2**2))
        return p0,p_s,p_h,None,None,None,None,None

    _Inf.forward=_inf_forward_stable
    _Gen.forward=_gen_forward_footshock
    return _Inf,_Gen

