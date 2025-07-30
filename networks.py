# -----------------------------------------------------------------------------
# networks_local.py
# -----------------------------------------------------------------------------

"""Lightweight reference implementations of the amortised inference network
and the generative SRNN used in the original Saxena‑lab codebase.  These are
*independent* of the classes inside ``SRNN/`` so you can swap them in and out
for ablation studies.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

__all__ = [
    "InferenceNetwork",
    "GenerativeSRNN",
    "build_inference_network",
    "build_generative_srnn",
]


class InferenceNetwork(nn.Module):
    """q(h | x) – simple RNN encoder producing a diagonal Gaussian."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Independent]:
        h_seq, _ = self.rnn(x)  # [B, T, H]
        mean = self.fc_mean(h_seq)
        logvar = self.fc_logvar(h_seq)
        std = torch.exp(0.5 * logvar)
        q_dist = Independent(Normal(mean, std), 1)
        h = q_dist.rsample()
        return h, q_dist


class GenerativeSRNN(nn.Module):
    """p(x, h) – discrete state RNN with Gaussian emissions."""

    def __init__(self, D: int, K: int, H: int):
        super().__init__()
        self.D, self.K, self.H = D, K, H

        # one RNN per discrete state
        self.rnns = nn.ModuleList([nn.RNN(D, H, batch_first=True) for _ in range(K)])

        self.emission = nn.Sequential(
            nn.Linear(H, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, D),
        )

        # discrete state machinery
        self.pi0 = nn.Parameter(torch.zeros(K))              # initial log‑probabilities
        self.trans = nn.RNNCell(H, K * K)
        if self.trans.bias is None:
            self.trans.bias = nn.Parameter(torch.zeros(K * K))

        # pre‑compute Gaussian constants
        self.std_h = 1e-2
        self.std_y = 1e-2
        self.const_h = -0.5 * (H * math.log(2 * math.pi * self.std_h**2))
        self.const_y = -0.5 * (D * math.log(2 * math.pi * self.std_y**2))

    # NOTE: forward identical to notebook – kept for clarity
    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        y: torch.Tensor,  # [B, T, 1]
        h_samp: torch.Tensor,  # [B, T, H]
    ):
        B, T, _ = x.shape
        device = x.device

        # p(s_0)
        pi = self.pi0 - torch.logsumexp(self.pi0, 0)  # [K]
        p0 = pi.unsqueeze(0).expand(B, -1)            # [B, K]

        p_s = torch.zeros(B, T, self.K, self.K, device=device)
        p_h = torch.zeros(B, T, self.K, device=device)
        p_y = torch.zeros(B, T, device=device)
        h_prev = torch.zeros(B, self.K * self.K, device=device)

        for t in range(T):
            if t == 0:
                p_s[:, 0] = torch.eye(self.K, device=device)[None]
                # p(h_0 | s_0=k) ≡ N(0, σ²I)
                diff0 = h_samp[:, 0, :]                # centred at 0
                p_h[:, 0] = self.const_h - (diff0**2).sum(dim=1) / (2 * self.std_h**2)
            else:
                flat = self.trans(h_samp[:, t - 1, :], h_prev)
                A = flat.view(B, self.K, self.K)
                p_s[:, t] = A - torch.logsumexp(A, dim=2, keepdim=True)
                h_prev = flat
                for k in range(self.K):
                    out_k, _ = self.rnns[k](x[:, t : t + 1, :], h_samp[:, t - 1, :].unsqueeze(0))
                    mean_h = out_k.squeeze(1)
                    diff = h_samp[:, t, :] - mean_h
                    p_h[:, t, k] = self.const_h - (diff**2).sum(dim=1) / (2 * self.std_h**2)

            mean_y = self.emission(h_samp[:, t, :])
            diff_y = y[:, t, :] - mean_y
            p_y[:, t] = self.const_y - (diff_y**2).sum(dim=1) / (2 * self.std_y**2)

        from SRNN import baum_welch  # heavy dependency, keep inside fn
        fwd = baum_welch.dis_forward_pass(p_s, p0, p_h, p_y)
        bwd = baum_welch.dis_backward_pass(p_s, p0, p_h, p_y)
        gamma, delta = baum_welch.get_gamma(fwd, bwd, p_s, p_h, p_y)
        return p0, p_s, p_h, p_y, gamma, delta, fwd, bwd


     
def build_inference_network(input_dim: int, hidden_dim: int = 64, device="cpu") -> InferenceNetwork:
    return InferenceNetwork(input_dim, hidden_dim).to(device)


def build_generative_srnn(D: int, K: int = 2, H: int = 64, device="cpu") -> GenerativeSRNN:
    return GenerativeSRNN(D, K, H).to(device)

