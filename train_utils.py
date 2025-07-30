# -----------------------------------------------------------------------------
# train_utils.py
# -----------------------------------------------------------------------------

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


################################################################################
# internal helpers
################################################################################

def _step(
    gen_model: nn.Module,
    infer_net: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    x, y = batch  # unpack window and speed
    x, y = x.to(device), y.to(device)

    infer_net.train(optimizer is not None)
    gen_model.train(optimizer is not None)

    h_samp, q_dist = infer_net(x)
    p0, p_s, p_h, p_y, *_ = gen_model(x, y, h_samp)

    log_q = q_dist.log_prob(h_samp).sum(dim=(1, 2))  # [B]
    L0 = p0.sum(dim=1)
    Ls = p_s.sum(dim=(1, 2, 3))
    Lh = p_h.sum(dim=(1, 2))
    Ly = p_y.sum(dim=1)
    elbo = (L0 + Ls + Lh + Ly - log_q).mean()
    loss = -elbo

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return {"loss": loss.item(), "elbo": elbo.item()}


def _run_epoch(
    gen_model: nn.Module,
    infer_net: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    stats = {"loss": 0.0, "elbo": 0.0}
    for batch in loader:
        out = _step(gen_model, infer_net, batch, optimizer, device)
        for k, v in out.items():
            stats[k] += v
    for k in stats:
        stats[k] /= len(loader)
    return stats

################################################################################
# public API
################################################################################

def train_srnn(
    gen_model: nn.Module,
    infer_net: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    checkpoint_dir: str | Path | None = None,
    verbose: bool = True,
) -> List[Dict[str, float]]:
    device = torch.device(device)
    infer_net.to(device)
    gen_model.to(device)

    optimizer = torch.optim.Adam(itertools.chain(gen_model.parameters(), infer_net.parameters()), lr=lr)

    history: List[Dict[str, float]] = []
    best_loss = float("inf")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None

    for epoch in range(1, epochs + 1):
        stats = _run_epoch(gen_model, infer_net, loader, optimizer, device)
        history.append(stats)

        if verbose:
            print(f"[Epoch {epoch}/{epochs}] loss={stats['loss']:.4f} elbo={stats['elbo']:.4f}")

        # checkpoint
        if ckpt_dir is not None and stats["loss"] < best_loss:
            best_loss = stats["loss"]
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "gen_model": gen_model.state_dict(),
                    "infer_net": infer_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "epoch": epoch,
                },
                ckpt_dir / "best.pth",
            )

    return history


def evaluate_srnn(
    gen_model: nn.Module,
    infer_net: nn.Module,
    loader: DataLoader,
    device: str | torch.device = "cpu",
) -> Dict[str, float]:
    device = torch.device(device)
    with torch.no_grad():
        return _run_epoch(gen_model, infer_net, loader, None, device)
