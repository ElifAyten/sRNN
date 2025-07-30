from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NeuralDataset(Dataset):
    """Windowed neural‑spike dataset.

    Parameters
    ----------
    rates : np.ndarray  – firing rates, shape ``[N, T]``
    speed : np.ndarray  – behaviour regressor, shape ``[T]``
    window_size : int   – number of time bins per training example
    """

    def __init__(self, rates: np.ndarray, speed: np.ndarray, window_size: int = 100):
        self.rates = torch.as_tensor(rates, dtype=torch.float32)  # [N, T]
        self.speed = torch.as_tensor(speed, dtype=torch.float32)  # [T]
        self.W = window_size

    # ---------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ---------------------------------------------------------------------

    def __len__(self) -> int:  # number of sliding windows
        return self.rates.shape[1] - self.W + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.rates[:, idx : idx + self.W].T        # [W, N]
        y = self.speed[idx : idx + self.W].unsqueeze(1)  # [W, 1]
        return x, y


def make_dataloader(
    rates: np.ndarray,
    speed: np.ndarray,
    window_size: int = 100,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Convenience wrapper that returns a ``DataLoader`` ready for training."""

    ds = NeuralDataset(rates, speed, window_size=window_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return loader
