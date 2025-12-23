"""Small 1D CNN for binary time-series classification"""

from __future__ import annotations

import torch
from torch import nn


class SequenceCNN(nn.Module):
    """1D CNN that maps (B, L, F) windows to logits (B, 1)"""

    def __init__(
        self,
        n_features: int,
        window_l: int = 50,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if window_l <= 0:
            raise ValueError(f"window_l must be > 0, got {window_l}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden_channels}")
        if not (0.0 <= dropout_p <= 1.0):
            raise ValueError(f"dropout_p must be in [0, 1], got {dropout_p}")

        self.n_features = int(n_features)
        self.window_l = int(window_l)

        padding = kernel_size // 2
        self.backbone = nn.Sequential(
            nn.Conv1d(self.n_features, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
        )
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim == 3, got {x.ndim} with shape {tuple(x.shape)}")

        _b, l, f = x.shape
        if l != self.window_l:
            raise ValueError(f"Expected L == {self.window_l}, got {l} with shape {tuple(x.shape)}")
        if f != self.n_features:
            raise ValueError(f"Expected F == {self.n_features}, got {f} with shape {tuple(x.shape)}")

        x = x.permute(0, 2, 1)
        x = self.backbone(x)          
        x = x.mean(dim=-1)            
        logits = self.head(x)      
        return logits
