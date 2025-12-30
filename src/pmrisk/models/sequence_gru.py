"""GRU model for sequence classification"""

from __future__ import annotations

import torch
from torch import nn


class SequenceGRU(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_l: int = 50,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        if n_features <= 0:
            raise ValueError(f"n_features must be > 0, got {n_features}")
        if window_l <= 0:
            raise ValueError(f"window_l must be > 0, got {window_l}")

        self.n_features = int(n_features)
        self.window_l = int(window_l)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        gru_dropout = float(dropout_p) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim == 3, got {x.ndim}")

        _b, l, f = x.shape
        if l != self.window_l:
            raise ValueError(f"Expected L == {self.window_l}, got {l}")
        if f != self.n_features:
            raise ValueError(f"Expected F == {self.n_features}, got {f}")

        _out, h_n = self.gru(x) 
        last_hidden = h_n[-1]    
        return self.fc(last_hidden) 
