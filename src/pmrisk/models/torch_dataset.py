"""PyTorch dataset for sequence-window classification"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from pmrisk.features.sequence import extract_window


class SequenceWindowDataset(Dataset):
    """Returns (X_window, y) for a (engine_id, end_pos) sequence index"""

    def __init__(
        self,
        engine_arrays: dict[int, dict[str, np.ndarray]],
        index: list[dict],
        window_length: int,
    ) -> None:
        if window_length < 1:
            raise ValueError("window_length must be >= 1")

        for entry in index:
            if "engine_id" not in entry or "end_pos" not in entry:
                raise ValueError("index entries must contain engine_id and end_pos")

            engine_id = entry["engine_id"]
            end_pos = entry["end_pos"]

            if engine_id not in engine_arrays:
                raise ValueError(f"engine_id {engine_id} not found in engine_arrays")

            arrays = engine_arrays[engine_id]
            if "y" not in arrays:
                raise ValueError(f"engine_arrays[{engine_id}] must contain 'y'")

            t = len(arrays["y"])
            if end_pos < 0 or end_pos >= t:
                raise ValueError(f"end_pos {end_pos} out of range for engine {engine_id} (T={t})")

            if end_pos < window_length - 1:
                raise ValueError(f"end_pos {end_pos} < window_length - 1 for engine {engine_id}")

        self.engine_arrays = engine_arrays
        self.index = index
        self.window_length = window_length

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        entry = self.index[idx]
        engine_id = entry["engine_id"]
        end_pos = entry["end_pos"]

        x_win = extract_window(self.engine_arrays, engine_id, end_pos, self.window_length)
        y_val = self.engine_arrays[engine_id]["y"][end_pos]

        x_tensor = torch.from_numpy(x_win)  # already float32
        y_tensor = torch.as_tensor(y_val, dtype=torch.int64)

        return x_tensor, y_tensor
