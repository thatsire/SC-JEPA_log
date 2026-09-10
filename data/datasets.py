import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

CONT_COLS = ["volt", "rotate", "pressure", "vibration", "age"]
NUM_MODELS = 4
FEATURE_NAMES = CONT_COLS + [f"model_{i}" for i in range(NUM_MODELS)]


class PredictiveMaintenanceDataset(Dataset):
    """
    Sliding windows over hourly telemetry, built per machine so windows never mix machines.

    All feature normalisation happens here, once, with a StandardScaler fitted on the
    training file. Pre-training and downstream training therefore see exactly the same
    input distribution (this replaces the per-window instance normalisation that was
    applied only in pre-training). `age` is standardised like the other continuous
    columns and `model` is one-hot encoded.

    mode='pretrain'   -> (x_past, x_future), each (num_patches, patch_len, C)
                         past = first half of the window, future = second half.
    mode='downstream' -> (x, y) with x (num_patches, patch_len, C) covering `window_size`
                         hours and y = 1 if any failure occurs in the next
                         `forecast_horizon` hours (no overlap with the input).
    """

    def __init__(self, csv_path, mode="pretrain", window_size=24, forecast_horizon=12,
                 patch_len=6, stride=1, scaler=None):
        assert mode in ("pretrain", "downstream")
        self.mode = mode
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.stride = stride
        input_hours = window_size // 2 if mode == "pretrain" else window_size
        assert input_hours % patch_len == 0, "input hours must be a multiple of patch_len"

        df = pd.read_csv(csv_path)
        df = df.sort_values(["machineID", "datetime"]).reset_index(drop=True)

        if scaler is None:
            scaler = StandardScaler().fit(df[CONT_COLS].values)
        self.scaler = scaler
        cont = scaler.transform(df[CONT_COLS].values).astype(np.float32)
        onehot = np.eye(NUM_MODELS, dtype=np.float32)[df["model"].values.astype(int)]
        self.X = np.ascontiguousarray(np.concatenate([cont, onehot], axis=1))  # (rows, C)

        self.label_cols = [c for c in df.columns if c.startswith("fail_")]
        self.Y = df[self.label_cols].values.max(axis=1).astype(np.float32)  # any failure at that hour

        self.starts = self._build_starts(df["machineID"].values)

    # ------------------------------------------------------------------ helpers
    def _build_starts(self, machine_ids):
        margin = self.window_size + (0 if self.mode == "pretrain" else self.forecast_horizon)
        bounds = np.flatnonzero(np.diff(machine_ids)) + 1
        starts = []
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(machine_ids)]):
            n = hi - lo
            if n >= margin:
                starts.extend(range(lo, lo + n - margin + 1, self.stride))
        return np.asarray(starts, dtype=np.int64)

    @property
    def in_channels(self):
        return self.X.shape[1]

    @property
    def num_patches(self):
        hours = self.window_size // 2 if self.mode == "pretrain" else self.window_size
        return hours // self.patch_len

    def window_labels(self):
        """Binary label of every window (downstream mode), vectorised. Used for class weights."""
        assert self.mode == "downstream"
        fut_max = sliding_window_view(self.Y, self.forecast_horizon).max(axis=1)
        return fut_max[self.starts + self.window_size].astype(np.int64)

    def windows(self):
        """All input windows as an array (n_windows, window_size, C). Used by the baselines."""
        view = sliding_window_view(self.X, self.window_size, axis=0)  # (rows-W+1, C, W)
        return np.ascontiguousarray(view[self.starts].transpose(0, 2, 1))

    # ------------------------------------------------------------------ Dataset API
    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        W, L = self.window_size, self.patch_len
        x = torch.from_numpy(self.X[s:s + W])  # (W, C)
        C = x.shape[1]
        if self.mode == "pretrain":
            half = W // 2
            return x[:half].view(half // L, L, C), x[half:].view(half // L, L, C)
        y = self.Y[s + W: s + W + self.forecast_horizon].max()
        return x.view(W // L, L, C), torch.tensor(int(y), dtype=torch.long)
