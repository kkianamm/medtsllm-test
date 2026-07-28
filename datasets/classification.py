"""
Datasets for whole-sequence time-series classification.

The repo's BaseDataset is built around a single long series with sliding
windows (segmentation / forecasting).  Classification instead needs a set of
independent labelled windows, so these classes subclass torch Dataset directly
while exposing the same attributes the trainer/model rely on
(n_features, n_classes, description, __len__, __getitem__).

Each __getitem__ returns:
    {"x_enc": FloatTensor[L, C], "label": int}

NpzClassificationDataset expects, under config.data.data_dir:
    train.npz, val.npz, test.npz     (val.npz optional -> carved from train)
each containing:
    X : float array [N, L, C]        (N windows, L timesteps, C channels)
    y : int   array [N]              (class ids, 0..n_classes-1)

L must equal config.history_len (== config.pred_len).
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ClassificationDataset(Dataset):
    supported_tasks = ["classification"]
    univariate = False
    clip_dataset = False

    def __init__(self, config, split):
        super().__init__()
        self.config = config
        self.split = split
        self.task = config.task
        self.name = config.data.dataset
        self.history_len = config.history_len

        X, y = self.get_data(split)                       # X:[N,L,C], y:[N]
        X = self._normalize(X, split)

        assert X.shape[1] == self.history_len, (
            f"window length {X.shape[1]} != history_len {self.history_len}; "
            f"set history_len (and pred_len) to {X.shape[1]} in the config."
        )

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self._n_classes = int(self.config.data.get("n_classes", 0)) or int(self.y.max().item() + 1)

        desc = self.config.data.get("description", None)
        self._description = desc or self.__doc__

    # -- to be provided by concrete subclasses -- #
    def get_data(self, split):
        raise NotImplementedError

    def _normalize(self, X, split):
        if not self.config.data.get("normalize", True):
            return X
        N, L, C = X.shape
        if split == "train" or getattr(self, "scaler", None) is None:
            train_X, _ = self.get_data("train")
            self.scaler = StandardScaler().fit(train_X.reshape(-1, train_X.shape[-1]))
        return self.scaler.transform(X.reshape(-1, C)).reshape(N, L, C)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return {"x_enc": self.X[idx], "label": int(self.y[idx].item())}

    @property
    def n_features(self):
        return self.X.shape[-1]

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def description(self):
        return self._description


class NpzClassificationDataset(ClassificationDataset):
    """Generic classification dataset backed by .npz files (X, y)."""

    _cache = {}

    def get_data(self, split):
        key = (self.name, split)
        if key in self._cache:
            return self._cache[key]

        base = Path(self.config.data.data_dir)
        fpath = base / f"{split}.npz"

        if not fpath.exists() and split == "val":
            # carve a validation split out of train (last 15%)
            Xtr, ytr = self.get_data("train")
            n_val = max(1, int(0.15 * len(ytr)))
            rng = np.random.default_rng(self.config.setup.seed)
            perm = rng.permutation(len(ytr))
            val_idx = perm[:n_val]
            X, y = Xtr[val_idx], ytr[val_idx]
            self._cache[key] = (X, y)
            return X, y

        data = np.load(fpath)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        self._cache[key] = (X, y)
        return X, y


npzcls_datasets = {
    "classification": NpzClassificationDataset,
}
