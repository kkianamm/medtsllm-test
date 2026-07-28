from abc import ABC
from pathlib import Path

import ast

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .base import BaseDataset


class ClassificationDataset(BaseDataset, ABC):
    """Sequence-level classification base.

    Unlike the per-point datasets, data here is stored as a stack of
    independent recordings ``self.records`` of shape ``[N, T, F]`` with one
    integer label per recording in ``self.record_labels`` (shape ``[N]``).
    Each ``__getitem__`` returns a single (cropped) recording window and its
    label, so the model can produce one prediction per sequence.
    """

    # Subclasses must set these.
    n_classes = 0
    class_names = None

    def load_data(self):
        data = self.get_data()

        records = data["data"]                    # [N, T, F] float
        labels = data["labels"]                   # [N] int

        records = np.asarray(records, dtype=np.float32)
        labels = np.asarray(labels)

        records = self.normalize(records)

        self.records = torch.tensor(records, dtype=torch.float32)
        self.record_labels = torch.tensor(labels, dtype=torch.long)

        self.record_descriptions = data.get("descriptions", None)

    def normalize(self, records):
        if not self.config.data.normalize:
            return records

        if self.normalizer is None:
            train_records = self.records.numpy() if (self.split == "train" and getattr(self, "records", None) is not None) \
                else self.get_data("train")["data"]
            train_records = np.asarray(train_records, dtype=np.float32)
            n, t, f = train_records.shape
            self.normalizer = StandardScaler().fit(train_records.reshape(n * t, f))

        n, t, f = records.shape
        records = self.normalizer.transform(records.reshape(n * t, f)).reshape(n, t, f)
        return records

    def resample_to_history(self, x):
        """Resample a recording to history_len along the time axis.

        Unlike center-cropping, this keeps the entire recording (the whole
        10 s / 1000-sample strip) and just changes its temporal resolution,
        which preserves diagnostic morphology spread across the full signal.
        """
        t = x.size(0)
        h = self.history_len
        if t == h:
            return x
        x = x.transpose(0, 1).unsqueeze(0)                          # [1, F, T]
        x = F.interpolate(x, size=h, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1)                         # [h, F]

    def __len__(self):
        return self.records.size(0)

    def __getitem__(self, idx):
        x = self.resample_to_history(self.records[idx])  # [history_len, F]
        y = self.record_labels[idx]                   # scalar long

        out = {"x_enc": x, "labels": y}

        if self.record_descriptions is not None:
            out["descriptions"] = self.record_descriptions[idx]

        return out

    def inverse_index(self, idx):
        # Not used for classification (no long-series reconstruction), but the
        # base class requires it to be implemented.
        return idx

    @property
    def n_points(self):
        return self.records.size(0) * self.records.size(1)

    @property
    def n_features(self):
        return self.records.size(-1)

    @property
    def class_weights(self):
        counts = torch.bincount(self.record_labels, minlength=self.n_classes).float()
        weights = counts.sum() / (counts.clamp(min=1) * self.n_classes)
        return weights


# Fixed, deterministic ordering of the five PTB-XL diagnostic superclasses.
PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
PTBXL_CLASS_TO_IDX = {c: i for i, c in enumerate(PTBXL_SUPERCLASSES)}


class PTBXLClassificationDataset(ClassificationDataset):

    supported_tasks = ["classification"]
    description = (
        "PTB-XL is a large publicly available dataset of 12-lead ECG recordings, each 10 seconds "
        "long, sampled at 100 Hz. Every recording is labeled with one of five diagnostic superclasses: "
        "Normal ECG, Myocardial Infarction, ST/T Changes, Conduction Disturbance, and Hypertrophy."
    )
    task_description = (
        "Classify the 12-lead ECG recording into one of: Normal ECG, Myocardial Infarction, "
        "ST/T Changes, Conduction Disturbance, or Hypertrophy."
    )

    n_classes = 5
    class_names = PTBXL_SUPERCLASSES

    SAMPLING_RATE = 100   # use the 100 Hz (low-res) records -> 1000 samples per lead

    # strat_fold assignment, matching the predefined patient-level split.
    SPLIT_FOLDS = {
        "train": set(range(1, 9)),   # folds 1-8
        "val":   {9},                # fold 9
        "test":  {10},               # fold 10
    }

    def get_data(self, split=None):
        split = split or self.split

        basepath = Path(__file__).parent / "../data/ptbxl/"
        cache_path = basepath / f"cache_{self.SAMPLING_RATE}hz_{split}.npz"

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            return {
                "data": cached["data"],
                "labels": cached["labels"],
                "descriptions": list(cached["descriptions"]),
            }

        data = self._build_split(basepath, split)

        np.savez_compressed(
            cache_path,
            data=data["data"],
            labels=data["labels"],
            descriptions=np.array(data["descriptions"], dtype=object),
        )
        return data

    def _build_split(self, basepath, split):
        # Lazy import so the framework imports fine without wfdb installed.
        import wfdb

        meta = pd.read_csv(basepath / "ptbxl_database.csv", index_col="ecg_id")
        meta.scp_codes = meta.scp_codes.apply(ast.literal_eval)

        agg_df = pd.read_csv(basepath / "scp_statements.csv", index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_superclass(scp_codes):
            # Sum likelihoods per superclass, then pick the dominant one (single-label).
            scores = {}
            for code, likelihood in scp_codes.items():
                if code in agg_df.index:
                    superclass = agg_df.loc[code, "diagnostic_class"]
                    scores[superclass] = scores.get(superclass, 0.0) + float(likelihood)
            if not scores:
                return None
            return max(scores, key=scores.get)

        meta["superclass"] = meta.scp_codes.apply(aggregate_superclass)

        # Keep only records with a valid superclass in this split's folds.
        folds = self.SPLIT_FOLDS[split]
        sel = meta[meta.superclass.notna() & meta.strat_fold.isin(folds)].copy()

        signals = []
        labels = []
        descriptions = []

        fn_col = "filename_lr" if self.SAMPLING_RATE == 100 else "filename_hr"

        for ecg_id, row in sel.iterrows():
            sig, _ = wfdb.rdsamp(str(basepath / row[fn_col]))
            signals.append(sig.astype(np.float32))            # [T, 12]
            labels.append(PTBXL_CLASS_TO_IDX[row.superclass])
            descriptions.append(self._make_description(row))

        data = np.stack(signals, axis=0)                       # [N, T, 12]
        labels = np.asarray(labels, dtype=np.int64)

        return {"data": data, "labels": labels, "descriptions": descriptions}

    @staticmethod
    def _make_description(row):
        parts = []
        age = row.get("age", None)
        sex = row.get("sex", None)
        if pd.notna(age):
            parts.append(f"age {int(age)}")
        if pd.notna(sex):
            parts.append("male" if int(sex) == 0 else "female")
        if pd.notna(row.get("heart_axis", None)):
            parts.append(f"heart axis {row['heart_axis']}")
        info = ", ".join(parts) if parts else "not available"
        return f"Patient information: {info}."


ptbxl_datasets = {
    "classification": PTBXLClassificationDataset,
}
