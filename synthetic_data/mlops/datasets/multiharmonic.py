import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_data.common.config import RemoteConfig
from synthetic_data.mlops.tools.api import load_dataset

RAY_DATAFOLDER = "/data/summer_internship22/cgan"
os.makedirs(RAY_DATAFOLDER, exist_ok=True)


class MultiHarmonicDataset(Dataset):

    filename_data = os.path.join(RAY_DATAFOLDER, "data.pt")
    filename_labels = os.path.join(RAY_DATAFOLDER, "labels.pt")

    def __init__(self, names: List[str], label_map: Dict[str, int]) -> None:
        super().__init__()
        self.names = names
        self.label_map = label_map

        if os.path.exists(self.filename_data) and os.path.exists(self.filename_labels):
            print("Loading cached data ...")
            self.data = torch.load(self.filename_data)
            self.labels = torch.load(self.filename_labels)
        else:
            print("Downloading data...")
            self.data, self.labels = self._load_data()
            torch.save(self.data, self.filename_data)
            torch.save(self.labels, self.filename_labels)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, label = self.data[index], int(self.labels[index])
        return data, label

    def get_sequence_length(self) -> int:
        return self.data.shape[1]

    def _load_data(self):
        cfg = RemoteConfig()

        all_labels = []
        all_datasets = []
        # Assuming that the names are in the same order as the labels.
        # Also, using chronological indices as 'labels' helps a lot for simplicity.
        for name, index in zip(self.names, self.label_map.values()):
            t1 = time.monotonic()
            dataset = load_dataset(cfg, name)
            t2 = time.monotonic()
            print(f"Loaded {name} in {t2 - t1:.3f} seconds")
            assert dataset.ndim == 2, f"MultiHarmonicDataset should be 2-D"
            labels = np.array([index] * dataset.shape[0])
            all_datasets.append(dataset)
            all_labels.append(labels)

        all_labels = np.concatenate(all_labels, axis=0)
        all_datasets = np.concatenate(all_datasets, axis=0)

        all_datasets = torch.from_numpy(all_datasets)
        all_labels = torch.from_numpy(all_labels)

        return all_datasets, all_labels
