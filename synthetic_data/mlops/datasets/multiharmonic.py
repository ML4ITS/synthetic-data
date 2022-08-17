import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_data.common.config import RemoteConfig
from synthetic_data.mlops.tools.api import load_dataset


def _fetch_dataset(
    datset_names: List[str], label_map: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:

    all_labels = []
    all_datasets = []
    cfg = RemoteConfig()

    # Assuming that the names are in the same order as the labels.
    # Also, using chronological indices as 'labels' helps a lot for simplicity.
    for datset_name, dataset_label in zip(datset_names, label_map.values()):
        t1 = time.monotonic()
        dataset = load_dataset(cfg, datset_name)
        t2 = time.monotonic()
        print(f"Downloaded {datset_name} in {t2 - t1:.3f} seconds")
        assert dataset.ndim == 2, f"MultiHarmonicDataset should be 2-D"
        labels = np.array([dataset_label] * dataset.shape[0])
        all_datasets.append(dataset)
        all_labels.append(labels)

    all_labels = np.concatenate(all_labels, axis=0)
    all_datasets = np.concatenate(all_datasets, axis=0)

    return torch.from_numpy(all_datasets), torch.from_numpy(all_labels)


class MultiHarmonicDataset(Dataset):

    _name = "MultiHarmonicDataset"

    _COMPUTE_FOLDER = "/data/summer_internship22/datasets"
    _path_to_dataset = os.path.join(_COMPUTE_FOLDER, _name)

    def __init__(
        self,
        names: List[str],
        label_map: Dict[str, int],
        transforms: torch.nn.Module = None,
        save: bool = False,
        local_dir: str = None,
    ) -> None:
        super().__init__()
        self.names = names
        self.label_map = label_map
        self.transforms = transforms

        if local_dir is not None:
            self.savedir = os.path.join(local_dir, self._name)
        else:
            self.savedir = os.path.join(self._COMPUTE_FOLDER, self._name)

        _path_to_data = os.path.join(self.savedir, "data.pt")
        _path_to_labels = os.path.join(self.savedir, "labels.pt")

        if os.path.exists(_path_to_data) and os.path.exists(_path_to_labels):
            print(f"Loaded data locally from {self.savedir}")
            self.data = torch.load(_path_to_data)
            self.labels = torch.load(_path_to_labels)
        else:
            print("Requesting data from API...")
            self.data, self.labels = _fetch_dataset(self.names, self.label_map)
            if save:
                os.makedirs(self.savedir, exist_ok=True)
                torch.save(self.data, _path_to_data)
                torch.save(self.labels, _path_to_labels)
                print(f"Saved data to {self.savedir}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        batch = self.data[index], int(self.labels[index])

        if self.transforms is not None:
            batch = self.transforms(batch)

        return batch

    def get_sequence_length(self) -> int:
        return self.data.shape[1]
