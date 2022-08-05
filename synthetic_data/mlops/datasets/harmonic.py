from typing import List, Union

from torch import Tensor
from torch.utils.data import Dataset

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.preprocessing import preprocess_dataset1D
from synthetic_data.mlops.tools.api import load_dataset


class HarmonicDataset(Dataset):
    def __init__(self, name, split_size, split_ratio):
        self.name = name
        self.split_size = split_size
        self.split_ratio = split_ratio
        self.dataset = self._load_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _load_dataset(self) -> Union[Tensor, List[Tensor]]:
        cfg = RemoteConfig()
        dataset = load_dataset(cfg, self.name)
        dataset = preprocess_dataset1D(dataset, self.split_ratio, self.split_size)
        # dataset = normalize_dataset(dataset)
        return dataset
