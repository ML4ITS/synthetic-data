from typing import List, Union

from torch import Tensor
from torch.utils.data import Dataset

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.preprocessing import normalize_dataset, preprocess_dataset1D
from synthetic_data.mlops.tools.api import load_dataset


def fetch_dataset(
    name: str, split_ratio: float, split_size: float, normalize: bool = False
) -> Union[Tensor, List[Tensor]]:
    cfg = RemoteConfig()
    dataset = load_dataset(cfg, name)
    dataset = preprocess_dataset1D(dataset, split_ratio, split_size)
    if normalize:
        dataset = normalize_dataset(dataset)
    return dataset


class HarmonicDataset(Dataset):
    def __init__(self, name, split_size, split_ratio):
        self.name = name
        self.split_size = split_size
        self.split_ratio = split_ratio
        self.dataset = fetch_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
