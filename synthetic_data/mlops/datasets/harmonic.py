from typing import List, Union

from torch import Tensor
from torch.utils.data import Dataset

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.preprocessing import normalize_dataset, preprocess_dataset1D
from synthetic_data.mlops.tools.api import load_dataset

_DEPR_MSG = "HarmonicDataset is deprecated. Use MultiHarmonic instead."


class HarmonicDataset(Dataset):
    def __init__(self, name, split_size, split_ratio):
        # self.name = name
        # self.split_size = split_size
        # self.split_ratio = split_ratio
        self.data = None
        raise DeprecationWarning(_DEPR_MSG)

    def __len__(self):
        raise DeprecationWarning(_DEPR_MSG)

    def __getitem__(self, idx):
        raise DeprecationWarning(_DEPR_MSG)
