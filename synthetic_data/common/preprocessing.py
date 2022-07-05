from typing import List, Tuple, Union

import numpy as np
import torch


def normalize(data: torch.Tensor, minval: int, maxval: int):
    """Normalize data to range [minval, maxval]"""
    return (maxval - minval) * (
        (data - data.min()) / (data.max() - data.min())
    ) + minval


def normalize_dataset(dataset: List[torch.Tensor]) -> List[torch.Tensor]:
    """Normalize a list of data to range [-1, 1]"""
    return [normalize(data, minval=-1, maxval=1) for data in dataset]


def reshape_and_split(
    data: np.ndarray, split_ratio: float = 0.2, split_size: int = 100
) -> Tuple[torch.Tensor]:
    """Reshapes the data (np.ndarray) from (1 x N) to (split_size, N/split_size) and
    splits it into train and test sets based on the split_ratio provided and returns
    its as a tuple of torch.Tensors"""
    data = _reshape_data(data, split_size)
    return _split_data(data, split_ratio)


def _reshape_data(data: np.ndarray, split_size: int = 100) -> np.ndarray:
    """Reshapes the data (np.ndarray) from (1 x N) to (split_size, N/split_size)"""
    if isinstance(data, np.ndarray):
        return data.reshape(split_size, -1)
    else:
        raise TypeError(f"Data must be provided as np.ndarray")


def _split_data(
    data: Union[torch.Tensor, np.ndarray], ratio: float = 0.2
) -> Tuple[torch.Tensor]:
    """Splits it into train and test sets based on the split_ratio provided and returns
    its as a tuple of torch.Tensors
    """
    split_idx = int(data.shape[0] * ratio)
    x_train = torch.from_numpy(data[split_idx:, :-1])
    y_train = torch.from_numpy(data[split_idx:, 1:])
    x_test = torch.from_numpy(data[:split_idx, :-1])
    y_test = torch.from_numpy(data[:split_idx, 1:])
    return x_train, y_train, x_test, y_test
