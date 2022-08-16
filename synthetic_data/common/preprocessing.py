from typing import List, Tuple, Union

import numpy as np
import torch


def normalize(data: torch.Tensor, minval: int, maxval: int) -> torch.Tensor:
    """Normalizes data between minval and maxval

    Args:
        data (torch.Tensor): the data to normalize
        minval (int): the minimum value
        maxval (int): the maximum value

    Returns:
        torch.Tensor: the normalized data
    """
    return (maxval - minval) * (
        (data - data.min()) / (data.max() - data.min())
    ) + minval


def normalize_dataset(
    dataset: Union[torch.Tensor, List[torch.Tensor]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Checks if dataset is a list or a single tensor and normalizes it accordingly

    Raises:
        TypeError: if dataset is not a list or a single tensor

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: the normalized dataset / list of datasets
    """
    if isinstance(dataset, torch.Tensor):
        return normalize(dataset, -1, 1)
    elif isinstance(dataset, list):
        return [normalize(d, -1, 1) for d in dataset]
    else:
        raise TypeError(f"Data must be provided as torch.Tensor (/list)")


def preprocess_dataset1D(
    data: np.ndarray, split_ratio: float, split_size: int
) -> Tuple[torch.Tensor]:
    """Reshapes the data (np.ndarray) from (1 x N) to (split_size, N/split_size) and
    splits it into train and test sets based on the split_ratio provided and returns
    its as as Tensors

    e.g.: use split_ratio=1, split_size=1 to return the data as is
    """
    _validate_preprocess_params(split_ratio, split_size)

    data = reshape_data(data, split_size)
    data = split_data(data, split_ratio)
    data = to_tensor(data)
    return data


def preprocess_dataset2D(data: np.ndarray) -> Tuple[torch.Tensor]:
    """_summary_

    Args:
        data (np.ndarray): _description_

    Returns:
        Tuple[torch.Tensor]: _description_
    """
    data = to_tensor(data)
    return data


def reshape_data(
    data: Union[np.ndarray, List[np.ndarray]], split_size: int
) -> Union[np.ndarray, List[np.ndarray]]:
    """Reshapes the data (np.ndarray) from (1 x N) to (split_size, N/split_size)"""
    if isinstance(data, np.ndarray):
        return data.reshape(split_size, -1)
    elif isinstance(data, list):
        return [d.reshape(split_size, -1) for d in data]
    else:
        raise TypeError(f"Data must be provided as np.ndarray")


def to_tensor(
    data: Union[np.ndarray, List[np.ndarray]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return [torch.from_numpy(d) for d in data]
    else:
        raise TypeError(f"Data must be provided as np.ndarray (/list)")


def split_data(
    data: np.ndarray, split_ratio: float
) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """Splits data into train / test sets based on the split_ratio provided
    :returned as tuple
    """
    if split_ratio == 1:
        return data
    split_idx = int(data.shape[0] * split_ratio)
    x_train = data[split_idx:, :-1]
    y_train = data[split_idx:, 1:]
    x_test = data[:split_idx, :-1]
    y_test = data[:split_idx, 1:]
    return x_train, y_train, x_test, y_test


def _validate_preprocess_params(split_ratio: float, split_size: int) -> None:
    if split_ratio < 0 or split_ratio > 1:
        raise ValueError(f"Split ratio must be between 0 and 1, got {split_ratio}")
    if split_size < 1:
        raise ValueError(f"Split size can't be less than 1, got {split_size}")
    elif split_size > 1:
        # check if split_ratio == 1, then we can split as many times as we want
        # if not, then we have to split accordingly to match sizes of x_train, y_train, x_test, y_test.
        if split_ratio != 1 and split_size % 4 != 0:  # x_train, y_train, x_test, y_test
            raise ValueError(f"Split size must be divisible by 4, got {split_size}")
