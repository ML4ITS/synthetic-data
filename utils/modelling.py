import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


def vizualize_and_save_prediction(
    outdir: str,
    val_predictions: np.ndarray,
    n_samples: torch.Tensor,
    future: int,
    epoch: int,
) -> None:
    plt.figure(figsize=(30, 10), dpi=100)
    plt.title(f"Epoch {epoch}", fontsize=40)
    plt.xlabel("x", fontsize=30)
    plt.ylabel("y", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    random_sample = val_predictions[0]

    # Actual time-series
    plt.plot(
        np.arange(n_samples),
        random_sample[:n_samples],
        "b",
        linewidth=3,
    )
    # Forecasted time-series
    plt.plot(
        np.arange(n_samples, n_samples + future),
        random_sample[n_samples:],
        "b:",
        linewidth=3,
    )
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/epoch{epoch}.png")
    plt.close()


def normalize(data: torch.Tensor, minval: int, maxval: int):
    """Normalize data to range [minval, maxval]"""
    return (maxval - minval) * (
        (data - data.min()) / (data.max() - data.min())
    ) + minval


def normalize_dataset(dataset: List[torch.Tensor]) -> List[torch.Tensor]:
    """Normalize a list of data to range [minval, maxval]"""
    return [normalize(data, minval=-1, maxval=1) for data in dataset]


def load_and_split(
    data: Union[str, np.ndarray], ratio: float = 0.2, batch_size: int = 100
) -> Tuple[torch.Tensor]:
    """Split data into train and test sets by ratio,
    if dataset is a single vector, it is split into batches of size batch_size"""
    data = _load_data(data, batch_size)
    return _split_data(data, ratio)


def _split_data(
    data: Union[torch.Tensor, np.ndarray], ratio: float = 0.2
) -> Tuple[torch.Tensor]:
    split_idx = int(data.shape[0] * ratio)
    x_train = torch.from_numpy(data[split_idx:, :-1])
    y_train = torch.from_numpy(data[split_idx:, 1:])
    x_test = torch.from_numpy(data[:split_idx, :-1])
    y_test = torch.from_numpy(data[:split_idx, 1:])
    return x_train, y_train, x_test, y_test


def _load_data(
    data: Union[str, np.ndarray], batch_size: int = 100
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(data, str):  # if only path is provided
        if data.endswith(".pt"):
            return torch.load(data)
        else:
            return torch.load(data + ".pt")
    elif isinstance(data, np.ndarray):  # if data is already a numpy array
        return data.reshape(batch_size, -1)  # reshape to batch_size x n_samples
    else:
        raise TypeError(f"Data must be provided as numpy array or path w/o .pt suffix")
