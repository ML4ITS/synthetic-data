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
):
    plt.figure(figsize=(30, 10), dpi=100)
    plt.title(f"Epoch {epoch}", fontsize=30)
    plt.xlabel("x", fontsize=20)
    plt.ylabel("y", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    random_idx = 0
    random_sample = val_predictions[random_idx]

    # Signal
    plt.plot(
        np.arange(n_samples),
        random_sample[:n_samples],
        "b",
        linewidth=2.0,
    )
    # Forecasted
    plt.plot(
        np.arange(n_samples, n_samples + future),
        random_sample[n_samples:],
        "b:",
        linewidth=2.0,
    )
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/epoch{epoch}.png")
    plt.close()


def normalize(data: torch.Tensor, minval: int, maxval: int):
    """Normalize data to range [minval, maxval]"""
    return (maxval - minval) * (
        (data - data.min()) / (data.max() - data.min())
    ) + minval


def normalize_data(datas: List[torch.Tensor]) -> List[torch.Tensor]:
    """Normalize a list of data to range [minval, maxval]"""
    return [normalize(data, minval=-1, maxval=1) for data in datas]


def load_and_split(
    data: Union[str, np.ndarray], ratio: float = 0.2, batch_size: int = 100
) -> Tuple[torch.Tensor]:
    """Split data into train and test sets by ratio,
    if dataset is a single vector, it is split into batches of size batch_size"""
    if isinstance(data, str):
        if data.endswith(".pt"):
            data = torch.load(data)
        else:
            data = torch.load(data + ".pt")
    if data.shape[0] == 1:
        data = data.reshape(batch_size, -1)
    split_idx = int(data.shape[0] * ratio)
    x_train = torch.from_numpy(data[split_idx:, :-1])
    y_train = torch.from_numpy(data[split_idx:, 1:])
    x_test = torch.from_numpy(data[:split_idx, :-1])
    y_test = torch.from_numpy(data[:split_idx, 1:])
    return x_train, y_train, x_test, y_test
