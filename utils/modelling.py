import json
import os
import tempfile
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mlflow.tracking import MlflowClient


def vizualize_and_save_prediction(
    outdir: str,
    predictions: np.ndarray,
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

    random_sample = predictions[0]

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


def vizualize_and_view_prediction(
    predictions: np.ndarray,
    n_samples: torch.Tensor,
    future: int,
) -> None:
    plt.figure(figsize=(12, 6))
    plt.title(f"Prediction", fontsize=16)

    random_sample = predictions[0]

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
    plt.show()


def vizualize_dataset(data: np.ndarray) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    ax[0].plot(data.reshape(-1))
    ax[0].set_title("Dataset preview", fontsize=16)
    ax[1].plot(data[0])
    ax[1].set_title("Batch preview", fontsize=16)
    plt.show()


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
    # Reshapes the data (np.ndarray) from (1 x N) to (split_size, N/split_size)
    if isinstance(data, np.ndarray):
        return data.reshape(split_size, -1)
    else:
        raise TypeError(f"Data must be provided as np.ndarray")


def _split_data(
    data: Union[torch.Tensor, np.ndarray], ratio: float = 0.2
) -> Tuple[torch.Tensor]:
    # splits it into train and test sets based on the split_ratio provided and returns
    # its as a tuple of torch.Tensors
    split_idx = int(data.shape[0] * ratio)
    x_train = torch.from_numpy(data[split_idx:, :-1])
    y_train = torch.from_numpy(data[split_idx:, 1:])
    x_test = torch.from_numpy(data[:split_idx, :-1])
    y_test = torch.from_numpy(data[:split_idx, 1:])
    return x_train, y_train, x_test, y_test


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def move_to_device(
    data: Union[torch.Tensor, List[torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [d.to(device) for d in data]
    else:
        raise TypeError(
            f"Data must be provided as torch.Tensor or list of torch.Tensor"
        )


def _load_json(dir: str, file) -> dict:
    with open(os.path.join(dir, file)) as f:
        return json.load(f)


def _load_state_dict(
    dir: str, file: str, map_location: torch.device
) -> torch.nn.Module:
    return torch.load(os.path.join(dir, file), map_location=map_location)


def fetch_and_load_state_dict(
    client: MlflowClient, run_id: str, src: str = ""
) -> torch.nn.Module:
    with tempfile.TemporaryDirectory() as tmpdir:
        path_to_files = client.download_artifacts(run_id, src, tmpdir)
        params = _load_json(path_to_files, "params.json")
        state_dict = _load_state_dict(
            path_to_files, "model.pt", map_location=get_device()
        )
        return state_dict, params
