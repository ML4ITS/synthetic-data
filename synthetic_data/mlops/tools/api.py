import numpy as np
import requests

from synthetic_data.common.config import RemoteConfig


def load_dataset(cfg: RemoteConfig, name: str) -> np.ndarray:
    ENDPOINT = cfg.URI_BACKEND_REMOTE + "/dataset"
    response = requests.get(ENDPOINT, params={"name": name})
    response = response.json()
    if "error" in response:
        raise FileNotFoundError(response["error"])
    dataset = response["dataset"]
    dataset = np.array(dataset["data"])
    if dataset.ndim == 1:
        dataset = dataset.reshape(-1, 1)
    return dataset
