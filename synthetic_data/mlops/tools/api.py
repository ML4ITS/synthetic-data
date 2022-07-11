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
    dataset = np.array(dataset["y"])  # only need the y-value (?)
    return dataset.reshape(1, -1)  # returns with shape (1, N)
