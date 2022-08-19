import numpy as np
import requests

from synthetic_data.common.config import RemoteConfig


def load_dataset(cfg: RemoteConfig, name: str) -> np.ndarray:
    """Loads a dataset from the remote server.

    Args:
        cfg (RemoteConfig): the remote config
        name (str): the name of the dataset to load

    Raises:
        FileNotFoundError: if the dataset is not found on the server

    Returns:
        np.ndarray: the dataset
    """
    ENDPOINT = cfg.URI_BACKEND_REMOTE + "/dataset"
    response = requests.get(ENDPOINT, params={"name": name})
    response = response.json()
    if "error" in response:
        raise FileNotFoundError(response["error"])
    params = response["parameters"]
    assert (
        float(params["std_noise"]) == 0.2000
    ), "std_noise should be 0.2000 but is {}".format(params["std_noise"])
    dataset = response["dataset"]
    dataset = np.array(dataset["data"])
    if dataset.ndim == 1:
        dataset = dataset.reshape(-1, 1)
    return dataset
