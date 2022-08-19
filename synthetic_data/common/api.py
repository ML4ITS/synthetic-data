from typing import Any

import numpy as np
import requests

from synthetic_data.common.config import LocalConfig

cfg = LocalConfig()


def save_time_series(name: str, data: np.ndarray, params: dict) -> dict:
    """Converts the parameters to a dictionary and saves the time-series to the database.
       (time-series data aka. the dataset)
    Args:
        name (str): the name of the the time-series data
        data (np.ndarray): the time-series data
        params (dict): the parameters of used to generate the time-series data

    Returns:
        dict: the success response of the request
    """
    payload = {
        "name": name,
        "data": data.tolist(),  # shape (M, N)
        "sample": data[0].tolist(),  # shape (1, N)
        "parameters": params,
    }
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/dataset"
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_all_time_series(limit: int = 100) -> Any:
    """Returns all the time-series from the database with its full 'data' content.

    Args:
        limit (int, optional): how many to fetch. Defaults to 100.

    Returns:
        Any: the response of the request
    """

    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/datasets"
    response = requests.get(ENDPOINT, params={"limit": int(limit)})
    return response.json()


def get_all_time_series_by_sample(limit: int = 100) -> Any:
    """Returns all the time-series from the database with its 'sample' content.

    Args:
        limit (int, optional): how many to fetch. Defaults to 100.

    Returns:
        Any: the response of the request
    """
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/datasets/sample"
    response = requests.get(ENDPOINT, params={"limit": int(limit)})
    return response.json()


def get_registrated_model_names() -> Any:
    """Returns a list of names of all the registrated model in ML Flow model registry

    Returns:
        Any: the response of the request
    """
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/models"
    r = requests.get(ENDPOINT)
    return r.json()


def get_version_by_model_names(model_name: str) -> Any:
    """Returns the a list of versions of the registrated model
    in ML Flow model registry with the given name.

    Args:
        model_name (str): the name of the model

    Returns:
        Any: the response of the request
    """
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/models/versions"
    payload = {"model_name": model_name}
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_prediction(params: dict) -> Any:
    """Returns the prediction of the model with the given parameters.

    Args:
        params (dict): the requested prediction parameters need for the model

    Raises:
        ValueError: if the payload_type is not supported

    Returns:
        Any: the response of the request
    """

    payload_type = params.get("payload_type")
    model_name = params.get("model_name")
    model_version = params.get("model_version")

    if payload_type == "conditional generation":
        z_dim = params.get("z_dim")
        n_classes = params.get("n_classes")
        return get_conditional_generation_prediction(
            model_name, model_version, z_dim, n_classes
        )
    elif payload_type == "generation":
        z_dim = params.get("z_dim")
        n_samples = params.get("n_samples")
        return get_generation_prediction(model_name, model_version, z_dim, n_samples)
    elif payload_type == "forecast":
        timesteps = params.get("timesteps")
        data = params.get("data")
        return get_forecast_prediction(model_name, model_version, timesteps, data)
    else:
        raise ValueError("Invalid payload type")


def get_conditional_generation_prediction(
    model_name: str, model_version: int, z_dim: int, n_classes: int
) -> Any:
    """Returns the conditional generation prediction of the model with the given parameters.

    Args:
        model_name (str): the name of the model
        model_version (int): the version of the model
        z_dim (int): the dimension of the latent space
        n_classes (int): the number of classes trained on the model

    Returns:
        Any: the response of the request
    """

    PAYLOAD_TYPE = "conditional generation"
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/predict"

    payload = {
        "payload_type": PAYLOAD_TYPE,
        "model_name": model_name,
        "model_version": model_version,
        "z_dim": z_dim,
        "n_classes": n_classes,
    }
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_generation_prediction(
    model_name: str, model_version: int, z_dim: int, n_samples: int
) -> Any:
    """Returns the generation prediction of the model with the given parameters.

    Args:
        model_name (str): the name of the model
        model_version (int): the version of the model
        z_dim (int): the dimension of the latent space
        n_samples (int): the number of samples to generate

    Returns:
        Any: the response of the request
    """

    PAYLOAD_TYPE = "generation"
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/predict"

    payload = {
        "payload_type": PAYLOAD_TYPE,
        "model_name": model_name,
        "model_version": model_version,
        "z_dim": z_dim,
        "n_samples": n_samples,
    }
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_forecast_prediction():
    raise NotImplementedError
