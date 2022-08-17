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
        "data": data.tolist(),  # shape (X, Y)
        "sample": data[0].tolist(),  # shape (1, Y)
        "parameters": params,
    }
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/dataset"
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_all_time_series(limit: int = 100) -> list:

    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/datasets"
    response = requests.get(ENDPOINT, params={"limit": int(limit)})
    return response.json()


def get_all_time_series_by_sample(limit: int = 100) -> list:
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/datasets/sample"
    response = requests.get(ENDPOINT, params={"limit": int(limit)})
    return response.json()


def get_registrated_model_names() -> Any:
    """returns a list of all the registrated model in ML Flow model registry"""
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/models"
    r = requests.get(ENDPOINT)
    return r.json()


def get_version_by_model_names(model_name: str) -> Any:
    """returns a list of all the registrated model in ML Flow model registry"""
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/models/versions"
    payload = {"model_name": model_name}
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_prediction(params: dict) -> Any:
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
):
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
):
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


def get_forecast_prediction(model_name, model_version, timesteps, data):
    PAYLOAD_TYPE = "forecast"
    raise NotImplementedError
    # ENDPOINT = cfg.URI_BACKEND_LOCAL + "/forecast"
    # payload = {
    #     "model_name": model_name,
    #     "model_version": model_version,
    #     "timesteps": timesteps,
    #     "data": data,
    # }
    # r = requests.post(ENDPOINT, json=payload)
    # return r.json()
