import numpy as np
import pandas as pd
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


def get_model_meta():
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/models"
    r = requests.get(ENDPOINT)
    return r.json()


def get_forecast_meta(model_name, model_version, timesteps, data):
    ENDPOINT = cfg.URI_BACKEND_LOCAL + "/forecast"
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "timesteps": timesteps,
        "data": data,
    }
    r = requests.post(ENDPOINT, json=payload)
    return r.json()
