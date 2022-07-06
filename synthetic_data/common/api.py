import numpy as np
import pandas as pd
import requests

from synthetic_data.common.config import LocalConfig

cfg = LocalConfig()


def save_time_series(name: str, ts: pd.DataFrame, params: dict) -> dict:
    payload = {
        "name": name,
        "x": ts["x"].tolist(),
        "y": ts["y"].tolist(),
        "parameters": params,
    }
    ENDPOINT = cfg.API_URI + "/dataset"
    response = requests.post(ENDPOINT, json=payload)
    return response.json()


def get_all_time_series(limit: int = 100) -> list:
    ENDPOINT = cfg.API_URI + "/datasets"
    response = requests.get(ENDPOINT, params={"limit": limit})
    return response.json()


def get_model_meta():
    ENDPOINT = cfg.API_URI + "/models"
    r = requests.get(ENDPOINT)
    return r.json()


def get_forecast_meta(model_name, model_version, timesteps, data):
    ENDPOINT = cfg.API_URI + "/forecast"
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "timesteps": timesteps,
        "data": data,
    }
    r = requests.post(ENDPOINT, json=payload)
    return r.json()
