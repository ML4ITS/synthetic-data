from typing import Any, Dict, List, Tuple, Union

import mlflow
import numpy as np
import torch
from flask import jsonify
from flask.wrappers import Response


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.device = torch.device("cpu")

    def load_model(self, model_name: str, version: int) -> Any:
        model_key = (model_name, version)
        if model_key not in self.models:
            self._register_model(model_key)
        return self.models[model_key]

    def delete_model(self) -> None:
        raise NotImplementedError()

    def list_model(self) -> None:
        raise NotImplementedError()

    def _register_model(self, model_key: Tuple[str, int]) -> None:
        print(f"Registering model {model_key}")
        self.models[model_key] = self._prepare_model(model_key)

    def _build_uri(self, model_key: Tuple[str, int]) -> str:
        name, version = model_key
        return f"models:/{name}/{version}"

    def _prepare_model(self, model_key: Tuple[str, int]) -> None:
        print(f"Loading model {model_key}")
        model_uri = self._build_uri(model_key)
        model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model


def error_response(
    msg: str, code: int, stacktrace: Exception = None
) -> Tuple[Response, int]:
    response = {"error": msg}
    if stacktrace is not None:
        response["stacktrace"] = str(stacktrace)
    return jsonify(response), code


def preprocess_data(data: Union[float, list]) -> torch.Tensor:
    if isinstance(data, float):
        return _build_scalar_data(data)
    elif isinstance(data, list):
        return _build_vector_data(data)
    else:
        raise ValueError("Data must be a float or a list")


def _build_scalar_data(data: float) -> torch.Tensor:
    if data < -1 or data > 1:
        raise ValueError("Data must be between -1 and 1")
    dummy = torch.from_numpy(np.array([data], np.float64))
    dummy = dummy.unsqueeze(0)
    return dummy


def _build_vector_data(data: list) -> torch.Tensor:
    raise NotImplementedError("List input is not implemented yet")


def create_label_response(data: torch.Tensor, predictions: torch.Tensor) -> list:
    raise NotImplementedError("TODO")


def create_forecast_response(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    future: int,
) -> Dict[str, List[float]]:

    outputs = outputs.detach().numpy()
    random_sample = outputs[0].reshape(1, -1)

    n_samples = inputs.size(1)
    x1 = np.arange(n_samples)
    y1 = random_sample[0, :n_samples]
    x2 = np.arange(n_samples, n_samples + future)
    y2 = random_sample[0, n_samples:]

    return {
        "x1": x1.tolist(),
        "y1": y1.tolist(),
        "x2": x2.tolist(),
        "y2": y2.tolist(),
    }
