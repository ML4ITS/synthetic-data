from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from flask import jsonify
from flask.wrappers import Response


def error_response(
    msg: str, code: int, stacktrace: Exception = None
) -> Tuple[Response, int]:

    """Helper function for creating error responses.

    Args:
        msg (str): the error message
        code (int): the error code
        stacktrace (Exception): the stacktrace of the error
    Returns:
        flask.wrappers.Response: the error response
    """
    response = {"error": msg}
    if stacktrace is not None:
        response["stacktrace"] = str(stacktrace)
    return jsonify(response), code


def preprocess_data(data: Union[float, list]) -> torch.Tensor:
    raise DeprecationWarning("This function is deprecated")

    if isinstance(data, float):
        return _build_scalar_data(data)
    elif isinstance(data, list):
        return _build_vector_data(data)
    else:
        raise ValueError("Data must be a float or a list")


def _build_scalar_data(data: float) -> torch.Tensor:
    raise DeprecationWarning("This function is deprecated")
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
    raise DeprecationWarning("This function is deprecated")
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
