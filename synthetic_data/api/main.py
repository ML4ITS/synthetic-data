import json
from functools import lru_cache
from typing import Dict, List, Union

import mlflow
import numpy as np
import torch
from bson import json_util
from common.config import LocalConfig
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from mlflow.tracking import MlflowClient
from requests import Response

cfg = LocalConfig()
app = Flask(__name__)
app.config["MONGO_URI"] = cfg.URI_DATABASE
mongo = PyMongo(app)

mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)
mlflow.set_registry_uri(cfg.URI_MODELREG_REMOTE)
ml_client = MlflowClient()


@app.route("/")
def index():
    return jsonify({"Hello": "World!"})


@app.route("/ping")
def ping():
    return jsonify(mongo.db.command("ping"))


@app.route("/names", methods=["GET"])
def get_names():
    try:
        names = mongo.db.list_collection_names()
        return jsonify({"names": names})
    except Exception as e:
        return error_response("No names found", 404)


@app.route("/datasets", methods=["GET"])
def get_datasets():
    if request.method == "GET":
        """returns a list of all the time-series in the database"""
        limit = request.args.get("limit", None, type=int)
        datasets = list(mongo.db.datasets.find(limit=limit))
        if len(datasets) == 0:
            return error_response("No dataset found", 404)
        modified_datasets = []
        for dataset in datasets:
            dataset = json_util.dumps(dataset)
            dataset = json.loads(dataset)
            modified_datasets.append(dataset)
        return jsonify({"datasets": modified_datasets})


@app.route("/dataset", methods=["GET", "POST"])
def get_dataset():
    if request.method == "GET":
        name = request.args.get("name", None, type=str)
        dataset = mongo.db.datasets.find_one({"name": name})
        if dataset is None:
            return jsonify({"error": f"No dataset found with name {name}"}), 404
        try:
            dataset = json_util.dumps(dataset)
            dataset = json.loads(dataset)
            return jsonify({"dataset": dataset})
        except Exception as e:
            return error_response(f"Could get dataset {name}", 500, e)

    elif request.method == "POST":
        """Saves a time-series to the database"""
        dataset = request.get_json()
        try:
            result = mongo.db.datasets.insert_one(dataset)
            return jsonify({"id": str(result.inserted_id)}), 201
        except Exception as e:
            return error_response("Could not save dataset", 500, e)


@app.route("/models", methods=["GET"])
def get_models():
    """returns a list of all the registrated model in ML Flow model registry"""
    if request.method == "GET":
        models = []
        try:
            registered_models = ml_client.list_registered_models()
            for registered_model in registered_models:
                # We except only a single latest model from a registered_model
                (latest_version,) = registered_model.latest_versions
                models.append(
                    {
                        "name": registered_model.name,
                        "run_id": latest_version.run_id,
                        "version": latest_version.version,
                        "tags": registered_model.tags,
                    }
                )
            return jsonify({"models": models})
        except Exception as e:
            return (
                jsonify({"error": "Could not get models", "stacktrace": str(e)}),
                500,
            )


@app.route("/forecast", methods=["GET", "POST"])
def get_forecast():
    if request.method == "POST":
        payload = request.get_json()
        model_name = payload["model_name"]
        model_version = payload["model_version"]
        timesteps = payload["timesteps"]
        data = payload["data"]

        try:
            data = _preprocess_data(data)
            model = _load_model(
                f"models:/{model_name}/{model_version}", device=torch.device("cpu")
            )
            predictions = model(data, future=timesteps)
            forecast = _create_forecast_response(data, predictions, timesteps)
            return jsonify({"forecast": forecast})
        except Exception as e:
            return (
                jsonify({"error": "Could not get forecast", "stacktrace": str(e)}),
                500,
            )


def error_response(msg: str, code: int, stacktrace: Exception = None) -> Response:
    response = {"error": msg}
    if stacktrace is not None:
        response["stacktrace"] = str(stacktrace)
    return jsonify(response), code


def _preprocess_data(data: Union[float, list]) -> torch.Tensor:
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


@lru_cache(maxsize=None)
def _load_model(model_uri: str, device: str):
    model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
    model.double()
    model.to(device)
    model.eval()
    return model


def _create_forecast_response(
    inputs: torch.Tensor,
    outputs: np.ndarray,
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


if __name__ == "__main__":
    app.run("0.0.0.0", cfg.BACKEND_PORT, debug=True)
