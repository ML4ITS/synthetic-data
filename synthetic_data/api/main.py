import json

import mlflow
import torch
from bson import json_util
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo
from mlflow.tracking import MlflowClient

from synthetic_data.api import utils
from synthetic_data.common.config import LocalConfig

cfg = LocalConfig()
app = Flask(__name__)
app.config["MONGO_URI"] = cfg.URI_DATABASE
mongo = PyMongo(app)

model_registry = utils.ModelRegistry()

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
        return utils.error_response("No names found", 404)


@app.route("/del", methods=["GET"])
def delete_timeseries():
    info = mongo.db.datasets.delete_many({})
    return jsonify({"deleted": info.deleted_count})


@app.route("/datasets", methods=["GET"])
def get_datasets():
    if request.method == "GET":
        """returns a list of all the time-series in the database"""
        limit = request.args.get("limit", None, type=int)
        datasets = list(mongo.db.datasets.find(limit=limit))
        if len(datasets) == 0:
            return utils.error_response("No dataset found", 404)
        modified_datasets = []
        for dataset in datasets:
            dataset = json_util.dumps(dataset)
            dataset = json.loads(dataset)
            modified_datasets.append(dataset)
        return jsonify({"datasets": modified_datasets})


@app.route("/datasets/sample", methods=["GET"])
def get_dataset_samples():
    if request.method == "GET":
        """returns a sample list of all the time-series in the database"""
        limit = request.args.get("limit", None, type=int)

        # retrieve the datasets without the main 'data' content,
        # we just want parse the the sample data + meta info.
        cursor = mongo.db.datasets.find(limit=limit, projection={"data": False})
        datasets = list(cursor)

        if len(datasets) == 0:
            return utils.error_response("No dataset found", 404)

        dataset_samples = []
        for dataset in datasets:
            dataset = json_util.dumps(dataset)
            dataset = json.loads(dataset)
            dataset_samples.append(dataset)
        return jsonify({"datasets": dataset_samples})


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
            return utils.error_response(f"Could get dataset {name}", 500, e)

    elif request.method == "POST":
        """Saves a time-series to the database"""
        dataset = request.get_json()
        try:
            result = mongo.db.datasets.insert_one(dataset)
            return jsonify({"id": str(result.inserted_id)}), 201
        except Exception as e:
            return utils.error_response("Could not save dataset", 500, e)


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


# "/inference/forecast" ?
@app.route("/predict", methods=["GET", "POST"])
def get_forecast():

    PAYLOAD_TYPES = ["forecast", "conditional generation", "generation"]

    if request.method == "POST":
        payload = request.get_json()
        payload_type = payload["payload_type"]

        if payload_type not in PAYLOAD_TYPES:
            return utils.error_response(
                f"Payload type {payload_type} not supported", 400
            )

        model_name = payload["model_name"]
        model_version = payload["model_version"]
        model = model_registry.load_model(model_name, model_version)

        if payload_type == "forecast":
            raise NotImplementedError

        elif payload_type == "conditional generation":
            z_dim = payload["z_dim"]
            n_classes = payload["n_classes"]

            labels = torch.arange(n_classes)
            noise = torch.randn((n_classes, z_dim))
            sequences = model(noise, labels)
            response = sequences.detach().numpy().tolist()

        elif payload_type == "generation":
            raise NotImplementedError

        return jsonify({"response": response})


if __name__ == "__main__":
    app.run("0.0.0.0", cfg.BACKEND_PORT, debug=False)
