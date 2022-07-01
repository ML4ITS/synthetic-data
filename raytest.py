import os

import torch
import mlflow
from mlflow.tracking import MlflowClient

from db import database as db
from utils.modelling import (
    reshape_and_split,
    normalize_dataset,
    move_to_device,
    vizualize_and_view_prediction,
    vizualize_dataset,
)
from utils.mlflow import (
    delete_all_registrated_models,
    print_all_registrated_models,
    find_latest_registrated_model_version,
    delete_registrated_model,
)
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


ML_HOST = os.environ["ML_HOST"]
ML_PORT = os.environ["ML_PORT"]
mlflow.set_tracking_uri(f"http://{ML_HOST}:{ML_PORT}")
mlflow.set_registry_uri(f"http://{ML_HOST}:{ML_PORT}")

client = MlflowClient()
print("Connected to ML Flow: ", mlflow.get_registry_uri())

model = find_latest_registrated_model_version(client, "lstm_model")

device = torch.device("cpu")

MODEL_NAME = model.name
MODEL_VERSION = model.version

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
print(f"Loading model from {model_uri}")

model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
model.double()
model.to(device)
model.eval()


# Load dataset
SPLIT_SIZE = 5
SPLIT_RATIO = 0.3
DATASET_NAME = "Simple Vibes"

dataset = db.load_time_series_as_numpy(DATASET_NAME)
dataset = reshape_and_split(dataset, split_ratio=SPLIT_RATIO, split_size=SPLIT_SIZE)
dataset = normalize_dataset(dataset)
x_train, y_train, x_test, y_test = move_to_device(data=dataset, device=device)

vizualize_dataset(x_train)

# Evaluate
future = 1000

predictions = model(x_test, future=future)

predictions = predictions.detach().numpy()
n_samples = x_test.size(1)

vizualize_and_view_prediction(predictions, n_samples, future)
