import mlflow
from mlflow.tracking import MlflowClient
import torch

# ML Flow
MLFLOW_URI = f"http://78.91.106.131:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_registry_uri(MLFLOW_URI)
ml_client = MlflowClient()

print("Connected to ML Flow: ", mlflow.get_registry_uri())

latest_model = None
latest_version = -1
device = torch.device("cpu")

for rm in ml_client.search_model_versions("name='{}'".format("lstm_model")):
    if int(rm.version) > latest_version:
        print("Found model: ", rm.name, rm.version)
        latest_model = rm
        latest_version = int(rm.version)
if latest_model is None:
    raise ValueError(f"No model found with name=lstm_model")

MODEL_NAME = latest_model.name
MODEL_VERSION = latest_model.version

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
print(f"Loading model = {model_uri}")

model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
