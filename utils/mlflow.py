from pprint import pprint
import mlflow
from mlflow.tracking import MlflowClient


def print_all_registrated_models(client: MlflowClient) -> None:
    registered_models = client.list_registered_models()
    for registered_model in registered_models:
        pprint(repr(registered_model))


def find_latest_registrated_model_version(client: MlflowClient, name: str) -> str:
    latest_model = None
    latest_version = -1
    for rm in client.search_model_versions("name='{}'".format(name)):
        if int(rm.version) > latest_version:
            latest_model = rm
            latest_version = int(rm.version)
    if latest_model is None:
        raise ValueError(f"No model found with name={name}")
    return latest_model


def delete_all_registrated_models(client: MlflowClient) -> None:
    registered_models = client.list_registered_models()
    for registered_model in registered_models:
        client.delete_registered_model(registered_model.name)


def delete_registrated_model(client: MlflowClient, name: str) -> None:
    client.delete_registered_model(name)
