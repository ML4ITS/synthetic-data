from mlflow.tracking import MlflowClient


def print_all_registrated_models(client: MlflowClient) -> None:
    """Print all registered models

    Args:
        client (MlflowClient): the mlflow client
    """
    registered_models = client.list_registered_models()
    for registered_model in registered_models:
        print(repr(registered_model))


def find_latest_registrated_model_version(client: MlflowClient, name: str) -> str:
    """Find the latest registered model version for a given name

    Args:
        client (MlflowClient): the mlflow client
        name (str): the name of the registered model

    Raises:
        ValueError: if no registered model is found for the given name

    Returns:
        str: the latest registered model version
    """
    latest_model = None
    latest_version = -1
    for rm in client.search_model_versions("name='{}'".format(name)):
        if int(rm.version) > latest_version:
            latest_model = rm
            latest_version = int(rm.version)
    if latest_model is None:
        raise ValueError(f"No model found with name={name}")
    return latest_model
