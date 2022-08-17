from typing import Any, Tuple

import mlflow
import torch


class ModelRegistry:
    """This class holds responsibility for loading models from MLflow,
    and serving them for inference requests.
    """

    def __init__(self):
        self.models = {}
        self.device = torch.device("cpu")

    def load_model(self, model_name: str, version: int) -> Any:
        """Loads a model from MLFlow model registry, and returns it.
        Is registrated if not already in the model cache.

        Args:
            model_name (str): name of the model
            version (int): version of the model

        Returns:
            Any: the PyTorch model
        """
        model_key = (model_name, version)
        if model_key not in self.models:
            self._register_model(model_key)
        return self.models[model_key]

    def delete_model(self) -> None:
        raise NotImplementedError()

    def list_model(self) -> None:
        raise NotImplementedError()

    def _register_model(self, model_key: Tuple[str, int]) -> None:
        """Registers a model in the model cache.

        Args:
            model_key (str, int): the key of the model
        """
        self.models[model_key] = self._prepare_model(model_key)

    def _build_uri(self, model_key: Tuple[str, int]) -> str:
        """Builds the URI of the model, as expected by ML Flow.

        Args:
            model_key (str, int): the key of the model

        Returns:
            str: the registration URI of the model
        """
        name, version = model_key
        return f"models:/{name}/{version}"

    def _prepare_model(self, model_key: Tuple[str, int]) -> None:
        """Loads the model from MLFlow, and prepares it for inference.

        Args:
            model_key (str, int): the key of the model

        Returns:
            torch.nn.Module: the model
        """
        model_uri = self._build_uri(model_key)
        model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model
