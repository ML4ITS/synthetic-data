import collections
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import LBFGS

from db import database as db
from models.lstm import LSTM
from utils.common import name_to_alias
from utils.logger import CustomFormatter
from utils.modelling import (
    reshape_and_split,
    normalize_dataset,
    vizualize_and_save_prediction,
)

# create logger with '__name__'
logger = logging.getLogger("Training")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


def validate_model(
    model: nn.Module,
    criterion: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    future: int,
):
    with torch.no_grad():
        prediction = model(x_test, future=future)
        loss = criterion(prediction[:, :-future], y_test)
        logger.info(f"Loss/test: {loss.item()}")
        return prediction.detach().numpy(), loss


def save_to_database_on_improvement(
    top_loss: float,
    val_loss: float,
    name: str,
    dataset: str,
    arch: str,
    state_dict: collections.OrderedDict,
) -> float:
    if val_loss < top_loss:
        top_loss = val_loss
        doc = db.update_model(name, dataset, arch, state_dict)
        if doc.modified_count > 0:
            logger.warning(f"Updated model: {name}")
    return top_loss


def load_or_create_model(name: str, arch: str) -> nn.Module:
    document = db.load_model(name)
    if document is None:
        cls = model_registry[arch]
        model = cls()
        model.double()
        document = db.create_model(MODEL_NAME, DATASET_NAME, arch, model.state_dict())
        if document.acknowledged:
            logger.warning(f"Created model: {name}")
        else:
            logger.critical(f"Failed to create model: {name}")
    else:
        logger.warning(f"Loaded model: {name}")
        cls = model_registry[document["arch"]]
        model = cls()
        model.double()
        model.load_state_dict(document["state_dict"])
    return model


if __name__ == "__main__":
    torch.manual_seed(1337)
    np.random.seed(1337)

    model_registry = {"LSTM": LSTM}

    # user preference
    ARCH = "LSTM"
    MODEL_NAME = "My LSTM 2"
    DATASET_NAME = "Quick Harmon"

    # hyperparams
    EPOCHS = 8
    LR_RATE = 0.75
    BATCH_SIZE = 20
    SPLIT_RATIO = 0.3
    TIMESTEP_PREDICTIONS = 500

    # paths/dirs
    PATH_TO_OUTPUTS = "out"
    PATH_TO_MODELS = "models"
    PATH_TO_DATASETS = "modelling"
    MODEL_ALIAS = name_to_alias(MODEL_NAME)
    OUTPUT_DIR = PATH_TO_OUTPUTS + f"/{MODEL_ALIAS}"

    # dataset
    dataset = db.load_time_series_as_numpy(DATASET_NAME)
    dataset = reshape_and_split(dataset, ratio=SPLIT_RATIO, batch_size=BATCH_SIZE)
    x_train, y_train, x_test, y_test = normalize_dataset(dataset)

    # params
    model = load_or_create_model(MODEL_NAME, ARCH)
    criterion = MSELoss()
    optimizer = LBFGS(model.parameters(), lr=LR_RATE)

    def closure():
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        logger.info(f"Loss/train: {loss.item()}")
        loss.backward()
        return loss

    top_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch: {epoch}")
        optimizer.step(closure)
        val_predictions, val_loss = validate_model(
            model, criterion, x_test, y_test, future=TIMESTEP_PREDICTIONS
        )
        vizualize_and_save_prediction(
            outdir=OUTPUT_DIR,
            val_predictions=val_predictions,
            n_samples=x_train.size(1),
            future=TIMESTEP_PREDICTIONS,
            epoch=epoch,
        )
        top_loss = save_to_database_on_improvement(
            top_loss,
            val_loss,
            name=MODEL_NAME,
            dataset=DATASET_NAME,
            arch=ARCH,
            state_dict=model.state_dict(),
        )
