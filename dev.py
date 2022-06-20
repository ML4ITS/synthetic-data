import logging

import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import LBFGS
from db.database import load_time_series_as_numpy
from utils.common import name_to_alias

from utils.modelling import (
    load_and_split,
    normalize_data,
    vizualize_and_save_prediction,
)


class LSTM(nn.Module):
    def __init__(self, hidden_layers: int = 64):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, x_train: np.ndarray, future: int = 0):
        outputs = []
        h_t = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)
        c_t = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)
        h_t2 = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)
        c_t2 = torch.zeros(x_train.size(0), self.hidden_layers, dtype=torch.double)

        for input_t in x_train.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for _ in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        return torch.cat(outputs, dim=1)


def validate(
    model: nn.Module,
    criterion: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    future: int,
):
    with torch.no_grad():
        prediction = model(x_test, future=future)
        loss = criterion(prediction[:, :-future], y_test)
        log.info("Loss/test:, %s", loss.item())
        return prediction.detach().numpy(), loss


def save_on_improvement(model, best_loss, running_loss, model_path):
    if running_loss < best_loss:
        best_loss = running_loss
        torch.save(model.state_dict(), model_path)
    return best_loss


if __name__ == "__main__":

    log = logging.getLogger(__name__)

    torch.manual_seed(1337)
    np.random.seed(1337)

    ## Adjust these parameters
    DATASET_NAME = "Quick Harmon"
    EPOCHS = 8
    LR_RATE = 0.75
    BATCH_SIZE = 20
    SPLIT_RATIO = 0.3
    TIMESTEP_PREDICTIONS = 500

    # --

    PATH_TO_OUTPUTS = "out"
    PATH_TO_MODELS = "models"
    PATH_TO_DATASETS = "modelling"
    DATASET_ALIAS = name_to_alias(DATASET_NAME)

    output_path = PATH_TO_OUTPUTS + f"/{DATASET_ALIAS}"
    model_path = PATH_TO_MODELS + f"/{DATASET_ALIAS}.pt"

    dataset = load_time_series_as_numpy(DATASET_NAME)

    dataset = load_and_split(dataset, ratio=SPLIT_RATIO, batch_size=BATCH_SIZE)
    x_train, y_train, x_test, y_test = normalize_data(dataset)

    model = LSTM()
    model.double()

    criterion = MSELoss()
    optimizer = LBFGS(model.parameters(), lr=LR_RATE)

    def closure():
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        log.info("Loss/train:, %s", loss.item())
        loss.backward()
        return loss

    top_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        log.info("Epoch:, %s", epoch)
        optimizer.step(closure)
        val_predictions, val_loss = validate(
            model, x_test, y_test, future=TIMESTEP_PREDICTIONS
        )
        vizualize_and_save_prediction(
            outdir=output_path,
            val_predictions=val_predictions,
            n_samples=x_train.size(1),
            future=TIMESTEP_PREDICTIONS,
            epoch=epoch,
        )
        top_loss = save_on_improvement(top_loss, val_loss, model_path=model_path)
