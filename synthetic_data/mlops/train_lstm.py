import os
from functools import partial
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import ray
import torch
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.torchutils import get_device, move_to_device
from synthetic_data.common.vizu import vizualize_and_save_prediction
from synthetic_data.mlops.datasets.multiharmonic import MultiHarmonicDataset
from synthetic_data.mlops.models.lstm import LSTM
from synthetic_data.mlops.models.lstm import __name__ as model_script
from synthetic_data.mlops.tools import summary
from synthetic_data.mlops.tools.model_register import MlflowModelRegister
from synthetic_data.mlops.transforms.transform import RandomRoll


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optim: Adam,
        criterion: MSELoss,
        params: Dict[str, Any],
    ):
        raise NotImplementedError("Out of order")
        self.model = model
        self.optim = optim
        self.criterion = criterion
        self.params = params
        self.device = get_device()
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    def train(self, dataloader: DataLoader, epochs: int) -> None:
        losses = []

        for epoch in range(epochs):
            avg_loss = self._train_epoch(dataloader, epoch + 1)
            val_loss = self._test
            if epochs % 5 == 0:
                self._test
                self._eval_and_savefig(epoch)
                self._save_statedict(epoch)

        plt.figure(figsize=(14, 12), dpi=300)
        plt.plot(losses, label="LSTM")
        plt.legend()
        plt.title("Average batch losses")
        plt.savefig("outputs/avg_losses.png")
        plt.close()

    def test(self, dataloader: DataLoader, epoch: int) -> None:
        for i, batch in enumerate(dataloader):
            global_step = i + epoch * len(dataloader.dataset)

            sequences = batch[0].to(self.device)
            batch_size = sequences.shape[0]

            x_test = sequences[:, :-1]
            y_test = sequences[:, 1:]

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        #    def train(model, opt, criterion, x_train, y_train):
        avg_loss = []
        for i, batch in enumerate(dataloader):
            global_step = i + epoch * len(dataloader.dataset)

            sequences = batch[0].to(self.device)
            batch_size = sequences.shape[0]

            x_train = sequences[:, :-1]
            y_train = sequences[:, 1:]

            out = self.model(x_train)
            loss = self.criterion(out, y_train)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # report
            tune.report(loss=loss.item(), step=global_step)
            avg_loss.append(loss.item())
        return sum(avg_loss) / len(avg_loss)

    def _test(self, x_test: torch.Tensor, y_test: torch.Tensor, future: int) -> None:
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x_test, future=future)
            loss = self.criterion(prediction[:, :-future], y_test)
            n_samples = x_test.size(1)
            self.vizualize_and_save_prediction(prediction, n_samples, future)
        self.model.train()
        return loss.detach().cpu().numpy()

    def vizualize_and_save_prediction(
        self,
        predictions: torch.Tensor,
        n_samples: int,
        future: int,
        epoch: int,
    ) -> None:
        plt.figure(figsize=(30, 10), dpi=100)
        plt.title(f"Epoch {epoch}", fontsize=40)
        plt.xlabel("Time steps", fontsize=30)
        plt.ylabel("Amplitude", fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        random_sample = predictions[0].detach().numpy()

        # Actual time-series
        plt.plot(
            np.arange(n_samples),
            random_sample[:n_samples],
            "b",
            linewidth=3,
        )
        # Forecasted time-series
        plt.plot(
            np.arange(n_samples, n_samples + future),
            random_sample[n_samples:],
            "b:",
            linewidth=3,
        )
        plt.savefig(f"outputs/epoch_{epoch}.png")
        plt.close()

    def _save_statedict(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
            },
            f"checkpoints/epoch_{epoch}.pkl",
        )


def run_training_session(config):
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = get_device()

    dataset = None  # TODO

    x_train, y_train, x_test, y_test = move_to_device(dataset, device)

    model = LSTM(hidden_layers=config["hidden_layers"])
    model.double()
    model.to(device)

    criterion = MSELoss()
    optim = Adam(model.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    # optimizer = LBFGS(model.parameters(), lr=config["lr"])
    # trainer = Trainer(model, optim, criterion, config)
    # trainer.train(dataloader, config["epochs"])


if __name__ == "__main__":
    pass

    # TODO: create working trainer class
    # TODO: make model train on the new MultiHarmonicDataset
    # TODO: change the training loop to use the new MultiHarmonicDataset
    # TODO: change the optimizer to Adam (more common, easier to understand etc)
