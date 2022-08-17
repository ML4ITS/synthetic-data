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


def run_training_session(config, dataset=None):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    x_train, y_train, x_test, y_test = move_to_device(dataset, device)

    model = LSTM(hidden_layers=config["hidden_layers"])
    model.double()
    model.to(device)

    criterion = MSELoss()
    optim = Adam(model.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    # optimizer = LBFGS(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["epochs"] + 1):
        train(model, optim, criterion, x_train, y_train)
        validation_predictions, validation_loss = test(
            model, criterion, x_test, y_test, config["future"]
        )
        tune.report(validation_loss=float(validation_loss))
        # TODO: Early stopping
        vizualize_and_save_prediction(
            outdir="plots",
            predictions=validation_predictions,
            n_samples=x_test.size(1),
            future=config["future"],
            epoch=epoch,
        )

    # Save on exit
    # mlflow.pytorch.save_model(model, "model")


# config = {
#     "hidden_layers": tune.choice([64, 96, 128]),
#     "lr": tune.choice(np.arange(0.55, 1, 0.1, dtype=float).round(2).tolist()),
#     "epochs": tune.choice([2]),
#     "future": tune.choice([500]),
# }

if __name__ == "__main__":

    # USER INPUT
    MODEL_NAME = "LSTM Baseline"
    SHOULD_REGISTER = False

    NUM_TRIAL_RUNS = 1
    EXPERIMENT_NAME = "lstm_experiment"
    RESOURCES_PER_TRIAL = {"cpu": 1, "gpu": 1}

    config = {
        "lr": tune.choice([0.0002]),
        "epochs": tune.choice([100]),
        "batch_size": tune.grid_search([32]),
    }

    ray.init()
    cfg = RemoteConfig()
    mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)

    analysis = tune.run(
        partial(run_training_session, dataset=None),
        name=EXPERIMENT_NAME,
        mode="min",
        verbose=0,
        num_samples=NUM_TRIAL_RUNS,
        log_to_file=["stdout.txt", "stderr.txt"],
        resources_per_trial=RESOURCES_PER_TRIAL,
        sync_config=tune.SyncConfig(syncer=None),
        local_dir="/tmp/ray_runs",
        config=config,
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri=cfg.URI_MODELREG_REMOTE,
                registry_uri=cfg.URI_MODELREG_REMOTE,
                experiment_name=EXPERIMENT_NAME,
                save_artifact=True,
                tags={"model_script": model_script},
            )
        ],
    )

    if SHOULD_REGISTER:
        registrator = MlflowModelRegister(EXPERIMENT_NAME)
        registrator.register(MODEL_NAME)
