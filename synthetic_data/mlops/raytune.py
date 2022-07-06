from functools import partial

import mlflow
import numpy as np
import ray
import torch
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.nn import MSELoss
from torch.optim import LBFGS

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.preprocessing import normalize_dataset, reshape_and_split
from synthetic_data.common.torchutils import get_device, move_to_device
from synthetic_data.common.vizu import vizualize_and_save_prediction
from synthetic_data.mlops.models.lstm import LSTM
from synthetic_data.mlops.tools.api import load_dataset
from synthetic_data.mlops.tools.model_register import MlflowModelRegister


def train(model, opt, criterion, x_train, y_train):
    model.train()

    def closure():
        opt.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        return loss

    opt.step(closure)


def test(model, criterion, x_test, y_test, future):
    with torch.no_grad():
        prediction = model(x_test, future=future)
        loss = criterion(prediction[:, :-future], y_test)
        return prediction.detach().cpu().numpy(), loss.detach().cpu().numpy()


def run_training_session(config, dataset=None):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    x_train, y_train, x_test, y_test = move_to_device(dataset, device)

    model = LSTM(hidden_layers=config["hidden_layers"])
    model.double()
    model.to(device)

    criterion = MSELoss()
    optimizer = LBFGS(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["epochs"] + 1):
        train(model, optimizer, criterion, x_train, y_train)
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
    mlflow.pytorch.save_model(model, "model")


# USER INPUT
NUM_TRIAL_RUNS = 1
MODEL_NAME = "LSTM Baseline"
DATASET_NAME = "Harmonic 20HZ"

# (RARE) USER INPUT: Should we auto-adjust this?
EXPERIMENT_NAME = "lstm_experiment"
SPLIT_SIZE = 40
SPLIT_RATIO = 0.3
RESOURCES_PER_TRIAL = {"cpu": 1, "gpu": 0}

SHOULD_REGISTER = True

# ---
cfg = RemoteConfig()

dataset = load_dataset(cfg, DATASET_NAME)
dataset = reshape_and_split(dataset, split_ratio=SPLIT_RATIO, split_size=SPLIT_SIZE)
dataset = normalize_dataset(dataset)


config = {
    "hidden_layers": tune.choice([64, 96, 128]),
    "lr": tune.choice(np.arange(0.55, 1, 0.1, dtype=float).round(2).tolist()),
    "epochs": tune.choice([12]),
    "future": tune.choice([500]),
}

ray.init()
mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)

analysis = tune.run(
    partial(run_training_session, dataset=dataset),
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
        )
    ],
)

if SHOULD_REGISTER:
    registrator = MlflowModelRegister(EXPERIMENT_NAME)
    registrator.register(MODEL_NAME)
