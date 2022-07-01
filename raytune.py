import os
import warnings
from functools import partial

import mlflow
import numpy as np
import ray
import torch
from dotenv import find_dotenv, load_dotenv
from utils.ray import MlflowModelRegister
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.nn import MSELoss
from torch.optim import LBFGS

from db import database as db
from models.lstm import LSTM
from utils.modelling import (
    get_device,
    reshape_and_split,
    move_to_device,
    normalize_dataset,
    vizualize_and_save_prediction,
)

warnings.filterwarnings("ignore")
load_dotenv(find_dotenv())


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
MODEL_NAME = "My LSTM"
DATASET_NAME = "Simple Vibes"

# (RARE) USER INPUT: Should we auto-adjust this?
EXPERIMENT_NAME = "lstm_experiment"
SPLIT_SIZE = 5
SPLIT_RATIO = 0.3
RESOURCES_PER_TRIAL = {"cpu": 4, "gpu": 1}
ML_HOST = os.environ["ML_HOST"]
ML_PORT = os.environ["ML_PORT"]
ML_SERVER = f"http://{ML_HOST}:{ML_PORT}"

# ---
dataset = db.load_time_series_as_numpy(DATASET_NAME)
dataset = reshape_and_split(dataset, split_ratio=SPLIT_RATIO, split_size=SPLIT_SIZE)
dataset = normalize_dataset(dataset)


config = {
    "hidden_layers": tune.choice([64, 96, 128]),
    "lr": tune.choice(np.arange(0.55, 1, 0.1, dtype=float).round(2).tolist()),
    "epochs": tune.choice([2]),
    "future": tune.choice([500]),
}

ray.init()
mlflow.set_tracking_uri(ML_SERVER)

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
            tracking_uri=ML_SERVER,
            registry_uri=ML_SERVER,
            experiment_name=EXPERIMENT_NAME,
            save_artifact=True,
        )
    ],
)

registrator = MlflowModelRegister(EXPERIMENT_NAME)
registrator.register(MODEL_NAME)
