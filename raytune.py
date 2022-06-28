import os
from functools import partial

import mlflow
import numpy as np
import ray
import torch
from mlflow.tracking import MlflowClient
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.nn import MSELoss
from torch.optim import LBFGS

from db import database as db
from models.lstm import LSTM
from utils.modelling import (
    get_device,
    load_and_split,
    normalize_dataset,
    vizualize_and_save_prediction,
)


def train(model, opt, criterion, x_train, y_train):
    model.train()

    def closure():
        opt.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        return loss

    opt.step(closure)


def test(model, criterion, x_test, y_test):
    future = 500
    with torch.no_grad():
        prediction = model(x_test, future=future)
        loss = criterion(prediction[:, :-future], y_test)
        return prediction.detach().cpu().numpy(), loss.detach().cpu().numpy()


def run_training_session(config, dataset=None):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    x_train, y_train, x_test, y_test = (
        dataset[0].to(device),
        dataset[1].to(device),
        dataset[2].to(device),
        dataset[3].to(device),
    )

    model = LSTM(hidden_layers=config["hidden_layers"])
    model.double()
    model.to(device)

    criterion = MSELoss()
    optimizer = LBFGS(model.parameters(), lr=config["lr"])

    EPOCHS = None  # replace
    TIMESTEP_PREDICTIONS = 500
    top_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train(model, optimizer, criterion, x_train, y_train)
        vpred, vloss = test(model, criterion, x_test, y_test)
        tune.report(validation_loss=float(vloss))
        vizualize_and_save_prediction(
            outdir="plots",
            val_predictions=vpred,
            n_samples=x_train.size(1),
            future=TIMESTEP_PREDICTIONS,
            epoch=epoch,
        )

        if vloss < top_loss:
            top_loss = vloss
            save_path = os.path.join(tune.get_trial_dir(), "model.pt")
            torch.jit.save(torch.jit.trace(model, x_train), save_path)


EXPERIMENT_NAME = "lstm_experiment"
NUM_SAMPLES = None  # replace

BATCH_SIZE = 10
SPLIT_RATIO = 0.3

DATASET_NAME = "Simple Vibes"
dataset = db.load_time_series_as_numpy(DATASET_NAME)
dataset = load_and_split(dataset, ratio=SPLIT_RATIO, batch_size=BATCH_SIZE)
dataset = normalize_dataset(dataset)

resources_per_trial = {"cpu": 8, "gpu": 2}

config = {
    "hidden_layers": tune.choice([32, 48, 64, 80, 96]),
    "lr": tune.choice(np.arange(0.5, 1, 0.1, dtype=float).round(1).tolist()),
}

ray.init()

ML_HOST = None  # replace
ML_PORT = None  # replace
ML_SERVER = f"http://{ML_HOST}:{ML_PORT}"

mlflow.set_tracking_uri(ML_SERVER)

analysis = tune.run(
    partial(run_training_session, dataset=dataset),
    name=EXPERIMENT_NAME,
    mode="min",
    verbose=0,
    num_samples=NUM_SAMPLES,
    log_to_file=["stdout.txt", "stderr.txt"],
    resources_per_trial=resources_per_trial,
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

cli = MlflowClient()
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

assert exp is not None

runs = cli.search_runs(experiment_ids=[exp.experiment_id])

best_validation_loss = float("inf")
best_experiment_run = None

for idx, run in enumerate(runs):

    if not run.data.metrics:
        # no more data left
        break

    if "done" in run.data.metrics:
        # check if the trial signal is completed
        if bool(run.data.metrics["done"]):
            break

    if "best_run" not in run.data.tags:
        if best_validation_loss > run.data.metrics["validation_loss"]:
            best_experiment_run = run
            best_validation_loss = run.data.metrics["validation_loss"]

if best_experiment_run is None:
    raise ValueError("Could not find best experiment run")

result = mlflow.register_model(
    model_uri=f"runs:/{best_experiment_run.info.run_id}", name="baseline_lstm"
)

print("# Registrated model")
print(f" Name   : {result.name}")
print(f" Version: {result.version}")
