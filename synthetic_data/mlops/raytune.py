from functools import partial

import mlflow
import numpy as np
import ray
import torch
from common import api
from common.config import Config
from common.preprocessing import normalize_dataset, reshape_and_split
from common.torchutils import get_device, move_to_device
from common.vizu import vizualize_and_save_prediction
from models.lstm import LSTM
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from tools.model_register import MlflowModelRegister
from torch.nn import MSELoss
from torch.optim import LBFGS


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
NUM_TRIAL_RUNS = None  # replace
MODEL_NAME = None  # replace
DATASET_NAME = None  # replace
SHOULD_REGISTER = None  # replace

# (RARE) USER INPUT: Should we auto-adjust this?
EXPERIMENT_NAME = None  # replace
SPLIT_SIZE = None  # replace
SPLIT_RATIO = None  # replace
RESOURCES_PER_TRIAL = None  # replace

# ---
dataset = api.load_time_series_as_numpy(DATASET_NAME)
dataset = reshape_and_split(dataset, split_ratio=SPLIT_RATIO, split_size=SPLIT_SIZE)
dataset = normalize_dataset(dataset)


config = None  # replace

ray.init()
cfg = Config()
mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)

analysis = tune.run(
    partial(run_training_session, dataset=dataset),
    name=EXPERIMENT_NAME,
    mode=None,  # replace
    verbose=None,  # replace
    num_samples=NUM_TRIAL_RUNS,
    log_to_file=["stdout.txt", "stderr.txt"],
    resources_per_trial=RESOURCES_PER_TRIAL,
    sync_config=tune.SyncConfig(syncer=None),
    local_dir="/tmp/ray_runs",
    config=config,
    callbacks=[
        MLflowLoggerCallback(
            tracking_uri=cfg.MLFLOW_TRACKING_URI,
            registry_uri=cfg.MLFLOW_REGISTRY_URI,
            experiment_name=EXPERIMENT_NAME,
            save_artifact=True,
        )
    ],
)

if SHOULD_REGISTER:
    registrator = MlflowModelRegister(EXPERIMENT_NAME)
    registrator.register(MODEL_NAME)
