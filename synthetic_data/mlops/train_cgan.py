import os
from functools import partial
from pprint import pprint
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import ray
import torch
from mlflow.tracking import MlflowClient
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.torchutils import get_device
from synthetic_data.mlops.datasets.multiharmonic import MultiHarmonicDataset
from synthetic_data.mlops.models.cgan import Discriminator, Generator
from synthetic_data.mlops.models.cgan import __name__ as model_script
from synthetic_data.mlops.tools import summary
from synthetic_data.mlops.transforms.transform import RandomRoll


class Trainer:
    def __init__(
        self,
        modelG: torch.nn.Module,
        modelD: torch.nn.Module,
        optimG: Adam,
        optimD: Adam,
        criterion: MSELoss,
        params: Dict[str, Any],
    ):
        self.modelG = modelG
        self.modelD = modelD
        self.optimG = optimG
        self.optimD = optimD
        self.criterion = criterion
        self.params = params
        self.device = get_device()
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        running_gloss = []
        running_dloss = []

        for i, batch in enumerate(dataloader):
            global_step = i + epoch * len(dataloader.dataset)

            real_sequences = batch[0].to(self.device)
            real_labels = batch[1].to(self.device)
            batch_size = real_sequences.shape[0]

            true_targets = torch.ones(real_labels.shape, device=self.device)
            fake_targets = torch.zeros(real_labels.shape, device=self.device)

            noise_shape = (batch_size, self.params["z_dim"])
            rand_noise = torch.randn(noise_shape, device=self.device)
            rand_targets = torch.randint(
                0, self.params["n_classes"], real_labels.shape, device=self.device
            )

            # Generate fake sequences
            fake_sequences = self.modelG(rand_noise, rand_targets)
            logits = self.modelD(fake_sequences, rand_targets)
            lossG = self.criterion(logits, true_targets)

            self.optimG.zero_grad()
            lossG.backward()
            self.optimG.step()
            running_gloss.append(lossG.data.item())

            # Discriminate on real sequences
            logits = self.modelD(real_sequences, real_labels)
            lossD_real = self.criterion(logits, true_targets)
            # Discriminate on fake sequences
            logits = self.modelD(fake_sequences.detach(), rand_targets)
            lossD_fake = self.criterion(logits, fake_targets)
            lossD = (lossD_real + lossD_fake) / 2

            self.optimD.zero_grad()
            lossD.backward()
            self.optimD.step()

            # LOG every 50th sample
            if global_step % 50 == 0:
                global_step = i + epoch * len(dataloader.dataset)
                running_gloss.append(lossG.data.item())
                running_dloss.append(lossD.data.item())

                tune.report(
                    epoch=epoch,
                    global_step=global_step,
                    lossG=lossG.data.item(),
                    lossD=lossD.data.item(),
                )

        avg_gloss = sum(running_gloss) / len(running_gloss)
        avg_dloss = sum(running_dloss) / len(running_dloss)
        return avg_gloss, avg_dloss

    def train(
        self, dataloader: DataLoader, epochs: int, should_registrate: bool = False
    ) -> None:
        avg_gloss = []
        avg_dloss = []

        for epoch in range(epochs):
            avg_g, avg_d = self._train_epoch(dataloader, epoch + 1)
            avg_gloss.append(avg_g)
            avg_dloss.append(avg_d)

            if epochs % 5 == 0:
                self._eval_and_savefig(epoch)
                self._save_statedict(epoch)

        plt.figure(figsize=(14, 12), dpi=300)
        plt.plot(avg_gloss, label="G")
        plt.plot(avg_dloss, label="D")
        plt.legend()
        plt.title("Average batch loss")
        plt.savefig("outputs/avg_losses.png")
        plt.close()

        # Save before exit
        if should_registrate:
            # NB: path needs to match the ending of the model_uri in mlflow.register_model(...)
            mlflow.pytorch.save_model(self.modelG, "model")

    @torch.no_grad()
    def _eval_and_savefig(self, epoch):
        self.modelG.eval()

        n_classes = self.params["n_classes"]
        labels = torch.arange(n_classes, device=self.device)
        noise = torch.randn((n_classes, self.params["z_dim"]), device=self.device)

        sequences = self.modelG(noise, labels)

        fig, axis = plt.subplots(
            nrows=n_classes,
            ncols=1,
            figsize=(14, 12),
            dpi=300,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for i in range(n_classes):
            sequence = sequences[i].detach().numpy()
            axis[i].plot(sequence, label=f"{i+1} Hz")
            axis[i].legend(loc="upper right")

        fig.suptitle("Generated frequencies")
        fig.supxlabel("Time steps")
        fig.supylabel("Amplitude")
        fig.savefig(f"outputs/epoch_{epoch}.png")
        plt.close(fig)
        self.modelG.train()

    def _save_statedict(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "modelG": self.modelG.state_dict(),
                "modelD": self.modelD.state_dict(),
                "optimG": self.optimG.state_dict(),
                "optimD": self.optimD.state_dict(),
            },
            f"checkpoints/epoch_{epoch}.pkl",
        )


def run_training_session(config, should_registrate=None):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    dataset_names = [
        "Harmon 1",
        "Harmon 2",
        "Harmon 3",
        "Harmon 4",
        "Harmon 5",
        "Harmon 6",
        "Harmon 7",
        "Harmon 8",
        "Harmon 9",
        "Harmon 10",
    ]

    dataset_labels = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "7": 6,
        "8": 7,
        "9": 8,
        "10": 9,
    }

    transforms = Compose([RandomRoll(p=-1.0)])

    dataset = MultiHarmonicDataset(
        names=dataset_names,
        label_map=dataset_labels,
        transforms=transforms,
        save=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    params = dict(
        z_dim=100,
        n_classes=10,
        seq_length=dataset.get_sequence_length(),
    )

    modelG = Generator(params["seq_length"], params["n_classes"], params["z_dim"])
    modelG.to(device)

    modelD = Discriminator(params["seq_length"], params["n_classes"])
    modelD.to(device)

    summary.summarize_conditional_gan(Generator, Discriminator, params["n_classes"])

    optimG = Adam(modelG.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    optimD = Adam(modelD.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    criterion = MSELoss()

    trainer = Trainer(modelG, modelD, optimG, optimD, criterion, params)
    trainer.train(
        dataloader, epochs=config["epochs"], should_registrate=should_registrate
    )


if __name__ == "__main__":

    # <SET YOUR CONFIG HERE>
    MODEL_NAME = "C-GAN"
    should_registrate = False

    NUM_TRIAL_RUNS = 1
    EXPERIMENT_NAME = "cgan_experiment"
    RESOURCES_PER_TRIAL = {"cpu": 1, "gpu": 1}

    config = {
        "lr": tune.choice([0.0002]),
        "epochs": tune.choice([300]),
        "batch_size": tune.choice([128]),
    }

    ray.init()
    cfg = RemoteConfig()
    mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)

    analysis = tune.run(
        partial(run_training_session, should_registrate=should_registrate),
        name=EXPERIMENT_NAME,
        metric="lossG",
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

    if should_registrate:
        """This section is for registering the a single model in a single trial with ML FLow.
        This is NOT a good practice for large experiments, as it will register the model in every trial.
        TODO: This should be refactored to a general solution.
        """
        top_trial = analysis.get_best_trial("lossG", "min", "last")

        # Trial name depens on name of training function
        top_trial_name = "run_training_session_" + str(top_trial.trial_id)
        # We use the trial name to correct run
        query = "tags.trial_name = '{}'".format(top_trial_name)
        (top_run,) = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=query,
            max_results=1,
            output_format="list",
        )
        # Finally, we use the RUN_ID to register the model
        mlflow.register_model(f"runs:/{top_run.info.run_id}/model", MODEL_NAME)
