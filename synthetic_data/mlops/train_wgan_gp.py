import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import ray
import torch
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.autograd import grad as torch_grad
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.torchutils import get_device
from synthetic_data.mlops.datasets.multiharmonic import MultiHarmonicDataset
from synthetic_data.mlops.models.wgan_gp import Discriminator, Generator
from synthetic_data.mlops.models.wgan_gp import __name__ as model_script
from synthetic_data.mlops.tools.summary import summarize_gan
from synthetic_data.mlops.transforms.transform import RandomRoll


class Trainer:
    def __init__(
        self,
        modelG: torch.nn.Module,
        modelD: torch.nn.Module,
        optimG: RMSprop,
        optimC: RMSprop,
        params: Dict[str, Any],
    ):
        self.modelG = modelG
        self.modelD = modelD
        self.optimG = optimG
        self.optimC = optimC
        self.params = params
        self.losses: Dict[str, list] = {
            "lossG": [],
            "lossD": [],
            "gradPenalty": [],
            "gradNorm": [],
        }
        self.device = get_device()
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    def _train_critic(self, real_data: torch.Tensor) -> None:

        noise = self._generate_noise(batch_size=real_data.size(0))

        fake_data = self.modelG(noise)
        real_logits = self.modelD(real_data)
        fake_logits = self.modelD(fake_data)

        gradient_penalty = self._gradient_penalty(real_data, fake_data)

        loss = fake_logits.mean() - real_logits.mean() + gradient_penalty

        self.optimC.zero_grad()
        loss.backward()
        self.optimC.step()

        self.losses["gradPenalty"].append(gradient_penalty.data.item())
        self.losses["lossD"].append(loss.data.item())

    def _train_generator(self, data):
        noise = self._generate_noise(batch_size=data.size(0))

        fake_data = self.modelG(noise)
        fake_logits = self.modelD(fake_data)
        loss = -fake_logits.mean()

        self.optimG.zero_grad()
        loss.backward()
        self.optimG.step()
        self.losses["lossG"].append(loss.data.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = interpolated.clone().detach().requires_grad_(True)

        # Pass interpolated data through Critic
        prob_interpolated = self.modelD(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        # Gradients have shape (batch_size, num_channels, series length),
        # here we flatten to take the norm per example for every batch
        gradients = gradients.view(batch_size, -1)
        self.losses["gradNorm"].append(gradients.norm(2, dim=1).mean().data.item())

        # Derivatives of the gradient close to 0 can cause problems because of the
        # square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

        return self.params["gp_weight"] * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        for i, (data, _) in enumerate(dataloader):
            global_step = i + epoch * len(dataloader.dataset)
            data = data.to(self.device)

            self._train_critic(data.float())

            # Update generator every critic_iterations iterations
            if global_step % self.params["critic_iterations"] == 0:
                self._train_generator(data)

            # LOG every 20th sample
            if global_step % 20 == 0:
                tune.report(
                    epoch=epoch,
                    global_step=global_step,
                    lossG=self.losses["lossG"][-1],
                    lossD=self.losses["lossD"][-1],
                    grad_penality=self.losses["gradPenalty"][-1],
                    grad_norm=self.losses["gradNorm"][-1],
                )

    def _generate_noise(self, batch_size: int = 1) -> torch.Tensor:
        noise_shape = (batch_size, self.params["z_dim"])
        return torch.randn(noise_shape, device=self.device)

    def train(self, dataloader: DataLoader, epochs: int) -> None:
        fixed_latents = self._generate_noise()

        for epoch in range(epochs):
            self._train_epoch(dataloader, epoch)
            # self._save_statedict(epoch)
            # Sample a new distribution to check for mode collapse
            if epochs % 5 == 0:
                dynamic_latents = self._generate_noise()
                self._eval_and_savefig_plt(epoch, fixed_latents, dynamic_latents)

        self._save_losses()

    @torch.no_grad()
    def _eval_and_savefig_plt(self, epoch, fixd_latent, dyn_latent):
        # Generate fake data using both fixed and dynamic latents

        def save_plot(data: torch.Tensor, name: str) -> None:
            plt.figure(figsize=(10, 3), dpi=100)
            plt.plot(data.numpy()[0].T)
            plt.savefig(f"outputs/{name}.png")
            plt.xlabel("Time steps")
            plt.ylabel("Amplitude")
            plt.close()

        self.modelG.eval()
        fake_data_fixed_latents = self.modelG(fixd_latent).cpu().data
        fake_data_dynamic_latents = self.modelG(dyn_latent).cpu().data
        save_plot(fake_data_fixed_latents, name=f"fix_{epoch}")
        save_plot(fake_data_dynamic_latents, name=f"dyn_{epoch}")
        self.modelG.train()

    def _save_losses(self) -> None:
        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(self.losses["lossG"], label="Generator")
        plt.plot(self.losses["lossD"], label="Critic")
        plt.legend()
        plt.savefig("outputs/losses.png")
        plt.close()

    def _save_statedict(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "modelG": self.modelG.state_dict(),
                "modelD": self.modelD.state_dict(),
                "optimG": self.optimG.state_dict(),
                "optimD": self.optimC.state_dict(),
            },
            f"checkpoints/epoch_{epoch}.pkl",
        )


def run_training_session(config):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    dataset_names: List[str] = [
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

    dataset_labels: Dict[str, int] = {
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
        gp_weight=10,
        critic_iterations=5,
        seq_length=dataset.get_sequence_length(),
    )

    modelG = Generator(params["seq_length"], params["z_dim"])
    modelG.to(device)

    modelD = Discriminator(params["seq_length"])
    modelD.to(device)

    summarize_gan(Generator, Discriminator)

    optimG = Adam(modelG.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    optimC = Adam(modelD.parameters(), lr=config["lr"], betas=(0.5, 0.999))

    trainer = Trainer(modelG, modelD, optimG, optimC, params)
    trainer.train(dataloader, epochs=config["epochs"])


if __name__ == "__main__":

    # MAX_GPU = 8
    # MAX_CPU = 16

    NUM_TRIAL_RUNS = 1
    EXPERIMENT_NAME = "wgan-gp_experiment"
    RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 1}

    config = {
        "lr": tune.choice([0.0002]),
        "epochs": tune.choice([200]),
        "batch_size": tune.grid_search([128]),
    }

    ray.init()
    cfg = RemoteConfig()
    mlflow.set_tracking_uri(cfg.URI_MODELREG_REMOTE)

    analysis = tune.run(
        run_training_session,
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
