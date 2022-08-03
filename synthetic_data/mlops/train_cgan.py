import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from ray import tune
from torch.utils.data import DataLoader

from synthetic_data.common.torchutils import get_device
from synthetic_data.mlops.models.cgan import Discriminator, Generator


class Trainer:
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        optimG: torch.optim.Optimizer,
        optimD: torch.optim.Optimizer,
        criterion: torch.nn.modules.loss,
        params: Dict[str, Any],
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.optimG = optimG
        self.optimD = optimD
        self.criterion = criterion
        self.params = params
        self.losses = {"lossG": [], "lossD": []}
        self.device = get_device()
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        for i, batch in enumerate(dataloader):
            real_sequences = batch[0].to(self.device)
            real_labels = batch[1].to(self.device)
            batch_size = real_sequences.shape[0]
            target_shape = (batch_size,)

            true_targets = torch.ones(target_shape, device=self.device)
            fake_targets = torch.zeros(target_shape, device=self.device)

            noise_shape = (batch_size, self.params["z_dim"])
            rand_noise = torch.randn(noise_shape, device=self.device)
            n_classes = self.params["n_classes"]
            rand_targets = torch.randint(0, n_classes, target_shape, device=self.device)

            # Generate fake sequences
            fake_sequences = self.generator(rand_noise, rand_targets)
            logits = self.discriminator(fake_sequences, rand_targets)
            lossG = self.criterion(logits, true_targets)

            self.optimG.zero_grad()
            lossG.backward()
            self.optimG.step()
            self.losses["lossG"].append(lossG.data.item())

            # Discriminate on real sequences
            logits = self.discriminator(real_sequences, real_labels)
            lossD_real = self.criterion(logits, true_targets)
            # Discriminate on fake sequences
            logits = self.discriminator(fake_sequences, rand_targets)
            lossD_fake = self.criterion(logits, fake_targets)
            lossD = (lossD_real + lossD_fake) / 2

            self.optimD.zero_grad()
            lossD.backward()
            self.optimD.step()
            self.losses["lossD"].append(lossD.data.item())

            if i % self.params["print_every"] == 0:
                global_step = i + epoch * len(dataloader.dataset)
                tune.report(
                    epoch=epoch,
                    global_step=global_step,
                    lossG=self.losses["lossG"][-1],
                    lossC=self.losses["lossC"][-1],
                )

    def train(self, dataloader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            self._train_epoch(dataloader, epoch + 1)
            self._eval_and_savefig(epoch)
            self._save_statedict(epoch)

    @torch.no_grad()
    def _eval_and_savefig(self, epoch):
        epoch_dir = os.path.join("outputs", str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        n_classes = self.params["n_classes"]

        noise = torch.randn((n_classes, self.args["z_dim"]), device=self.device)
        labels = torch.arange(n_classes, device=self.device)

        self.generator.eval()
        sequences = self.generator(noise, labels)
        self.generator.train()

        fig, axis = plt.subplots(
            nrows=n_classes,
            ncols=1,
            figsize=(14, 12),
            dpi=100,
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for i in range(n_classes):
            sequence = sequences[i]
            axis[i].plot(sequence, label=f"{i+1} Hz")
            axis[i].legend(loc="upper right")

        fig.suptitle("Generated sequences")
        fig.supxlabel("Time steps")
        fig.supylabel("Amplitude")
        fig.savefig(os.path.join(epoch_dir, "sequences.png"))

    def _save_statedict(self, epoch):
        if epoch % self.params["checkpoint_frequency"] == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "d_state_dict": self.discriminator.state_dict(),
                    "g_state_dict": self.generator.state_dict(),
                    "d_opt_state_dict": self.optimC.state_dict(),
                    "g_opt_state_dict": self.optimG.state_dict(),
                },
                f"checkpoints/epoch_{epoch}.pkl",
            )


def run_training_session(config, dataset=None):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    SPLIT_SIZE = None  # TODO
    SPLIT_RATIO = None  # TODO
    DATASET_NAME = None  # TODO

    dataset = None  # TODO
    dataloader = None  # TODO

    n_classes = None  # TODO
    latent_dim = None  # TODO
    print_every = None  # TODO
    seq_length = None  # TODO

    generator = Generator(seq_length, n_classes, latent_dim)
    generator.to(device)

    discriminator = Discriminator(seq_length, n_classes)
    discriminator.to(device)

    criterion = torch.nn.MSELoss()
    optimG = torch.optim.Adam(None, lr=None, betas=None)
    optimD = torch.optim.Adam(None, lr=None, betas=None)

    params = None  # TODO

    trainer = Trainer(generator, discriminator, optimG, optimD, criterion, params)
    trainer.train(dataloader, epochs=config["epochs"])


if __name__ == "__main__":

    NUM_TRIAL_RUNS = None  # TODO
    EXPERIMENT_NAME = None  # TODO
    RESOURCES_PER_TRIAL = None  # TODO

    config = None  # TODO
    analysis = None  # TODO
