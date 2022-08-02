import os
from functools import partial

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import ray
import torch
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader

from synthetic_data.common.config import RemoteConfig
from synthetic_data.common.torchutils import get_device
from synthetic_data.mlops.models.wgan_gp import Critic, Generator
from synthetic_data.mlops.tmp.data import HarmonicDataset


class Trainer:
    def __init__(
        self,
        generator,
        critic,
        optimG,
        optimC,
        seq_length: int,
        gp_weight: int = 10,
        critic_iterations: int = 5,
        print_every: int = 200,
        checkpoint_frequency: int = 200,
        plotmode: str = "plt",  # "plt" | "bokeh"
    ):
        self.generator = generator
        self.critic = critic
        self.optimG = optimG
        self.optimC = optimC
        self.seq_length = seq_length
        self.losses = {"lossG": [], "lossC": [], "gradPenalty": [], "gradNorm": []}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.checkpoint_frequency = checkpoint_frequency
        self.device = get_device()
        self.plotmode = plotmode

        if self.plotmode == "plt":
            self.fixdir = os.path.join("outputs", "fixed")
            self.dyndir = os.path.join("outputs", "dynamic")
            os.makedirs(self.fixdir, exist_ok=True)
            os.makedirs(self.dyndir, exist_ok=True)
        elif self.plotmode == "bokeh":
            outdir = os.path.join("outputs")
            os.makedirs(outdir, exist_ok=True)
            output_file(f"{outdir}/epochs.html")
            self.bokeh_figures = []
        else:
            raise ValueError(f"Unknown plotmode {self.plotmode}")

    def train_critic(self, real_data):
        batch_size = real_data.size(0)
        noise_shape = (batch_size, self.seq_length)
        noise = torch.randn(noise_shape, device=self.device)

        fake_data = self.generator(noise)

        real_logits = self.critic(real_data)
        fake_logits = self.critic(fake_data)

        gradient_penalty = self._gradient_penalty(real_data, fake_data)

        loss = fake_logits.mean() - real_logits.mean() + gradient_penalty

        self.optimC.zero_grad()
        loss.backward()
        self.optimC.step()

        self.losses["gradPenalty"].append(gradient_penalty.data.item())
        self.losses["lossC"].append(loss.data.item())

    def train_generator(self, data):
        batch_size = data.size(0)
        noise_shape = (batch_size, self.seq_length)
        noise = torch.randn(noise_shape, device=self.device)

        fake_data = self.generator(noise)

        fake_logits = self.critic(fake_data)
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
        prob_interpolated = self.critic(interpolated)

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

        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, dataloader, epoch):
        for i, data in enumerate(dataloader):
            self.num_steps += 1
            data = data.to(self.device)

            self.train_critic(data.float())

            # Only update generator every critic_iterations iterations
            if self.num_steps % self.critic_iterations == 0:
                self.train_generator(data)

            if i % self.print_every == 0:
                global_step = i + epoch * len(dataloader.dataset)

                if self.num_steps > self.critic_iterations:
                    tune.report(
                        epoch=epoch,
                        global_step=global_step,
                        lossG=self.losses["lossG"][-1],
                        lossC=self.losses["lossC"][-1],
                        grad_penality=self.losses["gradPenalty"][-1],
                        grad_norm=self.losses["gradNorm"][-1],
                    )
                else:
                    tune.report(
                        global_step=global_step,
                        lossC=self.losses["lossC"][-1],
                        grad_penality=self.losses["gradPenalty"][-1],
                        grad_norm=self.losses["gradNorm"][-1],
                    )

    def train(self, dataloader, epochs):
        noise_shape = (1, self.seq_length)
        fixed_latents = torch.randn(noise_shape, device=self.device)

        for epoch in range(epochs):
            self._train_epoch(dataloader, epoch + 1)

            if epoch % self.checkpoint_frequency == 0:
                self.save_statedict(epoch)

            if epoch % self.print_every == 0:
                # Sample a different region of the latent distribution to check for mode collapse
                dynamic_latents = torch.randn(noise_shape, device=self.device)

                if self.plotmode == "plt":
                    self.eval_and_savefig_plt(epoch, fixed_latents, dynamic_latents)
                elif self.plotmode == "bokeh":
                    self.eval_and_savefig_bokeh(epoch, fixed_latents, dynamic_latents)
        save(gridplot(self.bokeh_figures, ncols=2))

    @torch.no_grad()
    def eval_and_savefig_plt(self, epoch, fixd_latent, dyn_latent):

        self.generator.eval()
        # Generate fake data using both fixed and dynamic latents
        fake_data_fixed_latents = self.generator(fixd_latent).cpu().data
        fake_data_dynamic_latents = self.generator(dyn_latent).cpu().data

        plt.figure()
        plt.plot(fake_data_fixed_latents.numpy()[0].T)
        plt.savefig(f"{self.fixdir}/epoch_{epoch}.png")
        plt.close()

        plt.figure()
        plt.plot(fake_data_dynamic_latents.numpy()[0].T)
        plt.savefig(f"{self.dyndir}/epoch_{epoch}.png")
        plt.close()

        self.generator.train()

    @torch.no_grad()
    def eval_and_savefig_bokeh(self, epoch, fixd_latent, dyn_latent):

        self.generator.eval()
        # Generate fake data using both fixed and dynamic latents
        fake_data_fixed_latents = self.generator(fixd_latent).cpu().data
        fake_data_dynamic_latents = self.generator(dyn_latent).cpu().data

        y_fix_latents = fake_data_fixed_latents.numpy()[0].T
        x_fix_latents = range(len(y_fix_latents))

        y_dyn_latents = fake_data_dynamic_latents.numpy()[0].T
        x_dyn_latents = range(len(y_dyn_latents))

        fig = figure(
            title=f"Epoch {epoch}",
            x_axis_label="Time step",
            y_axis_label="Amplitude",
            aspect_ratio=16 / 9,
        )
        fig.line(x_fix_latents, y_fix_latents, legend_label="Fixed", line_width=2)
        fig.circle(
            x_fix_latents,
            y_fix_latents,
            legend_label="Fixed",
            line_width=2,
            fill_color="blue",
            size=5,
            alpha=0.75,
        )
        self.bokeh_figures.append(fig)

        fig = figure(
            title=f"Epoch {epoch}",
            x_axis_label="Time steps",
            y_axis_label="Amplitude",
            aspect_ratio=16 / 9,
        )
        fig.line(
            x_dyn_latents,
            y_dyn_latents,
            legend_label="Dynamic",
            line_width=2,
            color="orangered",
        )
        fig.circle(
            x_dyn_latents,
            y_dyn_latents,
            legend_label="Dynamic",
            line_width=2,
            fill_color="orangered",
            color="orangered",
            size=5,
            alpha=0.75,
        )
        self.bokeh_figures.append(fig)

        self.generator.train()

    def save_statedict(self, epoch):
        checkdir = "checkpoints"
        os.makedirs(checkdir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "d_state_dict": self.critic.state_dict(),
                "g_state_dict": self.generator.state_dict(),
                "d_opt_state_dict": self.optimC.state_dict(),
                "g_opt_state_dict": self.optimG.state_dict(),
            },
            f"{checkdir}/epoch_{epoch}.pkl",
        )

    def sample_generator(self, latent_shape):
        latent_samples = torch.randn(latent_shape, device=self.device)
        return self.generator(latent_samples)

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        return generated_data.data.cpu().numpy()


def run_training_session(config, dataset=None):
    torch.manual_seed(1337)
    np.random.seed(1337)

    device = get_device()

    SPLIT_SIZE = 50
    SPLIT_RATIO = 1
    DATASET_NAME = "AMP20"

    dataset = HarmonicDataset(DATASET_NAME, SPLIT_SIZE, SPLIT_RATIO)
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], num_workers=4, pin_memory=True
    )

    seq_length = dataset.dataset.shape[1]

    generator = Generator(seq_length=seq_length)
    generator.to(device)

    critic = Critic(seq_length=seq_length)
    critic.to(device)

    optimG = torch.optim.RMSprop(generator.parameters(), lr=config["lr"])
    optimC = torch.optim.RMSprop(critic.parameters(), lr=config["lr"])

    plotmode = "bokeh"

    trainer = Trainer(generator, critic, optimG, optimC, seq_length, plotmode=plotmode)
    trainer.train(dataloader, epochs=config["epochs"])


if __name__ == "__main__":

    NUM_TRIAL_RUNS = 4
    EXPERIMENT_NAME = "wgan-gp_experiment"
    RESOURCES_PER_TRIAL = {"cpu": 1, "gpu": 4}

    config = {
        "lr": tune.choice([0.00005]),
        "epochs": tune.choice([1400]),
        "batch_size": tune.choice([50]),
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
            )
        ],
    )
