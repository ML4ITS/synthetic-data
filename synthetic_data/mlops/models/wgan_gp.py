"""
Wasserstein GAN implementation (assumed to be used with Gradient Penalty)

inspiration from https://github.com/eriklindernoren/PyTorch-GAN
"""
from torch import Tensor, nn


class Generator(nn.Module):
    def __init__(self, seq_length: int, z_dim: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, seq_length),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return self.model(sequence)


class Discriminator(nn.Module):
    def __init__(self, seq_length: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(seq_length, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return self.model(sequence)


if __name__ == "__main__":

    from synthetic_data.mlops.tools.summary import summarize_gan

    summarize_gan(Generator, Discriminator)
