"""
Conditional Wasserstein GAN implementation (see arch in wgan_gp.py)

"""

from torch import Tensor, cat, nn


class Generator(nn.Module):
    def __init__(self, seq_length: int, n_classes: int, z_dim: int) -> None:
        super().__init__()
        self.embedder = nn.Embedding(n_classes, n_classes)
        input_channels = z_dim + n_classes
        output_channels = seq_length

        self.model = nn.Sequential(
            nn.Linear(input_channels, 128),
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
            nn.Linear(1024, output_channels),
            nn.Tanh(),
        )

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        label_embedding = self.embedder(labels)
        noise = cat((label_embedding, noise), -1)
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self, seq_length: int, n_classes: int) -> None:
        super().__init__()
        self.embedder = nn.Embedding(n_classes, n_classes)
        input_channels = seq_length + n_classes
        output_channels = 1

        self.model = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, output_channels),
        )

    def forward(self, seq: Tensor, labels: Tensor) -> Tensor:
        label_embedding = self.embedder(labels)
        seq = cat((seq, label_embedding), -1).float()
        out = self.model(seq)
        return out.squeeze()


if __name__ == "__main__":

    from synthetic_data.mlops.tools.summary import summarize_conditional_gan

    summarize_conditional_gan(Generator, Discriminator, n_classes=10)
