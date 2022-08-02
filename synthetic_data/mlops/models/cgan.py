# inspired by https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn


class Generator(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int, n_classes: int) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.embedder = nn.Embedding(self.n_classes, self.n_classes)
        input_channels = self.latent_dim + self.n_classes
        output_channels = self.seq_length

        self.model = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_channels),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embedding = self.embedder(labels)
        noise = torch.cat((label_embedding, noise), -1)
        signal = self.model(noise)
        signal = signal.view(signal.size(0), self.seq_length)
        return signal


class Discriminator(nn.Module):
    def __init__(self, seq_length: int, latent_dim: int, n_classes: int) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.embedder = nn.Embedding(self.n_classes, self.n_classes)
        input_channels = self.seq_length + self.n_classes
        output_channels = 1

        self.model = nn.Sequential(
            nn.Linear(input_channels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_channels),
        )

    def forward(self, seq: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.embedder(labels)
        seq = torch.cat((seq, emb), -1)
        return self.model(seq)


if __name__ == "__main__":

    N_CLASSES = 10
    SEQ_LENGTH = 200

    BATCH_SIZE = 256
    LATENT_DIM = 100

    gen = Generator(SEQ_LENGTH, LATENT_DIM, N_CLASSES)
    dis = Discriminator(SEQ_LENGTH, LATENT_DIM, N_CLASSES)

    noise_shape = (BATCH_SIZE, LATENT_DIM)
    noise = torch.randn(noise_shape)
    rand_labels = torch.randint(0, N_CLASSES, (BATCH_SIZE,))

    seq = gen(noise, rand_labels)
    probs = dis(seq, rand_labels)

    print("Noise :", noise.shape)
    print("  Seq :", seq.shape)
    print("Probs :", probs.shape)
