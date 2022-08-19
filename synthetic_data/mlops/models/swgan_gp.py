"""

NOTE: Experimental code.

Spectral Normalized- Wasserstein GAN implementation (assumed to be used with Gradient Penalty)

Not sure if spectral normalization should be used in conjunction with Gradient Penalty.

inspiration from https://github.com/mirkosavasta/GANetano
"""
from torch import Tensor, nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, seq_length: int, z_dim: int) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.z_dim = z_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=self.z_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.stem_layer = nn.Sequential(
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False
                ),
                n_power_iterations=10,
            ),
            nn.Upsample(256),
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(512),
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(1024),
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.output_layer = nn.Linear(in_features=1024, out_features=self.seq_length)

    def forward(self, x: Tensor) -> Tensor:
        out = self.input_layer(x)
        out = out.unsqueeze(1)
        out = self.stem_layer(out)
        out = out.squeeze(1)
        out = self.output_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, seq_length: int) -> None:
        super().__init__()
        self.seq_length = seq_length

        self.model = nn.Sequential(
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2),
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2),
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=32 * (self.seq_length // 4), out_features=50),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=50, out_features=15),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=15, out_features=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x.unsqueeze(1)
        out = self.model(out)
        return out


if __name__ == "__main__":

    from synthetic_data.mlops.tools.summary import summarize_gan

    summarize_gan(Generator, Discriminator)
