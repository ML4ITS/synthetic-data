import torch
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

from synthetic_data.mlops.tmp.data import HarmonicDataset


class Generator(nn.Module):
    def __init__(self, seq_length: int = 100, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        scalar = seq_length // 100
        self.l1 = nn.Sequential(
            nn.Linear(in_features=seq_length, out_features=100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l2 = nn.Sequential(
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False
                ),
                n_power_iterations=10,
            ),
            nn.Upsample(200),
        )
        self.l3 = nn.Sequential(
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
            nn.Upsample(400),
        )
        self.l4 = nn.Sequential(
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
            nn.Upsample(800),
        )
        self.l5 = nn.Sequential(
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=False
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.l6 = nn.Linear(in_features=800, out_features=scalar * 100)

    def forward(self, x: Tensor) -> Tensor:
        if self.debug:
            return self._forward_with_debug(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = out.unsqueeze(1)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = out.squeeze(1)
        out = self.l6(out)
        return out

    def _forward_with_debug(self, x: Tensor) -> Tensor:
        print("IN =", x.shape)
        out = self.l1(x)
        print("L1 =", out.shape)
        out = out.unsqueeze(1)
        print("US =", out.shape)
        out = self.l2(out)
        print("L2 =", out.shape)
        out = self.l3(out)
        print("L3 =", out.shape)
        out = self.l4(out)
        print("L4 =", out.shape)
        out = self.l5(out)
        print("L5 =", out.shape)
        out = out.squeeze(1)
        print("SQ =", out.shape)
        out = self.l6(out)
        print("L6 =", out.shape)
        return out


class Critic(nn.Module):
    def __init__(self, seq_length: int = 100, debug: bool = False) -> None:
        super().__init__()
        self.debug = debug
        scalar = seq_length // 100
        self.l1 = nn.Sequential(
            spectral_norm(
                module=nn.Conv1d(
                    in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False
                ),
                n_power_iterations=10,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.l2 = nn.Sequential(
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
        )

        self.l3 = nn.Sequential(
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
        )
        self.l4 = nn.Sequential(
            nn.Linear(in_features=scalar * 800, out_features=50),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=50, out_features=15),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=15, out_features=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.debug:
            return self._forward_with_debug(x)
        return self._forward(x)

    def _forward(self, x: Tensor) -> Tensor:
        out = x.unsqueeze(1)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        return out

    def _forward_with_debug(self, x: Tensor) -> Tensor:
        print("IN =", x.shape)
        out = x.unsqueeze(1)
        print("US =", out.shape)
        out = self.l1(out)
        print("L1 =", out.shape)
        out = self.l2(out)
        print("L2 =", out.shape)
        out = self.l3(out)
        print("L3 =", out.shape)
        out = self.l4(out)
        print("L4 =", out.shape)
        return out


@torch.no_grad()
def test_models():

    SPLIT_SIZE = 100
    SPLIT_RATIO = 1
    DATASET_NAME = "AMP20"

    config = {"batch_size": 10}

    dataset = HarmonicDataset(DATASET_NAME, SPLIT_SIZE, SPLIT_RATIO)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    signal_length = dataset.dataset.shape[1]

    gen = Generator(signal_length, debug=True)
    ctc = Critic(signal_length, debug=True)

    noise_shape = (config["batch_size"], signal_length)
    noise = torch.randn(noise_shape)

    # Test forward on noise
    _ = gen(noise)

    # Test forward on data
    for d in dataloader:
        _ = ctc(gen(d.float()))


if __name__ == "__main__":
    raise SystemExit(test_models())
