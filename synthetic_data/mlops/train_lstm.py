import numpy as np
import torch

from synthetic_data.common.torchutils import get_device
from synthetic_data.mlops.models.lstm import LSTM

_DEPRECATED_MSG = "Removed outdated implementation, see git history details."


class Trainer:
    def __init__(self) -> None:
        raise NotImplementedError(_DEPRECATED_MSG)

    def train(self) -> None:
        raise NotImplementedError(_DEPRECATED_MSG)

    def test() -> None:
        raise NotImplementedError(_DEPRECATED_MSG)

    def _train_epoch(self) -> None:
        raise NotImplementedError(_DEPRECATED_MSG)

    def _test(self) -> None:
        raise NotImplementedError(_DEPRECATED_MSG)

    def vizualize_and_save_prediction(self) -> None:
        raise NotImplementedError(_DEPRECATED_MSG)

    def _save_statedict(self) -> None:
        raise NotImplementedError(_DEPRECATED_MSG)


def run_training_session(config):
    torch.manual_seed(1337)
    np.random.seed(1337)
    device = get_device()

    dataset = None

    model = LSTM()
    model.double()
    model.to(device)

    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    raise NotImplementedError(_DEPRECATED_MSG)

    # TODO: create working trainer class
    # TODO: make model train on the new MultiHarmonicDataset
    # TODO: change the training loop to use the new MultiHarmonicDataset
    # TODO: change the optimizer to Adam (more common, easier to understand etc)
