from typing import Tuple

import numpy as np
import torch


class RandomRoll(torch.nn.Module):
    """Randomly rolls the input sequence along the time dimension.

    Args:
        p: The probability of we should roll or not.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(
        self, sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Depending on p, either rolls the sequence,
        retrives from input sample

        Args:
            sample (Tuple[torch.Tensor, torch.Tensor]): sequence, label

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: sequence, label
        """
        sequence = sample[0]
        target = sample[1]

        if torch.rand(1).item() < self.p:
            return sequence, target

        _min = 0
        _max = len(sequence)
        _shifts = int(torch.randint(_min, _max, (1,)).item())

        sequence = torch.roll(sequence, _shifts, dims=0)

        return sequence, target
