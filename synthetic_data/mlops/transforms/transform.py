from typing import Tuple

import torch

SequenceTuple = Tuple[torch.Tensor, torch.Tensor]


class RandomRoll(torch.nn.Module):
    """Randomly rolls/shifts the input sequence.

    For example: setting p = -1, will always shift the sequence.
    For example: setting p = 0.5, will randomly shift the sequence with 50% probability.

    Args:
        p: (float) the probability of we should roll or not.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    @torch.no_grad()
    def forward(self, sample: SequenceTuple) -> SequenceTuple:
        """Depending on p, either rolls the sequence,
        retrives from input sample

        Args:
            sample (torch.Tensor, torch.Tensor): sequence, label

        Returns:
            (torch.Tensor, torch.Tensor): the sequence, the label
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
