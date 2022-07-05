from typing import List, Union

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def move_to_device(
    data: Union[torch.Tensor, List[torch.Tensor]], device: torch.device
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if data is None:
        raise ValueError("Data cannot be None")

    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [d.to(device) for d in data]
    else:
        raise TypeError(
            f"Data must be provided as torch.Tensor or list of torch.Tensor"
        )
