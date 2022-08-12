from typing import Tuple

import torch
from torch.autograd import grad as torch_grad
from torch.nn import Module


def calc_gradient_penalty(
    modelD: Module,
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    gp_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = real_data.size(0)
    """Calculated the gradient penalty loss formulation,
    plus a gradient norm penalty to achieve Lipschitz continuity
    .Penalize the norm of gradient of the critic with respect to its input.

    Args:
        modelD (Module): the critic / discriminator
        real_data (torch.Tensor): the real data
        generated_data (torch.Tensor): the generated data
        gp_weight (float): the weight of the gradient penalty
        device (str): the device to use

    Returns:
        gradient loss, gradient norm (torch.Tensor, torch.Tensor): The gradient penalty and the gradient norm
    """

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = interpolated.clone().detach().requires_grad_(True)

    # Pass interpolated data through Critic
    prob_interpolated = modelD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Gradients have shape (batch_size, num_channels, series length),
    # here we flatten to take the norm per example for every batch
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1).mean().data.item()

    # Derivatives of the gradient close to 0 can cause problems because of the
    # square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    gp_loss = gp_weight * ((gradients_norm - 1) ** 2).mean()

    return gp_loss, grad_norm
