# 1. Import statements (organized and deduplicated)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from scipy import stats
import matplotlib.pyplot as plt
import random
import math
import time
from torch.autograd import grad

# 2. Seed initialization for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = "cpu"  # Device configuration

# 3. Utility functions
# 3. Utility functions
def _flatten(sequence):
    """ Flatten a sequence of tensors. """
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

def compute_cc_weights(nb_steps):
    """
    Compute Clenshaw-Curtis weights and steps for numerical integration.

    Args:
    - nb_steps (int): The number of steps for integration.

    Returns:
    - (Tensor, Tensor): A tuple containing the weights and steps tensors.
    """
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps

def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None):
    """
    Integrate using the Clenshaw-Curtis Quadrature Method.

    Args:
    - x0 (Tensor): The starting point for integration.
    - nb_steps (int): The number of steps for integration.
    - step_sizes (Tensor): The sizes of each step for integration.
    - integrand (Callable): The function representing the integrand.
    - h (Tensor): Additional parameters for the integrand.
    - compute_grad (bool): Whether to compute gradients during integration.
    - x_tot (Tensor, optional): If provided, used to compute gradients.

    Returns:
    - Tensor: The result of the integration, or a tuple containing gradients if compute_grad is True.
    """
    cc_weights, steps = compute_cc_weights(nb_steps)
    cc_weights, steps = cc_weights.to(x0.device), steps.to(x0.device)

    xT = x0 + nb_steps * step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        dzs = integrand(X_steps, h_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps + 1, -1)
        dzs = dzs * cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est * (xT - x0) / 2
    else:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) * cc_weights.unsqueeze(0).expand(x_tot.shape[0], -1, x_tot.shape[1])
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])

        g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps + 1)
        return g_param, g_h

def computeIntegrand(x, h, integrand, x_tot, nb_steps):
    """
    Compute the integrand and its gradients.

    Args:
    - x (Tensor): Input tensor for the integrand.
    - h (Tensor): Additional parameters for the integrand.
    - integrand (Module): The neural network module representing the integrand.
    - x_tot (Tensor): Input tensor for gradient computation.
    - nb_steps (int): The number of steps used in the integration.

    Returns:
    - Tuple[Tensor, Tensor]: A tuple of tensors representing the gradients.
    """
    h.requires_grad_(True)
    with torch.enable_grad():
        f = integrand.forward(x, h)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h.view(int(x.shape[0] / nb_steps), nb_steps, -1).sum(1)



