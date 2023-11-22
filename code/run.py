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

# 3. Loading pre-trained models
model_mon = Model().eval()
model_mon.load_state_dict(torch.load("model_umn_new_0.1.pt"))
model_mlp_grad = Model_mlp().eval()
model_mlp_grad.load_state_dict(torch.load("model_mlp_grad.pt"))
model_mlp = Model_mlp().eval()
model_mlp.load_state_dict(torch.load("model_mlp.pt"))

# 4. Data generation for testing
# Model parameters
a = np.asarray([[1.1, -0.1, 0, 0.1, 0, 0.2, 0, 0.1, -0.1, 0, 0, 0.1, -0.1, 0.2, -0.2]])
b = (-1) * np.asarray([[0.5, 0.1, -0.1, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0, 0.2, -0.1, -0.2, 0]])

# Generate synthetic data for a single instance
mu_S = np.repeat(1, M)  # Mean of features
cov_S = 0.2 * np.identity(M)  # Covariance matrix of features
X0_T = np.random.multivariate_normal(mu_S, cov_S, 1)  # Generate features

# Add intercept to features
X_T = sm.add_constant(X0_T, has_constant="add")
X_T[:, 8:] = 0  # Zero out some features for the target domain

# Assuming some pricing strategy for generating demand
p_T = np.random.uniform(low=0.2, high=2.0)  # Randomly generated price
d_T = a @ X_T.T + (b @ X_T.T) * p_T + np.random.normal(0, 0.1)  # Demand

# 5. Data preprocessing function
# Assuming d_min and d_max are known or computed previously across the dataset
d_min = np.min(d_T)  # Minimum demand value
d_max = np.max(d_T)  # Maximum demand value

def rescale(data, d_min, d_max):
    """ Rescale data to [0, 1] range based on known min and max values. """
    return (data - d_min) / (d_max - d_min)

# Apply the rescaling to the generated demand data
d_T_rescaled = rescale(d_T, d_min, d_max)

# 6. Testing and plotting results
# Function for demand calculation based on given parameters
def f(X_T, p_T, a, b, d_min, d_max):
    beta = a @ X_T.T
    alpha = (b @ X_T.T)
    d_T = beta + alpha * p_T
    p_opt = -beta / (2 * alpha)
    print("opt_p", p_opt)
    return rescale(d_T, d_min, d_max)

# Parameters for testing
a = np.asarray([[1.1, -0.1, 0, 0.1, 0, 0.2, 0, 0.1, -0.1, 0, 0, 0.1, -0.1, 0.2, -0.2]])
b = (-1) * np.asarray([[0.5, 0.1, -0.1, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0, 0.2, -0.1, -0.2, 0]])
d_min = -2.3989378562534847 
d_max = 2.025556543311613

# Generate test data
mu_S = np.repeat(1, M)
cov_S = 0.2 * np.identity(M, dtype=float)
X0_T = np.random.multivariate_normal(mu_S, cov_S, 1)
X_T = sm.add_constant(X0_T, has_constant="add")
X_T[:, 8:] = 0

# Prepare data for model input
x = torch.arange(0, 1, .1).unsqueeze(1).to(device)
y_true = f(X_T, 1.8 * x.numpy() + 0.2, a, b, d_min, d_max)
X_T1 = torch.tensor(X_T).expand(x.shape[0], -1).float().to(device)

# Get predictions from models
y_mon = torch.sigmoid(model_mon.d_f(-x, X_T1)[:, 0]).detach().cpu().numpy()
y_mlp = torch.sigmoid(model_mlp.d_f(x, X_T1)[:, 0]).detach().cpu().numpy()
y_mlp_grad = torch.sigmoid(model_mlp_grad.d_f(x, X_T1)[:, 0]).detach().cpu().numpy()

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x.numpy(), y_mon, label="Monotonic MLP")
plt.plot(x.numpy(), y_mlp, label="MLP")
plt.plot(x.numpy(), y_mlp_grad, label="MLP+GradReg")
plt.plot(x.numpy(), y_true, label="Ground Truth")
plt.xlabel("Price")
plt.ylabel("Demand")
plt.title("Price vs. Demand")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x.numpy(), x.numpy() * y_mon, label="Monotonic MLP")
plt.plot(x.numpy(), x.numpy() * y_mlp, label="MLP")
plt.plot(x.numpy(), x.numpy() * y_mlp_grad, label="MLP+GradReg")
plt.plot(x.numpy(), (1.8 * x.numpy() + 0.2) * y_true, label="Ground Truth")
plt.xlabel("Price")
plt.ylabel("Revenue")
plt.title("Price vs. Revenue")
plt.legend()

plt.tight_layout()
plt.savefig("Demand_and_Revenue.png")
plt.show()
