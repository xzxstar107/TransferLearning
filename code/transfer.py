# 1. Import necessary libraries for data handling, statistical modeling, and machine learning
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from scipy.stats import t, f, stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import copy
import time

# 2. Seed initialization for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 3. Data generation and preprocessing
# Simulation parameters
n_S = 1000000  # Number of data points in Source domain
n_T = int(0.001 * n_S)  # Number of data points in Target domain
M = 14  # Number of features
# Model parameters for generating synthetic data
a = np.asarray([[1.1, -0.1, 0, 0.1, 0, 0.2, 0, 0.1, -0.1, 0, 0, 0.1, -0.1, 0.2, -0.2]])
b = (-1) * np.asarray([[0.5, 0.1, -0.1, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0, 0.2, -0.1, -0.2, 0]])
# Generate synthetic data for Source and Target domains
# Source domain
mu_S = np.repeat(1, M)
cov_S = 0.2 * np.identity(M, dtype=float)
X0_S = np.random.multivariate_normal(mu_S, cov_S, n_S)
p_S = np.random.uniform(low=0.2, high=2.0, size=n_S)
# Add column of ones for intercept
X_S = sm.add_constant(X0_S)
d_S = a @ X_S.T + (b @ X_S.T) * p_S + np.random.normal(0, 0.1, n_S)
r_S = d_S * p_S  # Revenue in Source domain

# Target domain
df_T = 10
X0_T = stats.chi2.rvs(df_T, size=(n_T, M))
p_T = np.random.uniform(low=0.2, high=2.0, size=n_T)
# Add column of ones for intercept
X_T = sm.add_constant(X0_T)
X_T[:, 8:] = 0  # Setting some features to 0
d_T = a @ X_T.T + (b @ X_T.T) * p_T + np.random.normal(0, 0.1, n_T)
r_T = d_T * p_T  # Revenue in Target domain

# Function to rescale data to the [0, 1] range
def rescale(data):
    return (data - data.min()) / (data.max() - data.min())

# Rescale the data
d_S = rescale(d_S)
d_T = rescale(d_T)
p_S = rescale(p_S)
p_T = rescale(p_T)

# Convert data to PyTorch tensors and create datasets
# Convert the rescaled Source domain data to tensors
d_S_tensor = torch.tensor(d_S, dtype=torch.float32).unsqueeze(1)  # Demand in Source domain
p_S_tensor = torch.tensor(p_S, dtype=torch.float32).unsqueeze(1)  # Prices in Source domain
X_S_tensor = torch.tensor(X_S, dtype=torch.float32)  # Features in Source domain

# Convert the rescaled Target domain data to tensors
d_T_tensor = torch.tensor(d_T, dtype=torch.float32).unsqueeze(1)  # Demand in Target domain
p_T_tensor = torch.tensor(p_T, dtype=torch.float32).unsqueeze(1)  # Prices in Target domain
X_T_tensor = torch.tensor(X_T, dtype=torch.float32)  # Features in Target domain

# Combine the features and prices tensors for both domains
features_and_prices_S = torch.cat([p_S_tensor, X_S_tensor], dim=1)
features_and_prices_T = torch.cat([p_T_tensor, X_T_tensor], dim=1)

# Create labels by concatenating the demand with zeros for Source and ones for Target domain
# This is often done for domain adaptation tasks to distinguish between domains
labels_S = torch.cat([d_S_tensor, torch.zeros_like(d_S_tensor)], dim=1)
labels_T = torch.cat([d_T_tensor, torch.ones_like(d_T_tensor)], dim=1)

# Create TensorDatasets for both domains
pdS_dataset = data.TensorDataset(features_and_prices_S, labels_S)
pdT_dataset = data.TensorDataset(features_and_prices_T, labels_T)

# Split the Source domain dataset into training, validation, and test sets
VALID_RATIO = 0.8
n_train_examples = int(len(pdS_dataset) * VALID_RATIO)
n_valid_examples = len(pdS_dataset) - n_train_examples
pdS_train, pdS_valid = data.random_split(pdS_dataset, [n_train_examples, n_valid_examples])

# Do the same for the Target domain dataset
n_train_examples = int(len(pdT_dataset) * VALID_RATIO)
n_valid_examples = len(pdT_dataset) - n_train_examples
pdT_train, pdT_valid = data.random_split(pdT_dataset, [n_train_examples, n_valid_examples])

# Optionally, create a combined dataset for training on both domains
pd_train = data.ConcatDataset([pdS_train, pdT_train])

# Function to flatten a sequence of tensors into a single tensor.
def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

# Functions for Clenshaw-Curtis Quadrature integration.
# These functions are used to perform numerical integration which is part of the
# neural network's training loop to compute the integral of the learned function.

# Compute Clenshaw-Curtis weights given a number of steps.
def compute_cc_weights(nb_steps):
    # Create an array of cosines used in the Clenshaw-Curtis method.
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * np.pi / nb_steps)
    # Adjustments to the first and last weights
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    # Compute the final weights for the quadrature
    lam = lam * 2 / nb_steps
    # Weights vector for the Clenshaw-Curtis method
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[1::2] = 0  # Set every other weight to 0
    W = 2 / (1 - W ** 2)
    W[0] = 1  # First weight is 1
    W[1::2] = 0  # Reset every other weight to 0 again
    # Calculate the final Clenshaw-Curtis weights
    cc_weights = torch.tensor(lam.T @ W).float()
    # Steps in the Clenshaw-Curtis method as cosines
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * np.pi / nb_steps)).float()

    return cc_weights, steps

# Integrate using Clenshaw-Curtis Quadrature Method.
def integrate(x0, nb_steps, step_sizes, integrand, h, compute_grad=False, x_tot=None):
    # Calculate the weights and steps for the integration
    cc_weights, steps = compute_cc_weights(nb_steps)
    
    # Set device to CPU or GPU
    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)
    
    # Calculate the endpoints of integration
    xT = x0 + nb_steps * step_sizes
    
    # If not computing the gradient, perform the integration
    if not compute_grad:
        # Expand dimensions to prepare for computation across all steps
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        
        # Compute all the steps
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        
        # Flatten to pass through the integrand
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        
        # Evaluate the integrand function at all the steps
        dzs = integrand(X_steps, h_steps)
        
        # Reshape to have the outputs aligned with the cc weights
        dzs = dzs.view(xT_t.shape[0], nb_steps + 1, -1)
        
        # Apply the Clenshaw-Curtis weights
        dzs = dzs * cc_weights.unsqueeze(0).expand(dzs.shape)
        
        # Sum the results of the integration steps and scale by the interval
        z_est = dzs.sum(1) * (xT - x0) / 2
        return z_est
    else:
        # If computing the gradient, first set up the tensors with the proper shapes
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        
        # Prepare the tensor that accumulates gradients
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) * cc_weights.unsqueeze(0).expand(x_tot.shape[0], -1, x_tot.shape[1])
        
        # Compute the integrand and its gradients
        g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps + 1)
        return g_param, g_h

# Compute the integrand for the Clenshaw-Curtis Quadrature Method.
def computeIntegrand(x, h, integrand, x_tot, nb_steps):
    # Enable gradient calculation
    h.requires_grad_(True)
    with torch.enable_grad():
        # Compute the function value at given points
        f = integrand.forward(x, h)
        # Calculate gradients with respect to the integrand parameters and h
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    # Return gradients with respect to the parameters and h
    return g_param, g_h.view(int(x.shape[0] / nb_steps), nb_steps, -1).sum(1)


# 5. Neural network models
# Neural network for approximating the integrand in quadrature
class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        """
        Initializes the neural network used for approximating the integrand in quadrature.
        Args:
            in_d (int): The input dimension of the data.
            hidden_layers (list of int): A list defining the number of neurons in each hidden layer.
        """
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]  # Add output layer with 1 neuron
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # Remove the last ReLU for the output layer
        self.net.append(nn.ELU())  # Add ELU activation for the output
        self.net = nn.Sequential(*self.net)  # Create a sequential container

    def forward(self, x, h):
        """
        Forward pass through the integrand neural network.
        Args:
            x (Tensor): The input tensor containing the data points for integration.
            h (Tensor): The additional tensor containing parameters or conditions for the integration.
        Returns:
            Tensor: The result of the neural network approximation of the integrand.
        """
        return self.net(torch.cat((x, h), 1)) + 1.

# Neural network for monotonic integration
class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        """
        Initializes the neural network used for monotonic integration.
        Args:
            in_d (int): The input dimension of the data.
            hidden_layers (list of int): A list defining the number of neurons in each hidden layer.
            nb_steps (int): The number of steps used in the numerical integration.
            dev (str): The device to use for computation (cpu or cuda).
        """
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]  # Add output layer with 2 neurons for scaling and offset
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # Remove the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps

    def forward(self, x, h):
        """
        Forward pass through the monotonic neural network.
        Args:
            x (Tensor): The input tensor for which integration is to be performed.
            h (Tensor): Additional tensor containing parameters or conditions for the integration.
        Returns:
            Tensor: The result of the monotonic neural network integration.
        """
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        # Apply the Parallel Neural Integral, which is a custom autograd function
        return scaling * ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset

# 6. Training and evaluation functions
# Function to train the model for one epoch
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    model.train()
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Function to evaluate the model on a dataset
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)
            predictions = model(x)
            loss = criterion(predictions, y)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Function to calculate elapsed time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time % 60)
    return elapsed_mins, elapsed_secs

# 7. Main Training Loop
# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 20

# Creating DataLoaders for training, validation, and testing
train_data, valid_data, test_data = pd_train, pd_valid, pd_test
train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = data.DataLoader(test_data, batch_size=BATCH_SIZE)

# Setting up the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Model initialization
model = Model()
model.apply(init_weight)  # Initialize model weights

# Setting up the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Loss function(s)
criterion = (nn.MSELoss().to(device), nn.BCELoss().to(device))

# Training loop
best_valid_loss = float('inf')
model_name = "model_umn_new_0.1.pt"

for epoch in range(EPOCHS):
    start_time = time.monotonic()
    
    # Train and evaluate
    train_loss = train(model, train_iterator, optimizer, criterion, device)
    valid_loss = evaluate(model, valid_iterator, criterion, device)
    
    # Check if the validation loss improved
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_name)  # Save the model
    
    end_time = time.monotonic()

    # Calculate elapsed time for the epoch
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
    # Print epoch summary
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} ')
    print(f'\tVal. Loss: {valid_loss:.3f} ')

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load(model_name))
test_loss = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f}')
