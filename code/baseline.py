# 1. Import libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import random
import time

# 2. Seed initialization
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# 3. Data generation
# Parameters
n_S = 1000000  # Number of data points in Source domain
n_T = int(0.001 * n_S)  # Number of data points in Target domain
M = 14  # Number of features
a = np.asarray([[1.1, -0.1, 0, 0.1, 0, 0.2, 0, 0.1, -0.1, 0, 0, 0.1, -0.1, 0.2, -0.2]])
b = (-1) * np.asarray([[0.5, 0.1, -0.1, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0, 0.2, -0.1, -0.2, 0]])

# Source domain data
mu_S = np.repeat(1, M)
cov_S = 0.2 * np.identity(M)
X0_S = np.random.multivariate_normal(mu_S, cov_S, n_S)
p_S = np.random.uniform(low=0.2, high=2.0, size=n_S)
X_S = sm.add_constant(X0_S)  # Add intercept
d_S = a @ X_S.T + (b @ X_S.T) * p_S + np.random.normal(0, 0.1, n_S)  # Demand
r_S = d_S * p_S  # Revenue

# Target domain data
df_T = 10
X0_T = stats.chi2.rvs(df_T, size=(n_T, M))
p_T = np.random.uniform(low=0.2, high=2.0, size=n_T)
X_T = sm.add_constant(X0_T)  # Add intercept
X_T[:, 8:] = 0  # Zero out some features
d_T = a @ X_T.T + (b @ X_T.T) * p_T + np.random.normal(0, 0.1, n_T)  # Demand
r_T = d_T * p_T  # Revenue

# 4. Data preprocessing
# Function to rescale data to [0, 1] range
def rescale(data):
    return (data - data.min()) / (data.max() - data.min())

# Rescale the data
d_S = rescale(d_S)
d_T = rescale(d_T)
p_S = rescale(p_S)
p_T = rescale(p_T)

# Convert to PyTorch tensors
d_S = torch.tensor(d_S).transpose(0, 1).float()
p_S = torch.tensor(p_S).unsqueeze(1).float()
x_S = torch.tensor(X_S).float()
d_T = torch.tensor(d_T).transpose(0, 1).float()
p_T = torch.tensor(p_T).unsqueeze(1).float()
x_T = torch.tensor(X_T).float()

# Create combined datasets
d = torch.cat([d_S, d_T], dim=0)
p = torch.cat([p_S, p_T], dim=0)
x = torch.cat([x_S, x_T], dim=0)

# TensorDataset and DataLoader setup
pdS_dataset = data.TensorDataset(torch.cat([p_S, x_S], dim=-1), d_S)
pdT_dataset = data.TensorDataset(torch.cat([p_T, x_T], dim=-1), d_T)

# Split datasets into training, validation, and test sets
VALID_RATIO = 0.8
n_train_examples = int(d_S.shape[0] * VALID_RATIO)
n_valid_examples = d_S.shape[0] - n_train_examples
n_test_examples = d_T.shape[0] - int(d_T.shape[0] * VALID_RATIO)
pdS_train, pdS_valid, pdS_test = data.random_split(pdS_dataset, [n_train_examples, n_valid_examples, n_valid_examples])
pdT_train, pdT_valid, pdT_test = data.random_split(pdT_dataset, [n_train_examples, n_valid_examples, n_valid_examples])

# Combine training data from both domains
pd_train = data.ConcatDataset([pdS_train, pdT_train])
pd_valid = pdT_valid
pd_test = pdT_test

# 5. Model definition
class Hack(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(15, 1, bias=False)
        self.l2 = nn.Linear(15, 1, bias=False)

    def forward(self, x):
        p = x[:, 0].unsqueeze(1)
        xx = x[:, 1:]
        a = self.l2(xx)
        b = self.l1(xx)
        x = b + a * p
        p_opt = -b / (2 * a)
        r = (p_opt * a + b) * p_opt
        return r, x

# Initialize model
model = Hack()
model.apply(init_weight)  # Apply custom weight initialization

# 6. Training and evaluation functions
# Function to train the model for one epoch
def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    model.train()  # Set the model to training mode
    
    for (x, y) in iterator:        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        R, d_hat = model(x)  # Forward pass
        
        # Calculate losses
        est_loss = criterion[0](d_hat, y[:, :1])
        loss = est_loss
        loss.backward()  # Backward pass
        
        optimizer.step()  # Update weights
        epoch_loss += loss.item()  # Aggregate the loss
    
    # Average the loss over all batches
    return epoch_loss / len(iterator)

# Function to evaluate the model on a dataset
def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():  # No gradients needed
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            R, d_hat = model(x)  # Forward pass
            
            # Calculate losses
            est_loss = criterion[0](d_hat, y[:, :1])
            loss = est_loss
            epoch_loss += loss.item()  # Aggregate the loss
    
    # Average the loss over all batches
    return epoch_loss / len(iterator)

# Helper function to track the time of an epoch
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# 7. Main training loop
# Setup DataLoader for each set
train_iterator = data.DataLoader(pd_train, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = data.DataLoader(pd_valid, batch_size=BATCH_SIZE)
test_iterator = data.DataLoader(pd_test, batch_size=BATCH_SIZE)

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = Hack()
model.apply(init_weight)  # Apply custom weight initialization
model = model.to(device)

# Training hyperparameters
EPOCHS = 20
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
criterion = (nn.MSELoss().to(device), nn.BCELoss().to(device))  # Tuple of loss functions

# Train and evaluate the model
best_valid_loss = float('inf')
model_name = "baseline.pt"

for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    
    # Training step
    train_loss = train(model, train_iterator, optimizer, criterion, device)
    
    # Evaluation step on the validation set
    valid_loss = evaluate(model, valid_iterator, criterion, device)
    
    # Save the model if the validation loss is the best we've seen so far
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_name)
    
    end_time = time.monotonic()

    # Calculate elapsed time for the epoch
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
    # Print epoch summary
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tVal. Loss: {valid_loss:.3f}')

# Load the best model
model.load_state_dict(torch.load(model_name))

# Evaluate on the test set
test_loss = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f}')
