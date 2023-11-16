# import some libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import random
from scipy.stats import t, f
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats


import copy
import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# Generate the simulated data
# Initialize seed and parameters
# number of data points
n_S = 1000000
n_T = int(0.001 * n_S)
M = 14
# Model parameters
a = np.asarray([[1.1, -0.1, 0, 0.1, 0, 0.2, 0, 0.1, -0.1, 0, 0, 0.1, -0.1, 0.2, -0.2]])
b = (-1) * np.asarray([[0.5, 0.1, -0.1, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0, 0.2, -0.1, -0.2, 0]])
# independent variable in Sorce domain
mu_S = np.repeat(1, M)
cov_S = 0.2 * np.identity(M, dtype=float)
X0_S = np.random.multivariate_normal(mu_S, cov_S, n_S)
p_S = np.random.uniform(low=0.2, high=2.0, size=n_S)
# add column of ones for intercept
X_S = sm.add_constant(X0_S)
print(X_S.shape)
print(a.shape)
# dependent variable (i.e., demand ) in Sorce domain
d_S = a @ X_S.T+ (b @ X_S.T) * p_S + np.random.normal(0,0.1, n_S)


# revenue
r_S = d_S * p_S

# independent variable in Target domain 
#mu_T = np.repeat(0, M)
#cov_T = 0.05 * np.identity(M, dtype=float)
#X0_T = np.random.multivariate_normal(mu_T, cov_T, n_T)
df_T = 10
X0_T = stats.chi2.rvs(df_T, size=(n_T,M))
p_T = np.random.uniform(low=0.2, high=2.0, size=n_T)
# add column of ones for intercept
X_T = sm.add_constant(X0_T)
X_T[:,8:]=0
print(X_T.shape)
print(a.shape)
# dependent variable (i.e., demand ) in Target domain
d_T = a @ X_T.T+ (b @ X_T.T) * p_T + np.random.normal(0,0.1, n_T)
# revenue
r_T = d_T * p_T


def rescale(d_S):
    return (d_S-d_S.min())/(d_S.max()-d_S.min())




d_S =rescale(d_S)
d_T=rescale(d_T)
p_S = rescale(p_S)
p_T =rescale(p_T)
print(X_T.shape,p_T.shape)
print(d_S.max(),d_S.min())
#res = stats.linregress(np.concatenate((X_T,np.expand_dims(p_T,axis=1)),axis=1),d_T.T)

d_S=torch.tensor(d_S).transpose(0,1).float()
p_S=torch.tensor(p_S).unsqueeze(1).float()
x_S=torch.tensor(X_S).float()

d_T=torch.tensor(d_T).transpose(0,1).float()
p_T=torch.tensor(p_T).unsqueeze(1).float()
x_T=torch.tensor(X_T).float()


d_S = torch.cat([d_S,torch.zeros_like(d_S)],dim=-1)
d_T = torch.cat([d_T,torch.ones_like(d_T)],dim=-1)
d= torch.cat([d_S,d_T], dim=0)
p= torch.cat([p_S,p_T], dim=0)
x= torch.cat([x_S,x_T], dim=0)

print(d.shape ,p.shape, x.shape)



pdS_dataset = data.TensorDataset(torch.cat([p_S,x_S],dim=-1), d_S)
pdT_dataset = data.TensorDataset(torch.cat([p_T,x_T],dim=-1), d_T)



VALID_RATIO = 0.8
n_train_examples = int(d_S.shape[0] * VALID_RATIO)
n_valid_examples = (d_S.shape[0] - n_train_examples)//2
n_test_examples = (d_S.shape[0] - n_train_examples)//2
pdS_train, pdS_valid, pdS_test= data.random_split(pdS_dataset, 
                                           [n_train_examples, n_valid_examples,n_test_examples])
VALID_RATIO = 0.8
n_train_examples = int(d_T.shape[0] * VALID_RATIO)
n_valid_examples = (d_T.shape[0] - n_train_examples)//2
n_test_examples = (d_T.shape[0] - n_train_examples)//2
pdT_train, pdT_valid, pdT_test= data.random_split(pdT_dataset, 
                                           [n_train_examples, n_valid_examples,n_test_examples])
pd_train = pdT_train
pd_valid = pdT_valid
pd_test = pdT_test


lamb = 0.1
def train(model, iterator, optimizer, criterion, device):
    
    epoch_loss = 0
    
    model.train()
    
    epoch_rl = 0
    epoch_el = 0
    epoch_dl = 0
    epoch_gl = 0
    for (x, y) in iterator:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
                
        R, d_hat = model(x)
        
        r_loss = (R).mean()
        est_loss = criterion[0](d_hat, y[:,:1])
        #dom_loss = criterion[1](dom_cls, y[:,1:])
        #grad_loss = 1e6*grad_loss
        loss = est_loss#+dom_loss#+grad_loss
        loss.backward()
        
        optimizer.step()
        if r_loss >1000:
            print(r_loss)
        epoch_loss += loss.item()
        epoch_rl += r_loss.item()
        epoch_el += est_loss.item()
        #epoch_dl += dom_loss.item()
        #epoch_gl += grad_loss.item()
    print('train', epoch_rl/len(iterator), epoch_el/len(iterator), epoch_dl/len(iterator),epoch_gl/len(iterator))
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    
    model.eval()
    epoch_rl = 0
    epoch_el = 0
    epoch_dl = 0
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            R, d_hat = model(x)
        
            r_loss = (R).mean()
            est_loss = criterion[0](d_hat, y[:,:1])
            #dom_loss = criterion[1](dom_cls, y[:,1:])

            #loss = -lamb*r_loss+est_loss

            #epoch_loss += loss.item()
            epoch_rl += r_loss.item()
            epoch_el += est_loss.item()
            #epoch_dl += dom_loss.item()
    
    print('val', epoch_rl/len(iterator), epoch_el/len(iterator), epoch_dl/len(iterator))        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


BATCH_SIZE = 64
train_data, valid_data, test_data = pd_train, pd_valid, pd_test
train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data, 
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)





def init_weight(l):
  if isinstance(l,nn.Linear):
    nn.init.normal_(l.weight,mean=0,std=0.02)
    #l.bias.data.fill_(0)



class Hack(nn.Module):
  def __init__(self,):
    super().__init__()
    self.l1 =  nn.Linear(15,1,bias=False)
    self.l2=  nn.Linear(15,1,bias=False)
  def forward(self,x):
    p=x[:,0].unsqueeze(1)
    xx=x[:,1:]
    a = self.l2(xx)
    b = self.l1(xx)
    x = b+a*p
    
    p_opt= -b/(2*a)
    
    r = (p_opt*a+b)*p_opt
    return r, x


model=Hack()
#model=Model()
model.apply(init_weight)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {count_parameters(model):,} trainable parameters')



EPOCHS = 20

optimizer = optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.01)
#criterion = nn.L1Loss()

model = model.to(device)
criterion = (nn.MSELoss().to(device), nn.BCELoss().to(device))


best_valid_loss = float('inf')

model_name = "baseline.pt"
for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    
    train_loss = train(model, train_iterator, optimizer, criterion, device)
    valid_loss = evaluate(model, valid_iterator, criterion, device)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_name)
    
    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} ')
    print(f'\t Val. Loss: {valid_loss:.3f} ')

model.load_state_dict(torch.load(model_name))

test_loss= evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f}')
