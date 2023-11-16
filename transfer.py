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




#print(d_S.min(), d_S.max())
d_S =rescale(d_S)
#raise ValueError
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
pd_train = data.ConcatDataset([pdS_train,pdT_train])
pd_valid = pdT_valid
pd_test = pdT_test


import torch
import numpy as np
import math


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
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
    #Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    xT = x0 + nb_steps*step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        h_steps = h.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        h_steps = h_steps.contiguous().view(-1, h.shape[1])
        dzs = integrand(X_steps, h_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps+1, -1)
        dzs = dzs*cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est*(xT - x0)/2
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

        g_param, g_h = computeIntegrand(X_steps, h_steps, integrand, x_tot_steps, nb_steps+1)
        return g_param, g_h


def computeIntegrand(x, h, integrand, x_tot, nb_steps):
    h.requires_grad_(True)
    with torch.enable_grad():
        f = integrand.forward(x, h)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
        g_h = _flatten(torch.autograd.grad(f, h, x_tot))

    return g_param, g_h.view(int(x.shape[0]/nb_steps), nb_steps, -1).sum(1)


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, h, nb_steps=20):
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, h, False)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone(), h)
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, h = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad, h_grad = integrate(x0, nb_steps, x/nb_steps, integrand, h, True, grad_output)
        x_grad = integrand(x, h)
        x0_grad = integrand(x0, h)
        # Leibniz formula
        return -x0_grad*grad_output, x_grad*grad_output, None, integrand_grad, h_grad.view(h.shape), None

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

    def forward(self, x, h):
        return self.net(torch.cat((x, h), 1)) + 1.

class MonotonicNN(nn.Module):
    def __init__(self, in_d, hidden_layers, nb_steps=50, dev="cpu"):
        super(MonotonicNN, self).__init__()
        self.integrand = IntegrandNN(in_d, hidden_layers)
        self.net = []
        hs = [in_d-1] + hidden_layers + [2]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        # It will output the scaling and offset factors.
        self.net = nn.Sequential(*self.net)
        self.device = dev
        self.nb_steps = nb_steps

    '''
    The forward procedure takes as input x which is the variable for which the integration must be made, h is just other conditionning variables.
    '''
    def forward(self, x, h):
        x0 = torch.zeros(x.shape).to(self.device)
        out = self.net(h)
        offset = out[:, [0]]
        scaling = torch.exp(out[:, [1]])
        return scaling*ParallelNeuralIntegral.apply(x0, x, self.integrand, _flatten(self.integrand.parameters()), h, self.nb_steps) + offset





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
                
        R, d_hat, dom_cls, grad_loss = model(x)
        
        r_loss = (R).mean()
        est_loss = criterion[0](d_hat, y[:,:1])
        dom_loss = criterion[1](dom_cls, y[:,1:])
        #grad_loss = 1e6*grad_loss
        loss = lamb*r_loss+est_loss+dom_loss#+grad_loss
        loss.backward()
        
        optimizer.step()
        if r_loss >1000:
            print(r_loss)
        epoch_loss += loss.item()
        epoch_rl += r_loss.item()
        epoch_el += est_loss.item()
        epoch_dl += dom_loss.item()
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

            R, d_hat, dom_cls,_ = model(x)
        
            r_loss = (R).mean()
            est_loss = criterion[0](d_hat, y[:,:1])
            dom_loss = criterion[1](dom_cls, y[:,1:])

            #loss = -lamb*r_loss+est_loss

            #epoch_loss += loss.item()
            epoch_rl += r_loss.item()
            epoch_el += est_loss.item()
            epoch_dl += dom_loss.item()
    
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

def FeedForward(dim,out_dim, n=1, expansion_factor = 4, dropout = 0.5, dense = nn.Linear,
                act=nn.ReLU(inplace=True)):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        act,
        nn.Dropout(dropout),
        *[nn.Sequential(dense(dim * expansion_factor, dim * expansion_factor),
        act,
        nn.Dropout(dropout)) for _ in range(n)],
        dense(dim * expansion_factor, out_dim),
    )
from torch.autograd import grad
class Demand(nn.Module):
  def __init__(self,fn = FeedForward):
    super().__init__()
    self.linear =  nn.Linear(1,15)
    #self.mlp1=fn(1,15,n=0, expansion_factor = 16)
    
    self.mlp2=fn(30,1, n = 1)
  def forward(self,p,x):    
    
    x=torch.cat([self.linear(p),x],dim=-1)
    x = self.mlp2(x)
    return x



class GradRev(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx,input):
        return input

    @staticmethod
    def backward(ctx,grad_out):
        return -grad_out    


class Model(nn.Module):
    def __init__(self,fn = FeedForward):
        super().__init__()
        self.mlp1=fn(15,15, n = 2)
        self.d_f = MonotonicNN(16, [60, 60], nb_steps=100, dev=device)
        self.p_f=fn(15,1, n = 1)
        self.critic = fn(15,1,n=1)

    def forward(self, feat):
        p=feat[:,0].unsqueeze(1)
        x=feat[:,1:]
        z=self.mlp1(x)
        #p.requires_grad = True
        d_hat= torch.sigmoid(self.d_f(-p,z)[:, :1])
        p_opt = torch.sigmoid(self.p_f(z))

                #with torch.no_grad():
        d_opt = torch.sigmoid(self.d_f(-p_opt,z)[:, :1])
        r_opt = p_opt*d_opt


        #cp = p.clone()
        #cp_opt = p_opt.clone()
        loss = 0
        #if self.training:
        #    p_grad = grad(d_hat.sum(),p,create_graph = True)[0]
        #    p_opt_grad = grad(d_opt.sum(),p_opt,create_graph = True)[0]
        #    loss = torch.relu(p_grad).sum() + torch.relu(p_opt_grad).sum()

        if r_opt.mean() >1000:
            print(p_opt.mean(),d_opt.mean())
        r_opt =GradRev.apply(r_opt)
        z=GradRev.apply(z)
        dom_cls = torch.sigmoid(self.critic(z))
        return r_opt, d_hat, dom_cls, loss





def init_weight(l):
  if isinstance(l,nn.Linear):
    nn.init.normal_(l.weight,mean=0,std=0.02)
    l.bias.data.fill_(0)



class Hack(nn.Module):
  def __init__(self,fn = FeedForward):
    super().__init__()
    self.l1 =  nn.Linear(15,1,bias=False)
    self.l2=  nn.Linear(15,1,bias=False)
  def forward(self,x):
    p=x[:,0].unsqueeze(1)
    xx=x[:,1:]
    x = self.l1(xx)+self.l2(xx)*p
    return x


#model=Hack()
model=Model()
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

model_name = "model_umn_new_0.1.pt"
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
