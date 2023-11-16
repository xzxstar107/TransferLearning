draw_graph.py
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
device = "cpu"


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


class Model_mlp(nn.Module):
    def __init__(self,fn = FeedForward):
        super().__init__()
        self.mlp1=fn(15,15, n = 2)
        self.d_f = Demand()#MonotonicNN(16, [60, 60], nb_steps=100, dev=device)
        self.p_f=fn(15,1, n = 1)
        self.critic = fn(15,1,n=1)

    def forward(self, feat):
        p=feat[:,0].unsqueeze(1)
        x=feat[:,1:]
        z=self.mlp1(x)
        #p.requires_grad = True
        d_hat= torch.sigmoid(self.d_f(p,z)[:, :1])
        p_opt = torch.sigmoid(self.p_f(z))

                #with torch.no_grad():
        d_opt = torch.sigmoid(self.d_f(p_opt,z)[:, :1])
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




model_mon =  Model().eval()
model_mon.load_state_dict(torch.load("model_umn_new_0.1.pt"))
model_mlp_grad =  Model_mlp().eval()
model_mlp_grad.load_state_dict(torch.load("model_mlp_grad.pt"))
model_mlp =  Model_mlp().eval()
model_mlp.load_state_dict(torch.load("model_mlp.pt"))



M = 14
# Model parameters
a = np.asarray([[1.1, -0.1, 0, 0.1, 0, 0.2, 0, 0.1, -0.1, 0, 0, 0.1, -0.1, 0.2, -0.2]])
b = (-1) * np.asarray([[0.5, 0.1, -0.1, 0, 0, 0, 0, 0.2, 0.1, 0.2, 0, 0.2, -0.1, -0.2, 0]])
#
mu_S = np.repeat(1, M)
cov_S = 0.2 * np.identity(M, dtype=float)
X0_T = np.random.multivariate_normal(mu_S, cov_S, 1)

#df_T =10
#X0_T = stats.chi2.rvs(df_T, size=(1,M))
# add column of ones for intercept
X_T = sm.add_constant(X0_T,has_constant="add")
X_T[:,8:]=0
print(X_T)
print(X_T.shape)
print(a.shape)
d_min = -2.3989378562534847 
d_max = 2.025556543311613

def rescale(d_S):
    return (d_S-d_min)/(d_max-d_min)


def f(X_T,p_T):
    beta = a @ X_T.T
    alpha = (b @ X_T.T)
    d_T = beta +  alpha * p_T 
    p_opt = -beta/(2* alpha) 
    print("opt_p", p_opt)
    return rescale(d_T)

# <<TEST>>
x = torch.arange(0, 1, .1).unsqueeze(1).to(device)
y = f(X_T, 1.8*x.numpy()+0.2)
X_T1 = torch.tensor(X_T).expand(x.shape[0],-1).float()

y_mon = torch.sigmoid(model_mon.d_f(-x, X_T1)[:, 0]).detach().cpu().numpy()
p_mon = torch.sigmoid(model_mon.p_f(X_T1)[:, 0]).detach().cpu().numpy()
print("p_mon", p_mon)
y_mlp = torch.sigmoid(model_mlp.d_f(x, X_T1)[:, 0]).detach().cpu().numpy()
p_mlp = torch.sigmoid(model_mlp.p_f(X_T1)[:, 0]).detach().cpu().numpy()
print("p_mlp", p_mlp)

y_mlp_grad = torch.sigmoid(model_mlp_grad.d_f(x, X_T1)[:, 0]).detach().cpu().numpy()
p_mlp_grad = torch.sigmoid(model_mlp_grad.p_f(X_T1)[:, 0]).detach().cpu().numpy()
print("p_mlp_grad", p_mlp_grad)

x = x.detach().cpu().numpy()
plt.plot(x, y_mon, label="Monotonic MLP")
plt.plot(x, y_mlp, label="MLP")
plt.plot(x, y_mlp_grad, label="MLP+GradReg ")
plt.plot(x, y, label="Groundtruth")
plt.xlabel("Price")
plt.ylabel("Demand")
plt.legend()
plt.savefig("Monotonicity.png")
plt.show()
plt.close()




x = torch.arange(0, 1, .1).unsqueeze(1).to(device)
y = f(X_T, 1.8*x.numpy()+0.2)[:,0]
X_T1 = torch.tensor(X_T).expand(x.shape[0],-1).float()

y_mon = torch.sigmoid(model_mon.d_f(-x, X_T1)[:, 0]).detach().cpu().numpy()
y_mlp = torch.sigmoid(model_mlp.d_f(x, X_T1)[:, 0]).detach().cpu().numpy()
y_mlp_grad = torch.sigmoid(model_mlp_grad.d_f(x, X_T1)[:, 0]).detach().cpu().numpy()
x = x.detach().cpu().numpy()[:,0]
plt.plot(x, x*y_mon, label="Monotonic MLP")
plt.plot(x, x*y_mlp, label="MLP")
plt.plot(x, x*y_mlp_grad, label="MLP+GradReg ")
plt.plot(x, (1.8*x+0.2)*y, label="Groundtruth")
plt.xlabel("Price")
plt.ylabel("Revenue")
plt.legend()
plt.savefig("R-p.png")
plt.show()
