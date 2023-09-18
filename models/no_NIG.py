
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import EvidentialRegressionLoss

class NormalInverseGammaNetwork(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            hidden_dim,
    ):
        super(NormalInverseGammaNetwork, self).__init__()
        
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = DenseNormalGamma(hidden_dim, dim_out)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DenseNormalGamma(nn.Module):
    def __init__(self, hidden_dim, units):
        super().__init__()
        self.units = int(units)
        self.dense = nn.Linear(hidden_dim, 4*self.units)
    
    def evidence(self, x):
        return torch.log(1+torch.exp(x)) # softplus

    def forward(self, x):
        output = self.dense(x)
        
        mu, log_v, log_alpha, log_beta = torch.chunk(output,4, dim=-1)

        v = self.evidence(log_v)
        alpha = self.evidence(log_alpha) + 1
        beta = self.evidence(log_beta)

        return torch.cat((mu, v, alpha, beta), dim=-1)
