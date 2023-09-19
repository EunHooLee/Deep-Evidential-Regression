import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, hidden_dim, dim_out):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, dim_out * 4)
    
    def evidence(self, x):
        return torch.log(1+torch.exp(x)) # softplus

    def forward(self, x):
        output = self.linear(x).view(x.shape[0], -1, 4)
        mu, log_v, log_alpha, log_beta = [w.squeeze(-1) for w in torch.split(output, 1, dim=-1)]
        
        
        v = self.evidence(log_v)
        alpha = self.evidence(log_alpha) + 1
        beta = self.evidence(log_beta)
        
        # return torch.cat((mu, v, alpha, beta), dim=-1)
        return mu, v, alpha, beta
    
class SimpleNormalInverseGammaNetwork(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            hidden_dim,
    ):
        super(SimpleNormalInverseGammaNetwork, self).__init__()
        
        self.fc1 = nn.Linear(dim_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = SimpleDenseNormalGamma(hidden_dim, dim_out)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
class SimpleDenseNormalGamma(nn.Module):
    def __init__(self, hidden_dim, dim_out):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, dim_out * 4)
    
    def evidence(self, x):
        return torch.log(1+torch.exp(x)) # softplus

    def forward(self, x):
        output = self.linear(x).view(x.shape[0], -1, 4)
        mu, log_v, log_alpha, log_beta = [w.squeeze(-1) for w in torch.split(output, 1, dim=-1)]
        
        
        v = self.evidence(log_v)
        alpha = v + 1
        beta = self.evidence(log_beta)
        
        # return torch.cat((mu, v, alpha, beta), dim=-1)
        return mu, v, alpha, beta