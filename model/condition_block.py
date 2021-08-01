import torch
from torch import nn
from fastai.layers import init_linear

class CIN(nn.Module):
    
    def __init__(self, dim_out, embed_dim):
        super().__init__()
       
        self.gamma = nn.Linear(embed_dim, dim_out)
        init_linear(self.gamma)
        self.beta = nn.Linear(embed_dim, dim_out)
        init_linear(self.beta)
    
    def forward(self, x, embed):
        sigma, mu = torch.std_mean(x, dim=2, keepdim=True)
        sigma = torch.clamp(sigma, min=1e-7)
        gamma = self.gamma(embed)[..., None]
        beta = self.gamma(embed)[..., None]
        return gamma * (x - mu) / sigma + beta


class ConditioningBlock(nn.Module):
    
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, embed_dim):
        super(ConditioningBlock, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=dim_in, out_channels=dim_out, 
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        self.cin = CIN(dim_out, embed_dim)
        self.glu = nn.GLU(dim=1)

    def forward(self, x, embed):
        x = self.conv(x)
        x = self.cin(x, embed)
        x = self.glu(x)
        return x
