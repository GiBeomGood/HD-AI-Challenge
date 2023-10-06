import torch
from torch import nn


class SmallBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()

        self.small_block_layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.activation = activation
        self.relu = nn.ReLU()

        return
    
    def forward(self, x):
        output = self.small_block_layer(x)
        if self.activation is True:
            return self.relu(output)
        else:
            return output


class BigBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.small = nn.Sequential(
            SmallBlock(in_dim, hid_dim),
            SmallBlock(hid_dim, out_dim, activation=False),
        )
        self.relu = nn.ReLU()
        
        if in_dim==out_dim:
            self.identity = lambda x: x
        else:
            self.identity = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        
        return
    
    def forward(self, x):
        output = self.small(x)
        output = self.relu(output + self.identity(x))
        return output


class Baseline(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden_dim = int(1.5*input_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.big1 = BigBlock(hidden_dim, input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

        return
    
    def forward(self, x):
        output = self.layer1(x)
        output = self.big1(output)
        output = self.layer2(output).squeeze(dim=1)
        return output