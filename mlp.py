import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=layer_dim)
        self.fc2 = nn.Linear(in_features=layer_dim, out_features=layer_dim)
        self.output = nn.Linear(in_features=layer_dim, out_features=output_dim)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.output(x)
        return x