import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        

    def forward(self,x):
        return x