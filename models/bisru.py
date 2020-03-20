import os
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

class BISRU(nn.Module):

    def __init__(self):
        super(BISRU, self).__init__()


    def forward(self,x):
        return x
