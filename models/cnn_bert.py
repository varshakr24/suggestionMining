import os
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from transformers import *

#bert large uncased
#hiddenlayers = 1024

class CNN_BERT(nn.Module):

    def __init__(self,out_dim=300,embed_dim=1024,max_len=1000, dropout=0.5):
        super(CNN_BERT, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-large-uncased')
        self.conv_3 = nn.Conv1d(1, out_dim, embed_size * 3, stride=embed_size)
        self.conv_4 = nn.Conv1d(1, out_dim, embed_size * 4, stride=embed_size)
        self.conv_5 = nn.Conv1d(1, out_dim, embed_size * 5, stride=embed_size)

    def get_conv_out(self,conv,x, num):
        return nn.functional.max_pool1d(nn.functional.relu(conv(x)), 
                                        max_len - num + 1).view(-1, feature_map)

    def forward(self,x):
        x = self.bert_layer(x)[0].view(-1, 1, embed_size*max_len)
        conv_3 = self.get_conv_out(self.conv_3,x,3)
        conv_4 = self.get_conv_out(self.conv_4,x,4)
        conv_5 = self.get_conv_out(self.conv_5,x,5)
        x = torch.cat([conv_3,conv_4,conv_5], 1)
        x = F.dropout(x, p=dropout)
        return x


