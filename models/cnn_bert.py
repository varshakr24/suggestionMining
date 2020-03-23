import os
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from transformers import BertModel

# Bert-large uncased (IN_CHANNELS : 300)
# hiddenlayers = 1024 (OUT_CHANNELS : 1024)

class CNN_BERT(nn.Module):

    def __init__(self, out_dim=300, embed_dim=1024, max_len=1000, dropout=0.5):
        super(CNN_BERT, self).__init__()
        
        self.embed_dim = embed_dim
        self.out_dim = out_dim # not 100?
        self.max_len = max_len # ?
        self.dropout = dropout

        self.bert_layer = BertModel.from_pretrained('bert-large-uncased')
        self.conv_3 = nn.Conv1d(1, out_dim, embed_dim * 3, stride=embed_dim)
        self.conv_4 = nn.Conv1d(1, out_dim, embed_dim * 4, stride=embed_dim)
        self.conv_5 = nn.Conv1d(1, out_dim, embed_dim * 5, stride=embed_dim)

    def get_conv_out(self,conv,x, num):
        return F.max_pool1d(F.relu(conv(x)), 
                            self.max_len - num + 1).view(-1, self.out_dim)

    def forward(self,x):
        x = self.bert_layer(x)[0].view(-1, 1, self.embed_dim * self.max_len) # better 1 right ?
        conv_3 = self.get_conv_out(self.conv_3, x, 3)
        conv_4 = self.get_conv_out(self.conv_4, x, 4)
        conv_5 = self.get_conv_out(self.conv_5, x, 5)
        x = torch.cat([conv_3, conv_4, conv_5], 1)
        x = F.dropout(x, p=self.dropout)
        return x


