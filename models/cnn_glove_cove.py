import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data


class CNN_GC(nn.Module):

    def __init__(self,out_dim=200,gc_dim=900,max_len=512, dropout=0.5):
        super(CNN_GC, self).__init__()
        
        """cove_model = CoVeEmbeddings(
            word_embeddings_dir='../model/text/stanford/glove/', 
            tokenizer=tokenizer,
            max_sequence_length=max_len, verbose=20)
        glove_model = GloVeEmbeddings()
        glove_model.load_model(dest_dir='../model/text/stanford/glove/', process=False)"""
        
        self.gc_dim = gc_dim
        self.out_dim = out_dim
        self.max_len = max_len
        self.dropout = dropout

        self.conv_3 = nn.Conv1d(gc_dim, out_dim, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv1d(gc_dim, out_dim, 5, stride=1, padding=2)
        self.conv_7 = nn.Conv1d(gc_dim, out_dim, 7, stride=1, padding=3)
        self.attn = nn.Linear(3*out_dim*max_len, max_len)

    def forward(self,x):
        #cove_embed = cove_model.encode(x)
        #tokens = [sentence.split(" ") for sentence in x]
        #glove_embed = glove_model.encode(tokens)
        #x = torch.cat([cove_embed,glove_embed], 2)
        
        conv_3 = F.relu(self.conv_3(x))
        conv_5 = F.relu(self.conv_5(x))
        conv_7 = F.relu(self.conv_7(x))
        x = torch.cat([conv_3,conv_5,conv_7], 1)
        print(x.shape)
        
        non_linear_x = F.relu(x.view(-1, 600*self.max_len))
        print(non_linear_x.shape)
        attn_weights = F.softmax(self.attn(non_linear_x), dim=1)
        print(attn_weights.shape)
        #attn_applied = torch.bmm(attn_weights.unsqueeze(1), x)
        #attn_applied = attn_weights*x
        attn_applied = torch.zeros(x.shape[0], x.shape[1], x.shape[2])
        for i in range(x.shape[0]):
            attn_applied[i,:,:] = x[i,:,:]*attn_weights[i]
        print("hello")
        print(attn_applied.shape)
        
        x = attn_applied.sum(dim=2)
        return x