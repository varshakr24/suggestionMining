import os
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from aion.embeddings.cove import CoVeEmbeddings
from aion.embeddings.glove import GloVeEmbeddings


class CNN_GC(nn.Module):

    def __init__(self,out_dim=200,gc_dim=1024,max_len=1000, dropout=0.5):
        super(CNN_GC, self).__init__()
        
        cove_model = CoVeEmbeddings(
            word_embeddings_dir='../model/text/stanford/glove/', 
            max_sequence_length=max_len, verbose=20)
        glove_model = GloVeEmbeddings()
        glove_model.load_model(dest_dir='../model/text/stanford/glove/', process=False)
        
        self.gc_dim = gc_dim
        self.out_dim = out_dim
        self.max_len = max_len
        self.dropout = dropout

        self.bert_layer = BertModel.from_pretrained('bert-large-uncased')
        self.conv_3 = nn.Conv1d(gc_dim, out_dim, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv1d(gc_dim, out_dim, 5, stride=1, padding=2)
        self.conv_7 = nn.Conv1d(gc_dim, out_dim, 7, stride=1, padding=3)
        self.attn = nn.Linear(gc_dim*max_len, max_len)

    def forward(self,x):
        cove_embed = cove_model.encode(x)
        glove_embed = glove_model.encode(x)
        x = torch.cat([cove_embed,glove_embed], 2)
        
        conv_3 = F.relu(self.conv_3(x))
        conv_5 = F.relu(self.conv_5(x))
        conv_7 = F.relu(self.conv_7(x))
        x = torch.cat([conv_3,conv_4,conv_5], 1)
        
        non_linear_x = F.relu(x.view(-1, 600*self.max_len))
        attn_weights = F.softmax(self.attn(non_linear_x), dim=1)
        #attn_applied = torch.bmm(attn_weights.unsqueeze(1), x)
        attn_applied = attn_weights*x
        
        x = attn_applied.sum(dim=2)
        return x


