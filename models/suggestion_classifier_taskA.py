from models.cnn_bert import *
from models.cnn_glove_cove import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from stanfordcorenlp import StanfordCoreNLP

#######################################################################
#
#   Architecture is as follows
#   1. Bert word encoding + CNN with max pooling for sent.embed
#   2. Glove & CoVe word Encoding  + CNN with attention for sent.embed
#   3. (1) and (2) concated and fed to MLP for classification (1-sugg)
#
########################################################################

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.l0 = nn.Linear(TBD,300)
        self.dp0 = nn.Dropout(p=0.5)
        self.l1 = nn.Linear(300,300)
        self.dp1 = nn.Dropout(p=0.5)
        self.l3 = nn.Linear(300,2)

    def forward(self,x):
        x = self.dp0(F.sigmoid(self.l0(x)))
        x = self.dp1(F.sigmoid(self.l1(x)))
        x = F.sigmoid(self.l3(x))
        return x


class SuggestionClassifier(nn.Module):
    def __init_(self):
        # path all the models together
        # BERT->CNN + CNN->ATT
        pass

    def forward(self,x):
        pass


def pre_process(text):
    '''
    Lowercase, TOkenize (Stanford CoreNLP)
    '''
    text = text.lower()
    # Ref : https://github.com/Lynten/stanford-corenlp
    nlp = StanfordCoreNLP(r'..\pkgs\stanford-corenlp-full-2016-10-31')
    result = nlp.word_tokenize(text)
    return result


def train():
    pass


def validate():
    pass


def test():
    pass


########################
# DATA LOADING
train_loader = null
valid_loader = null
test_loader = null


########################
# MODEL CREATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuggestionClassifier()
#### ??? WEIGHT INIT

model.train()
model.to(device)

#########################
# Hyperparameters

WEIGHT_DECAY = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.1 ### ??

#######################
# OPTIMIZATION

optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)