import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import os
import numpy as np
from csv import reader
import re
import pathlib
from stanfordcorenlp import StanfordCoreNLP
import sys

sys.path.append('models')
from cnn_bert import CNN_BERT
from cnn_glove_cove import CNN_GC

sys.path.append('utils')
from prep_data import pre_process_text, pre_process_data

sys.path.append('dataLoaders')
from suggestion_loader import SuggestionDataset

#######################################################################
# 
#  Global Variable, to avoid repeated load
#   

# Ref : https://github.com/Lynten/stanford-corenlp
prefix = str(pathlib.Path(__file__).parent.parent)
path  = os.path.join(prefix, "pkgs", "stanford-corenlp-full-2016-10-31")
nlp = StanfordCoreNLP(path)

#######################################################################
#
#   Architecture is as follows
#   1. Bert word encoding + CNN with max pooling for sent.embed
#   2. Glove & CoVe word Encoding  + CNN with attention for sent.embed
#   3. (1) and (2) concated and fed to MLP for classification (1-sugg)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.l0 = nn.Linear(900,300) # BERT-> CNN (300), G,C->CNN (600)
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
        super(SuggestionClassifier,self).__init__()
        self.CNN_b = CNN_BERT()
        self.CNN_gc = CNN_GC()
        self.MLP = MLP()


    def forward(self,x):
        bert_x = self.CNN_b(x)
        gc_x = self.CNN_gc(x)
        torch.cat(bert_x,gc_x) # TODO : ensure dimensions align
        x = self.MLP(x)
        return x


#############################
# HELPER FUNCTIONS

def train(model,train_loader,valid_loader,test_loader, numEpochs=5):
    '''
    Args:

    Ret: 
    '''
    # TODO : 10-fold cross validation (build 10 models)
    # TODO : Ensemble, pick top three models, majority vote between these three

    model.train()
    val_loss_min = None
    val_loss_min_delta = 0
    val_patience = 0
    val_loss_counter = 0

    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in enumerate(train_loader):
            feats,labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(feats)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels

        torch.save(model.state_dict(), "jessi_A_"+str(epoch)+".pt")

        val_loss, val_acc = validate(model, valid_loader)
        test_loss, test_acc = test(model, test_loader, epoch)
        print('Epoch{:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}\tTest Loss: {:.4f}\tTest Accuracy: {:.4f}'.format(epoch, val_loss, val_acc, test_loss, test_acc))

        # Early stopping
        if val_loss_min is None:
            val_loss_min = val_loss
        elif val_loss_min >= (val_loss + val_loss_min_delta) :
            val_loss_counter += 1
            if (val_loss_counter > val_patience) :
                print("Validation Loss: {}\t Lowest Validation Loss {}\t".format(val_loss, val_loss_min))
                print("Training stopped early, Epoch :"+str(epoch))
                break
        else:
            val_loss_min = val_loss




def validate(model, valid_loader):
    '''
    Args:

    Ret:
    '''

    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(valid_loader):
        feats,labels = feats.to(device), labels.to(device)
        output = model(feats)

        _, pred_labesl = torch.max(output, dim=1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(output, labels.long())
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])

        #torch.cuda.empty_cache()
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total
    #return 0, accuracy/total


def test(model, test_loader, epoch):
    '''
    Args:

    Ret:
    '''

    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    
    outcomes = []
    
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[0]
        
        
        _, pred_labels = torch.max(outputs, dim=1)
        #loss = criterion(outputs, labels.long())
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        #test_loss.extend([loss.item()]*feats.size()[0])
        
        pred_labels = pred_labels.view(-1)
        pred_labels = pred_labels.cpu()
        pred_labels = np.array(pred_labels)
        outcomes += [ [test_loader.dataset.samples[batch_num*BATCH_SIZE+i][0].split('/')[-1],
            pred_labels[i]] for i in range(len(pred_labels)) ]
        
        del feats
        del labels

    model.train()
    
    f = open("Suggestion"+str(epoch)+".csv", "w", newline="")
    writer = csv.writer(f)
    writer.writerows(outcomes)
    f.close()
    
    #return np.mean(test_loss), accuracy/total
    return 0, accuracy/total


#########################
# HYPER PARAMETER TUNING

NUM_EPOCHS = 10
WEIGHT_DECAY = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.1 # TODO : Experiment, as LR not given

########################
# DATA LOADING
train_dataset = SuggestionDataset()
valid_dataset = SuggestionDataset(file="SubtaskA_Trial_Test_Labeled.csv")
test_dataset = SuggestionClassifier(file="SubtaskA_EvaluationData_labeled.csv", mode=0)

train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=8)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=8)

########################
# MODEL CREATION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SuggestionClassifier()
# TODO:  WEIGHT INIT

model.train()
model.to(device)

#######################
# OPTIMIZATION

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)