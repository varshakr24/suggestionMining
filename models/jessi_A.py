import torch
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

# sys.path.append('models')
# from cnn_bert import CNN_BERT
# from cnn_glove_cove import CNN_GC

sys.path.append('utils')
from prep_data import pre_process_text, load_data, create_cross_val_train_test, create_folds

sys.path.append('dataLoaders')
from dummy_loader import DummyDataset

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
        self.l1 = nn.Linear(300,300)
        self.l3 = nn.Linear(300,2)
        self.dp = nn.Dropout(p=0.5)

    def forward(self,x):
        print(x)
        print(x.shape)
        quit()
        x = self.dp(F.sigmoid(self.l0(x)))
        x = self.dp(F.sigmoid(self.l1(x)))
        x = F.sigmoid(self.l3(x))
        return x


# class SuggestionClassifier(nn.Module):
#     def __init_(self):
#         # path all the models together
#         # BERT->CNN + CNN->ATT
#         super(SuggestionClassifier,self).__init__()
#         # self.CNN_b = CNN_BERT()
#         # self.CNN_gc = CNN_GC()
#         self.MLP = MLP()


#     def forward(self, x):
#         # bert_x = self.CNN_b(x[0])
#         # gc_x = self.CNN_gc(x[1])
#         bert_x = x[0]
#         gc_x = x[1]
        
#         out = torch.cat(bert_x,gc_x) # TODO : ensure dimensions align (1x300, 1x600)
#         out = self.MLP(out) # Dim : 1x900
#         return out


#############################
# HELPER FUNCTIONS

def train(model, train_loader, valid_loader, test_loader, foldId, numEpochs=5):
    '''
    Args:

    Ret: 
    '''

    model.train()
    val_loss_min = None
    val_loss_min_delta = 0
    val_patience = 0
    val_loss_counter = 0

    # TODO : 10-fold cross validation (build 10 models)
    # TODO : Ensemble, pick top three models based on train loss
    # TODO : Voting Mechanism to evaluate on Validation data
    
    # for each fold
    #     for each epoch
    #       min_batch_train(model)
    #       val_loss = model(trial_data)
    #       early_stopping(val_loss)
    #     models += models
    #     three_models = select_best_3(models) based on train_loss
        
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
                print('Fold: {}\t Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(foldId+1, epoch+1, batch_num+1, avg_loss/50))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels

        model_name = "jessi_A_fold_"+str(foldId+1)+"_epoch_"+str(epoch+1)+".pt"
        torch.save(model.state_dict(), model_name)

        val_loss, val_acc = validate(model, valid_loader)
        test_loss, test_acc = validate(model, test_loader)
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

        break

        return test_acc, model_name


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


def test(models, test_loader, id_map=0):
    '''
    Args:

    Ret:
    '''

    test_loss = []
    accuracy = 0
    total = 0
    
    outcomes = []
    
    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        predictions = torch.tensor(np.zeros(len(labels))).to(device)

        for model in models:
            model.eval()
            outputs = model(feats)
            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            predictions += pred_labels.view(-1)
            model.train()

        predictions = np.round(predictions/3).astype(int)
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)       
        # outcomes += [ [id_map(test_loader.dataset.samples[batch_num * BATCH_SIZE + i]),predictions[i]] for i in range(len(predictions)) ]
            
        del feats
        del labels
    
    # f = open("Suggestion.csv", "w", newline="")
    # writer = csv.writer(f)
    # writer.writerows(outcomes)
    # f.close()
    
    #return np.mean(test_loss), accuracy/total
    return -1, accuracy/total


#########################
# HYPER PARAMETER TUNING

NUM_EPOCHS = 10
WEIGHT_DECAY = 3
BATCH_SIZE = 32
NFOLDS = 10
NUM_WORKERS = 1
LEARNING_RATE = 0.1 # TODO : Experiment, as LR not given

########################
# MODEL CREATION

def init_weights(m):
    torch.nn.init.kaiming_normal_(m.weight.data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP()
# TODO:  WEIGHT INIT
# model.apply(init_weights)
model.train()
model.to(device)

#######################
# OPTIMIZATION

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

########################
# DATA LOADING

# valid_dataset = SuggestionDataset(load_data(filename="SubtaskA_Trial_Test_Labeled.csv"))
valid_dataset = DummyDataset()
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=False, num_workers=NUM_WORKERS)

# test_dataset = SuggestionDataset(load_data(filename="SubtaskA_EvaluationData_labeled.csv"), mode=0)
# test_id_map = test_dataset.get_map()
test_dataset = DummyDataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=False, num_workers=NUM_WORKERS)

data = load_data()
data_folds = create_folds(data)
model_perfs = []
model_names = []
for i in range(NFOLDS):
    # train, test = create_cross_val_train_test(data_folds,i)
    # train_dataset = SuggestionDataset(train)
    train_dataset = DummyDataset()
    train_loader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=True, num_workers=NUM_WORKERS)
    # test_dataset = SuggestionClassifier(test)
    test_dataset = DummyDataset()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                                               shuffle=False, num_workers=NUM_WORKERS)

    test_loss, model_name = train(model, train_loader, valid_loader, test_loader, i, NUM_EPOCHS)

    model_perfs.append(test_loss)
    model_names.append(model_name)

best_three_model_idx = np.flip(np.argsort(model_perfs))[0:3]
models = []
for idx in best_three_model_idx:
    # load the model
    temp_model = SuggestionClassifier()
    temp_model.load_state_dict(torch.load(model_names[idx]))
    models.append(temp_model)
# testing
test(models, test_loader)

