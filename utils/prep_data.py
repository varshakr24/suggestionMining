import os
from csv import reader
import re
import pathlib
from stanfordcorenlp import StanfordCoreNLP
import sys


# Ref : https://github.com/Lynten/stanford-corenlp
prefix = str(pathlib.Path(__file__).parent.parent)
path  = os.path.join(prefix, "pkgs", "stanford-corenlp-full-2016-10-31")
nlp = StanfordCoreNLP(path)

# def pre_process_data(folder="Subtask-A", filename="SubtaskA_EvaluationData_labeled.csv"):
#     '''
#     '''
#     prefix = str(pathlib.Path(__file__).parent.parent)
#     path = os.path.join(prefix,"data", folder, filename)
    
#     f = open(path,'r', encoding="utf-8")
#     data_reader = reader(f, delimiter=",")
#     data = [row for row in data_reader]
#     f.close()

#     ids = [datum[0] for datum in data]
#     id_map = {k:v for k,v in enumerate(ids)}

#     labels = [datum[2] for datum in data]

#     feats = [pre_process_text(datum[1]) for datum in data]
#     return feats, labels, id_map


def pre_process_text(text):
    '''
    Lowercase, TOkenize (Stanford CoreNLP)
    '''
    text = text.lower()
    result = nlp.word_tokenize(text)
    return result


def load_data(folder= "Subtask-A", filename="SubtaskA_EvaluationData_labeled.csv"):
    '''
    Args : folder name, file name
    Ret : return data loaded into list of lists [id, string, labels] 
    '''
    prefix = str(pathlib.Path(__file__).parent.parent)
    path = os.path.join(prefix,"data", folder, filename)
    f = open(path,'r', encoding="utf-8")
    data_reader = reader(f, delimiter=",")
    data = [row for row in data_reader]
    f.close()
    return data


def pre_process_data_from_dataset(data):
    '''
    Args : data is list of lists [id, string, label]
    Ret: list of features, labels, id_map (i.e index to id mapping)
    '''
    ids = [ datum[0]+",\""+datum[1]+"\"" for datum in data]
    id_map = {k:v for k,v in enumerate(ids)}

    labels = [datum[2] for datum in data]

    feats = [pre_process_text(datum[1]) for datum in data]
    return feats, labels, id_map


def create_folds(data, folds=10):
    '''
    Split data into 'folds' number of batches

    Args : data, list of lists of form [id, string, label]
            folds, number of batches of data to be created
    Rets : data batched into 'fold' lists, each wich is in turn list of lists [id, string, label]
    '''
    data_size = len(data)
    batch_size = int(data_size/folds)
    data_batch = []
    last_index = 0
    for i in range(folds-1):
        batch = data[i * batch_size: (i+1)*batch_size]
        data_batch.append(batch)
    data_batch.append(data[(folds-1) * batch_size:])
    return data_batch


def create_cross_val_train_test(data_batches,id, folds=10):
    '''
    Create test set from batched data, where test set while batch[id]
    and train set will everything else

    Args : data_batches, data that is split into 'fold' number of groups
            id, index of batch to be made as test_set
            folds, number of batches the data is split into
    Rets :
    '''
    train = []
    test = data_batches[id]
    for i in range(folds):
        if i != id:
            train += data_batches[i]
    return train, test



#testing
data = load_data()
data_folds = create_folds(data)
for datum in data_folds:
    print(len(datum))
print(data_folds[0][0])
train, test = create_cross_val_train_test(data_folds,0)
print("Length of train and test: ", len(train),len(test))

feats, labels, id_map = pre_process_data_from_dataset(train)
print(feats[0])
print(id_map[2])
# feats, labels, id_map = pre_process_data_from_dataset(data_folds[1])
# print(feats[1])
