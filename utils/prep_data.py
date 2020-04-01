import os
import re
import sys
import pathlib
import numpy as np
import csv
from csv import reader
from stanfordcorenlp import StanfordCoreNLP
from transformers import AutoTokenizer
import random
# from aion.embeddings.cove import CoVeEmbeddings
# from aion.embeddings.glove import GloVeEmbeddings
# from pytorch_pretrained_bert.tokenization import BertTokenizer

# Ref : https://github.com/Lynten/stanford-corenlp
prefix = str(pathlib.Path(__file__).parent.parent)
path  = os.path.join(prefix, "pkgs", "stanford-corenlp-full-2016-10-31")
nlp = StanfordCoreNLP(path)

# Bert TOkenizer
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
def tokenizerfnc(str):
    return tokenizer.encode(str)

# Ref : 
# Bert Embedding
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

# # Ref : 
# # Glove Embedding
# glove_model = GloVeEmbeddings()
# glove_model.load_model(dest_dir='../model/text/stanford/glove/', process=False)


# # Ref
# # Cove Embedding
# cove_model = CoVeEmbeddings(
#             word_embeddings_dir='../model/text/stanford/glove/', 
#             tokenizer=nlp,
#             max_sequence_length=1000, verbose=20)


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
    # result = text.split(' ')
    return result


# def bert_embedding(text):
#     '''
#     Get bert tokenized sentences
#     '''
#     return tokenizer.encode(tokenizer.tokenize(text))


# def glove_cove_embedding(text, tokenizer=nlp):
#     '''
#     Get Glove_Cove Embedding
#     '''
#     tokens = [nlp.word_tokenize(sentence) for sentence in text]
#     glove_embed = glove_model.encode(tokens)
#     cove_embed = cove_model.encode(tokens)
#     result = np.concatenate(glove_embed,cove_embed, axis=2)
#     return result




def load_data(folder= "Subtask-A", filename="SubtaskA_EvaluationData_labeled.csv", header=True):
    '''
    Args : folder name, file name
    Ret : return data loaded into list of lists [id, string, labels] 
    '''
    prefix = str(pathlib.Path(__file__).parent.parent)
    path = os.path.join(prefix,"data", folder, filename)
    f = open(path,'r', encoding="utf-8", newline='')
    data_reader = reader(f, delimiter=",")
    data = [row for row in data_reader]
    if header:
        data = data[1:]
    #tokenizerfnc
    # lens = [len(pre_process_text(datum[1])) for datum in data]
    # lens = [len(tokenizerfnc(datum[1])) for datum in data]
    # lens.sort()
    # print(lens)
    val = [int(datum[2]) for datum in data]
    val = np.array(val)
    print("Number of ones: ", val.sum())
    print("Length of Obs: ", len(val))

    # Class Imbalance
    # all_data = []
    # for datum in data:
    #     if int(datum[2]) == 1:
    #         all_data.append(datum)
    #         all_data.append(datum)
    #     all_data.append(datum)
    # f = open("out.csv", "w", encoding="utf-8", newline="")
    # writer = csv.writer(f)
    # writer.writerows(all_data)
    # f.close()
    # random.shuffle(all_data)

    return data


def pre_process_data_from_dataset(data):
    '''
    Args : data is list of lists [id, string, label]
    Ret: list of features, labels, id_map (i.e index to id mapping)
    '''
    ids = [ datum[0]+",\""+datum[1]+"\"" for datum in data]
    id_map = {k:v for k,v in enumerate(ids)}

    labels = [datum[2] for datum in data]

    bert_feats = [pre_process_text(datum[1]) for datum in data]
    # glove_cove_feats = glove_cove_embedding([datum[1] for datum in data], nlp)
    return bert_feats, labels, id_map


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


def create_cross_val_train_test(data_batches, id, folds=10):
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
data = load_data(filename='out.csv')
# # #print(len(data))
# data_folds = create_folds(data)
# #for datum in data_folds:
#     #print(len(datum))
# #print(data_folds[0][0])
# for i in range(10):
#     train, test = create_cross_val_train_test(data_folds,i)

#     f = open("train_"+str(i)+".csv", "w", encoding='utf-8', newline="")
#     writer = csv.writer(f)
#     writer.writerows([['id','sentence','label']])
#     writer.writerows(train)
#     print(len(train))
#     f.close()
#     print("Done")

#     f = open("test_"+str(i)+".csv", "w", encoding='utf-8', newline="")
#     writer = csv.writer(f)
#     writer.writerows([['id','sentence','label']])
#     writer.writerows(test)
#     print(len(test))
#     f.close()
#     print("Done")

# data = load_data(filename="train_1.csv")
# print(len(data))

#print("Length of train and test: ", len(train),len(test))

# feats, bert, labels, id_map = pre_process_data_from_dataset(train)
# print(feats[0])
# print(bert[0])
# print(id_map[2])
# feats, labels, id_map = pre_process_data_from_dataset(data_folds[1])
# print(feats[1])
