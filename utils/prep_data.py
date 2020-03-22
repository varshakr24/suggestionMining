import os
from csv import reader
import re
import pathlib
from stanfordcorenlp import StanfordCoreNLP


# Ref : https://github.com/Lynten/stanford-corenlp
prefix = str(pathlib.Path(__file__).parent.parent)
path  = os.path.join(prefix, "pkgs", "stanford-corenlp-full-2016-10-31")
nlp = StanfordCoreNLP(path)

def pre_process_text(text):
    '''
    Lowercase, TOkenize (Stanford CoreNLP)
    '''
    text = text.lower()
    result = nlp.word_tokenize(text)
    return result

def pre_process_data(folder="Subtask-A", filename="SubtaskA_EvaluationData_labeled.csv"):
    '''
    '''
    prefix = str(pathlib.Path(__file__).parent.parent)
    path = os.path.join(prefix,"data", folder, filename)
    
    f = open(path,'r', encoding="utf-8")
    data_reader = reader(f, delimiter=",")
    data = [row for row in data_reader]
    f.close()

    ids = [datum[0] for datum in data]
    id_map = {k:v for k,v in enumerate(ids)}

    labels = [datum[2] for datum in data]

    feats = [pre_process_text(datum[1]) for datum in data] # TODO : removed quotes
    return feats, labels, id_map


# Testing
# feats, labels, id_map = pre_process_data()
# print(feats[0])
# print(labels)
# print(id_map)
