import torch
from torch.utils.data import Dataset, DataLoader
from stanfordcorenlp import StanfordCoreNLP
import sys
import pathlib
import os
from csv import reader
import re

sys.path.append('utils')
from prep_data import pre_process_data_from_dataset, pre_process_text, load_data, create_folds, create_cross_val_train_test 
from get_embeddings import bert_embedding

# prefix = str(pathlib.Path(__file__).parent.parent)
# path  = os.path.join(prefix, "pkgs", "stanford-corenlp-full-2016-10-31")
# nlp = StanfordCoreNLP(path)

class SuggestionDataset(Dataset):
    def __init__(self, dataset, mode=1):
        self.feats, self.labels, self.id_map = pre_process_data_from_dataset(dataset)
        self.mode = mode

        self.bert_feats = get_bert_embedding(dataset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.mode == 0:
            return self.feats[index], self.bert_feats[index], '0'
        else:
            return self.feats[index], self.bert_feats[index], self.labels[index]

    def get_map(self):
        return self.id_map

#Testing
# data = load_data()
# data_folds = create_folds(data)
# train, test = create_cross_val_train_test(data_folds,0)
# train_loader = SuggestionDataset(train)
# print(train_loader.__getitem__(0))

