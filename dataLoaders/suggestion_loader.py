import torch
from torch.utils.data import Dataset, DataLoader
from stanfordcorenlp import StanfordCoreNLP

from utils.prep_data import pre_process_data, pre_process_text

prefix = str(pathlib.Path(__file__).parent.parent)
path  = os.path.join(prefix, "pkgs", "stanford-corenlp-full-2016-10-31")
nlp = StanfordCoreNLP(path)


class SuggestionDataset(Dataset):
    def __init__(self, folder='Subtask-A', file='V1.4_Training.csv', mode=1):
        self.feats, self.labels, self.id_map = pre_process_data(filename=filename)
        self.mode = mode


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if mode == 0:
            return self.feats[index], '0'
        else:
            return self.feats[index], self.labels[index]

