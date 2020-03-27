import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DummyDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 96

    def __getitem__(self, index):
        bert_embed = torch.tensor(np.zeros((300,1000)))
        glove_cove_embed = torch.tensor(np.zeros((600,1000)))
        return (bert_embed, glove_cove_embed), torch.tensor(0)

    def get_map(self):
        return self.id_map


sample = DummyDataset()
#print(sample.__getitem__(0))
train_loader =  torch.utils.data.DataLoader(sample, batch_size=32, 
                                               shuffle=True, num_workers=1)

for batch_num, (feats, labels) in enumerate(train_loader):
    print("Feats", feats)
    print("Labels", labels)
    break


