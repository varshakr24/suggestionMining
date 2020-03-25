from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe

from cove import MTLSTM

TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
LABEL = data.Field(lower=True, include_lengths=True, batch_first=True)

train_path = "C:\\Users\\bhara\\Downloads\\NNNlpHW3\\suggestionMining\\data\\Subtask-A\\V1.4_Training.csv"
train = data.TabularDataset(
        path=train_path, format='csv',
        fields=[('id', None),
                ('sentence', TEXT),
                 ('label', LABEL)])

TEXT.build_vocab(train, vectors=GloVe(name='840B', dim=300, cache='.embeddings'))
LABEL.build_vocab(train)
outputs_cove_with_glove = MTLSTM(n_vocab=len(TEXT.vocab), vectors=TEXT.vocab.vectors, residual_embeddings=True, model_cache='.embeddings')
#glove_then_first_then_last_layer_cove = outputs_both_layer_cove_with_glove(<pass a sentence Glove embedding>)

train_iter = data.Iterator(
    (train),
    batch_size=100)

z = None
for batch_idx, batch in enumerate(train_iter):
    z = batch
    glove_then_last_layer_cove = outputs_cove_with_glove(*batch.sentence)
    print(glove_then_last_layer_cove.size())