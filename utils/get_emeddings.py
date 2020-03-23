from pytorch_pretrained_bert.tokenization import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)

def bert_embedding(data):
	feats = [tokenizer.tokenize(datum) for datum in data]
	return feats

