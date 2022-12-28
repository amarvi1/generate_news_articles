import torch
import torch.nn as nn
from torchtext.transforms import BERTTokenizer
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import numpy as np

from data_loader import prep_data, make_dataset, get_batch
from model import NewsGenerator

# Load data
filename = '/content/articles1.csv'
news_data = pd.read_csv(filename)

headlines_articles = news_data[['title', 'content']].values.tolist()
x = [headlines + ' ' + articles for headlines, articles in headlines_articles]

# Create BERT tokenizer & tokenize data
VOCAB_FILE = 'https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt'
tokenizer = BERTTokenizer(VOCAB_FILE, do_lower_case=True, return_tokens=True)

x_tokens = tokenizer(x)

# Build vocabulary (vocab_size = 10000)
vocab = build_vocab_from_iterator(x_tokens, specials=['<unk>'], max_tokens=10000)
vocab.set_default_index(vocab['<unk>'])

# Create dataset
data_tokens, vocab = prep_data()
train_data, train_labels = make_dataset(data_tokens[:100], 30)
# Batch dataset
batch_data, batch_labels = get_batch(train_data, train_labels, vocab, 128)

vocab_size = len(vocab)
input_size = 100
hidden_size = 256
num_layers = 2
# Init model
model = NewsGenerator(vocab_size, input_size, hidden_size, num_layers)
hidden = model.init_hidden(128)

# Train model
train_pred, hidden = model(batch_data, hidden)
