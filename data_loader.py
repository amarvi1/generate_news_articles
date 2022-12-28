import torch
import torch.nn as nn
from torchtext.transforms import BERTTokenizer
from torchtext.vocab import build_vocab_from_iterator

import pandas as pd
import numpy as np
import random

def prep_data():
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

  return x_tokens, vocab

def make_dataset(tokenized_data, seq_length):
  seq_len = seq_length
  train_data = []
  train_labels = []

  # Create train data and labels
  for article in tokenized_data:
    for i in range(len(article)-seq_len):
      sequence = article[i:i+seq_len]

      train_data.append(sequence)
      train_labels.append(article[i+seq_len])

  return train_data, train_labels

def get_batch(data, labels, vocab, batch_size):
  # Select a random subset of the data
  idx = random.sample(range(len(data)+1), batch_size)

  batch_train = []
  batch_labels = []
  for i in idx:
    data_seq = []
    for token in data[i]:
      data_seq.append(vocab[token])

    batch_train.append(data_seq)
    batch_labels.append(vocab[labels[i]])

  # Return batch as a tensor
  return torch.tensor(batch_train, dtype=torch.int32), torch.tensor(batch_labels)