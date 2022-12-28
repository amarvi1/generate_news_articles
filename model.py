import torch
import torch.nn as nn

class NewsGenerator(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers):
        super(NewsGenerator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, hidden):
        x = self.embedding(x)

        out, hidden = self.lstm(x, hidden)
        out = self.linear(out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size), 
                torch.zeros(self.num_layers, batch_size, self.hidden_size))