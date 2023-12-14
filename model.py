import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return math.sqrt(self.d_model) * self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        positional_encoding_matrix = torch.zeros(seq_len, d_model)

        # vector - (seq_len, 1)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # mysterious positional encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        positional_encoding_matrix[:, 0:2] = torch.sin(positions * div_term)
        positional_encoding_matrix[:, 1:2] = torch.cos(positions * div_term)

        # todo: there will be a batch of sentences so we unsqueeze. I dont have a good feel for this hence the todo

        # (1, seq_len, d_model)
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0)

        self.register_buffer('positional_encoding_matrix', positional_encoding_matrix)

    def forward(self, x):
        x = x + (self.positional_encoding_matrix[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
