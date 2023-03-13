
# modified example from https://pytorch.org/docs/1.2.0/nn.html#lstm

import torch
from torch.nn import RNNCell, LSTMCell
import rnnlib

seq_len = 5
batch_size = 3
input_size = 8

hidden_size = 10
num_layers = 2
dropout = 0.3
r_dropout = 0.25
bidirectional = True
num_directions = 2 if bidirectional else 1

# ------------------------ Examples of RNNFrame ------------------------
rnn_cells = [
    [RNNCell(input_size, hidden_size),
     RNNCell(input_size, hidden_size)],  # 1st bidirectional RNN layer
    [RNNCell(hidden_size * num_directions, hidden_size),
     RNNCell(hidden_size * num_directions, hidden_size)]  # 2nd bidirectional RNN layer
]

assert len(rnn_cells) == num_layers
assert all(len(rnn_layer_cells) == num_directions for rnn_layer_cells in rnn_cells)

# with batch_first=False ------------------------------------------------
rnn = rnnlib.RNNFrame(rnn_cells, dropout=dropout, bidirectional=bidirectional)
# rnn = torch.nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional)

input = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
output, hn = rnn(input, h0)

print(output.size())

# with batch_first=True ------------------------------------------------
rnn = rnnlib.RNNFrame(rnn_cells, dropout=dropout, bidirectional=bidirectional, batch_first=True)
# rnn = torch.nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)

input = torch.randn(batch_size, seq_len, input_size)
h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
output, hn = rnn(input, h0)

print(output.size())

# ------------------------ Examples of LSTMFrame ------------------------
rnn_cells = [
    [LSTMCell(input_size, hidden_size),
     LSTMCell(input_size, hidden_size)],  # 1st bidirectional LSTM layer
    [LSTMCell(hidden_size * num_directions, hidden_size),
     LSTMCell(hidden_size * num_directions, hidden_size)]  # 2nd bidirectional LSTM layer
]
# 'rnn_cells' is a list of forward/backward LSTM cell pairs.
# Each pair corresponds to a layer of bidirectional LSTM.
# You can replace 'LSTMCell' with your custom LSTM cell class.
# Also you can compose 'rnn_cells' with heterogeneous LSTM cells.
#
# Caution: Non-LSTM cells, which don't distinguish hidden states and cell states,
#          such as 'RNNCell' or 'GRUCell', are not allowed to be included in 'rnn_cells'

assert len(rnn_cells) == num_layers
assert all(len(rnn_layer_cells) == num_directions for rnn_layer_cells in rnn_cells)

# with batch_first=False ------------------------------------------------
rnn = rnnlib.LSTMFrame(rnn_cells, dropout=dropout, bidirectional=bidirectional)
# rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

input = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))

print(output.size())

# with batch_first=True ------------------------------------------------
rnn = rnnlib.LSTMFrame(rnn_cells, dropout=dropout, bidirectional=bidirectional, batch_first=True)
# rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)

input = torch.randn(batch_size, seq_len, input_size)
h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))

print(output.size())

# ------------------------ Examples of LayerNormLSTM ------------------------
rnn = rnnlib.LayerNormLSTM(input_size, hidden_size, num_layers, dropout=dropout, r_dropout=r_dropout,
                             bidirectional=bidirectional, layer_norm_enabled=True)
# rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

input = torch.randn(seq_len, batch_size, input_size)
h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
output, (hn, cn) = rnn(input, (h0, c0))

print(output.size())
