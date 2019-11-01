
# modified example from https://pytorch.org/docs/1.2.0/nn.html#lstm

import torch
import torch.nn as nn
import rnn_util


bidirectional = True
num_directions = 2 if bidirectional else 1

rnn = rnn_util.LayerNormLSTM(10, 20, 2, dropout=0.3, r_dropout=0.25,
                             bidirectional=bidirectional, layer_norm_enabled=True)
# rnn = nn.LSTM(10, 20, 2, bidirectional=bidirectional)

input = torch.randn(5, 3, 10)
h0 = torch.randn(2 * num_directions, 3, 20)
c0 = torch.randn(2 * num_directions, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))

print(output.size())
