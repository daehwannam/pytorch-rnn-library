# Copyright (c) 2019 Daehwan Nam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch


def no_dropout(x): return x


no_dropout.p = 0


def no_layer_norm(x): return x


def get_indicator(lengths, max_length=None):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.Tensor(lengths)
    lengths_size = lengths.size()

    flat_lengths = lengths.view(-1, 1)

    if not max_length:
        max_length = lengths.max()
    unit_range = torch.arange(max_length)
    # flat_range = torch.stack([unit_range] * flat_lengths.size()[0],
    #                          dim=0)
    flat_range = unit_range.repeat(flat_lengths.size()[0], 1)
    flat_indicator = flat_range < flat_lengths

    return flat_indicator.view(lengths_size + (-1, 1))


def create_lstm_init_state(hidden_size, learn_init_state):
    init_hidden = nn.Parameter(torch.zeros(hidden_size), learn_init_state)
    init_cell = nn.Parameter(torch.zeros(hidden_size), learn_init_state)

    init_state = (init_hidden, init_cell)
    _init_state = nn.ParameterList(init_state)

    return init_state, _init_state


def enable_cuda(model, arg):
    if is_cuda_enabled(model):
        arg = arg.cuda()
    else:
        arg = arg.cpu()
    return arg


def is_cuda_enabled(model):
    return next(model.parameters()).is_cuda
